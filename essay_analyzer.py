from tqdm import tqdm
import os
import joblib
from get_logprobs import LMLogProbs
from feature_extractor import get_features
from feature_extractor import split_sentences_max_words
import numpy as np
import torch
import json
import sys
from tqdm import tqdm
from itertools import chain




def resource_path(rel_path):
    if hasattr(sys, "_MEIPASS"):
        path = os.path.join(sys._MEIPASS, rel_path)
    else:
        path = os.path.join(os.path.abspath("."), rel_path)
    # convert backslashes to forward slashes for Transformers
    return path.replace("\\", "/")

lm_model = {
    "gpt2": LMLogProbs(resource_path(os.path.join("models","gpt2-medium")))
}

MODEL_DIR = resource_path("classifiers")

# always join paths so we don't accidentally concatenate without a separator
sentence_model = joblib.load(os.path.join(MODEL_DIR, "sentence_rf_detector_calibrated.joblib"))
sentence_scaler = joblib.load(os.path.join(MODEL_DIR, "scaler_sentence_rf.joblib"))

meta_model = joblib.load(os.path.join(MODEL_DIR, "meta_classifier.joblib"))
meta_scaler = joblib.load(os.path.join(MODEL_DIR, "scaler_meta.joblib"))

def predict_sentence_probs(text_list):
    feats = []

    for item in text_list:
        f = get_features(item["log_prob"], item["sentence"])
        f = np.delete(f, [11, 12, 13, 14, 15, 19, 20, 21])
        feats.append(f)

    if len(feats) == 0:
        return np.array([])

    X = np.vstack(feats)
    X_scaled = sentence_scaler.transform(X)
    probs = sentence_model.predict_proba(X_scaled)[:, 1]

    return probs

def meta_predict(meta_features):
    X = np.array(meta_features).reshape(1, -1)     # shape: (n_samples, n_features)
    X_scaled = meta_scaler.transform(X)
    probs = meta_model.predict_proba(X_scaled)[:, 1]
    return probs

def get_chunk_features(sentence_probs, chunk_size=4):
    chunk_feats = []
    for i in range(0, len(sentence_probs), chunk_size):
        chunk = np.array(sentence_probs[i:i+chunk_size])
        chunk_feats.append([
            chunk.mean(),
            chunk.max(),
            chunk.min(),
            chunk.std(),
            np.percentile(chunk, 10),
            np.percentile(chunk, 90),
            np.median(chunk)
        ])
    return np.vstack(chunk_feats)

def get_essay_features(text):
    num_sentences = len(text)
    num_tokens = sum(len(s.split()) for s in text)
    avg_sent_len = num_tokens / max(1, num_sentences)
    return np.array([num_sentences, num_tokens, avg_sent_len])


def prepare_meta_features(texts, progress_callback=None):

    if progress_callback:
        progress_callback("prepare_meta_features")
    # sentence-level probabilities
    sentence_probs = predict_sentence_probs(texts)

    

    p = np.array(sentence_probs)

    if len(p) < 3:
        return np.zeros(14)

    # -----------------------------
    # 🔥 CORE DISTRIBUTION FEATURES
    # -----------------------------
    dist_feats = np.array([
        p.mean(),                         # overall confidence
        p.std(),                          # spread
        np.min(p),                        # weakest sentence
        np.percentile(p, 25),             # lower quartile
        np.percentile(p, 75),             # upper quartile
        entropy(p + 1e-8),                # randomness
    ])

    # -----------------------------
    # 🔥 LOW-PROBABILITY MASS (VERY IMPORTANT)
    # -----------------------------
    prop_feats = np.array([
        np.mean(p < 0.05),                # strongest signal
        np.mean(p < 0.1),
        np.mean(p < 0.2),
    ])

    # -----------------------------
    # 🔥 CHUNK EXTREMES (KEY)
    # -----------------------------
    chunk_feats = get_chunk_features(p)

    chunk_min = chunk_feats.min(axis=0)
    

    chunk_selected = np.concatenate([
        [chunk_min[2]],   # ✅ wrap
        [chunk_min[5]],   # ✅ wrap
    ])

    # -----------------------------
    # 🔥 POSITION SIGNAL
    # -----------------------------
    third = len(p) // 3
    if third > 0:
        pos_feats = np.array([
            p[:third].mean(),
            p[-third:].mean(),  # end_mean (important)
        ])
    else:
        pos_feats = np.zeros(2)

    #prop_grad= np.array([
    #    np.mean(np.diff(p)),
    #    np.std(np.diff(p)),
    #    np.max(np.abs(np.diff(p)))
    #])

    low_mask = (p < 0.1).astype(int)
    max_run = max_consecutive_ones(low_mask)/ len(p)

    # -----------------------------
    # FINAL VECTOR
    # -----------------------------
    meta_features = np.concatenate([
        dist_feats,
        prop_feats,
        chunk_selected,
        pos_feats,
        #prop_grad,
        [max_run]   # wrap scalar
    ])

    return meta_features.astype(np.float32)

def max_consecutive_ones(arr):
    max_run = 0
    current_run = 0

    for x in arr:
        if x == 1:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 0

    return max_run

BATCH_SIZE = 16       
MAX_TOKENS = 1024    
DEVICE = "cpu"
window = 4

for model_name, lm in lm_model.items():
    if lm.tokenizer.pad_token is None:
        lm.tokenizer.pad_token = lm.tokenizer.eos_token
    lm.model.to(DEVICE)
    lm.model.float()
    lm.model.eval()

def get_batch_token_logprobs_and_tokens(lm, texts, max_length=MAX_TOKENS):
    enc = lm.tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    ).to(DEVICE)

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    with torch.no_grad():
        out = lm.model(input_ids=input_ids, attention_mask=attention_mask)
        
        hidden_states = out.last_hidden_state
        logits = hidden_states @ lm.model.wte.weight.T
        
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        token_logps = log_probs.gather(
            2, input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)

    lengths = attention_mask.sum(dim=1) - 1

    results = []
    for i in range(len(texts)):
        L = lengths[i].item()
        toks = lm.tokenizer.convert_ids_to_tokens(input_ids[i, 1:L+1])
        lps = token_logps[i, :L].cpu().numpy()
        results.append((toks, lps))

    return results

def sentence_probs(text, progress_callback=None):
    sentences = split_sentences_max_words(text)

    results = []
    total = len(sentences)

    for i, s in enumerate(sentences):
        if progress_callback:
            progress_callback(f"Processing sentence {i+1}/{total}")
        if len(s) < 5:
            continue

        tokens_gpt2, log_probs_gpt2 = get_batch_token_logprobs_and_tokens(
            lm_model["gpt2"], [s]
        )[0]

        item = {
            "sentence": s,
            "log_prob": log_probs_gpt2
        }

        prob = predict_sentence_probs([item])[0]
        item["prob"] = prob

        results.append(item)

    return results

def chunk_probs(text):
    sentences = split_sentences_max_words(text)

    if len(sentences) <= window:
        chunks = [" ".join(sentences)]
    else:
        chunks = []
        for j in range(0, len(sentences) - window + 1, 1):
            chunks.append(" ".join(sentences[j:j+window]))

    results = []

    for c in chunks:
            
        tokens_gpt2, log_probs_gpt2 = get_batch_token_logprobs_and_tokens(lm_model["gpt2"], [c])[0]

        feats = get_features(log_probs_gpt2, c)
    
        if feats is None:
            continue

        feats = feats.reshape(1, -1)  

        prob = 0.0

        results.append({
            "chunk": c,
            "prob": float(prob),
            "length": len(c.split())
        })

    return results

def predict_essay(text, progress_callback=None):
    sent_results = sentence_probs(text, progress_callback)


    meta_features = prepare_meta_features(sent_results, progress_callback)
    meta_results = meta_predict(meta_features)

    return {
        "sentence_results": sent_results,
        "meta_results": meta_results
        #"chunks": chunk_results
    }

def entropy(p):
    p = np.asarray(p)
    p = p[p > 0]
    return -np.sum(p * np.log(p))