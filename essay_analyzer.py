from tqdm import tqdm
import os
import joblib
from get_logprobs import LMLogProbs
from feature_extractor import get_features
from feature_extractor import split_sentences_max_words
import numpy as np
from scipy.stats import entropy
import torch
import json
from tqdm import tqdm
from itertools import chain


lm_model = {
    "gpt2": LMLogProbs("gpt2-medium")
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "classifiers\\")

sentence_model = joblib.load(MODEL_DIR + "sentence_rf_detector_calibrated-all-data.joblib", mmap_mode="r")
sentence_scaler = joblib.load(MODEL_DIR + "scaler_sentence_rf-all-data.joblib")

meta_model = joblib.load(MODEL_DIR + "meta_classifier-all-data-lf.joblib", mmap_mode="r")
meta_scaler = joblib.load(MODEL_DIR + "scaler_meta-all-data-lf.joblib")

def predict_sentence_probs(text_list):
    feats = []

    for item in text_list:
        f = get_features(item["log_prob"], item["sentence"])
        feats.append(f)

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
            np.median(chunk)
        ])
    return np.vstack(chunk_feats)

def get_essay_features(text):
    num_sentences = len(text)
    num_tokens = sum(len(s.split()) for s in text)
    avg_sent_len = num_tokens / max(1, num_sentences)
    return np.array([num_sentences, num_tokens, avg_sent_len])

# -----------------------------
# Prepare meta features for one essay
# -----------------------------
def prepare_meta_features(sent_results):
    # sentence-level probabilities
    texts = []
    p = []

    for sent in sent_results:
        texts.append(sent["sentence"])
        p.append(sent["prob"])

    p = np.array(p)

    if len(p) < 3:
        return np.zeros(40)


    chunk_feats = get_chunk_features(p)
    agg_chunk_feats = np.concatenate([
        chunk_feats.mean(axis=0),
        chunk_feats.max(axis=0),
        chunk_feats.min(axis=0)
    ])


    dist_feats = np.array([
        p.mean(),
        p.std(),
        p.min(),
        p.max(),
        np.median(p),
        np.percentile(p, 25),
        np.percentile(p, 75),
        entropy(p + 1e-8),
    ])


    dp = np.diff(p)
    dyn_feats = np.array([
        np.mean(np.abs(dp)),     # volatility
        np.std(dp),              # burstiness
        np.max(np.abs(dp)),      # max jump
        np.polyfit(np.arange(len(p)), p, 1)[0],  # slope
    ])


    prop_feats = np.array([
        np.mean(p > 0.9),
        np.mean(p > 0.8),
        np.mean(p < 0.2),
    ])

    third = len(p) // 3
    if third > 0:
        pos_feats = np.array([
            p[:third].mean(),
            p[third:2*third].mean(),
            p[2*third:].mean(),
        ])
    else:
        pos_feats = np.zeros(3)
    



    sent_lens = np.array([len(s.split()) for s in texts])
    struct_feats = np.array([
        sent_lens.mean(),
        sent_lens.std(),
        sent_lens.max(),
        sent_lens.min(),
    ])

    # -----------------------------
    # ESSAY FEATURES (your existing)
    # -----------------------------
    essay_feats = get_essay_features(texts)

    # -----------------------------
    # FINAL META VECTOR
    # -----------------------------
    meta_features = np.concatenate([
        agg_chunk_feats,
        dist_feats,
        dyn_feats,
        prop_feats,
        pos_feats,
        struct_feats,
        essay_feats
    ])

    return meta_features.astype(np.float32)

BATCH_SIZE = 16       # Number of sentences per batch
MAX_TOKENS = 1024     # Max tokens per sentence
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
window = 4

for model_name, lm in lm_model.items():
    if lm.tokenizer.pad_token is None:
        lm.tokenizer.pad_token = lm.tokenizer.eos_token
    lm.model.to(DEVICE)
    lm.model.eval()
    lm.model.half()

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
        with torch.cuda.amp.autocast():
            out = lm.model(input_ids=input_ids, attention_mask=attention_mask)
            log_probs = torch.nn.functional.log_softmax(out.logits, dim=-1)
    
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

def sentence_probs(text):
    sentences = split_sentences_max_words(text)

    results = []

    for s in sentences:
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
        #tokens_pythia, log_probs_pythia = get_batch_token_logprobs_and_tokens(lm_models["pythia"], [c])[0]  

        feats = get_features(log_probs_gpt2, c)
    
        if feats is None:
            continue

        
        feats = feats.reshape(1, -1)   # 🔑 REQUIRED

        #Xc = scaler_chunk.transform(feats)
        #prob = chunk_model.predict_proba(Xc)[0, 1]
#
        #print("Chunk feats:", combined)
        #print("Any NaN:", np.isnan(combined).any())
        #print("Std:", np.std(combined))

        results.append({
            "chunk": c,
            "prob": float(prob),
            "length": len(c.split())
        })

    return results

def predict_essay(text):
    sent_results = sentence_probs(text)


    meta_features = prepare_meta_features(sent_results)
    meta_results = meta_predict(meta_features)

    return {
        "sentence_results": sent_results,
        "meta_results": meta_results
        #"chunks": chunk_results
    }