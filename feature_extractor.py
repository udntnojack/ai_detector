
import numpy as np
from collections import Counter
import textstat

import re

def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]

def as_vec(x):
    """Force x to be a 1D numpy vector."""
    return np.atleast_1d(np.asarray(x, dtype=np.float64))

def get_features(gpt2_rec, text=0):
    if text != 0:
        gpt2_sent = text
        gpt2_log_prob = gpt2_rec
    else:
        gpt2_sent = gpt2_rec.text
        gpt2_log_prob = gpt2_rec.word_log_probs

    # ---- basic guards ----
    if not isinstance(gpt2_sent, str) or len(gpt2_sent) < 5:
        return np.zeros(1, dtype=np.float64)

    if gpt2_log_prob is None or len(gpt2_log_prob) == 0:
        return np.zeros(1, dtype=np.float64)

    # ---- sentence stats ----
    sentences = split_sentences_max_words(gpt2_sent)
    sent_feats = []
    for s in sentences:
        sf = sentence_stats_features(s)
        if sf is not None:
            sent_feats.append(sf)

    if len(sent_feats) == 0:
        sent_feats = np.zeros(1, dtype=np.float64)
    else:
        sent_feats = np.mean(np.vstack(sent_feats), axis=0)

    # ---- logprob features ----
    gpt2_base = extract_word_features(gpt2_log_prob)
    log_var   = local_variance(gpt2_log_prob)
    readability = readability_features(gpt2_sent)

    # ---- SAFE concatenate ----
    return np.concatenate([
        as_vec(gpt2_base),
        as_vec(readability),
        as_vec(log_var),
        as_vec(sent_feats),
    ])



def local_variance(logps, k=5):
    if len(logps) < k:
        return [0,0,0]
    vars = [np.var(logps[i:i+k]) for i in range(len(logps)-k+1)]
    return np.array([
        np.mean(vars),
        np.std(vars),
        np.max(vars)
    ], dtype=np.float64)



def readability_features(text):
    try:
        vals = [
            textstat.flesch_reading_ease(text),
            textstat.flesch_kincaid_grade(text),
            textstat.gunning_fog(text)
        ]
    except Exception:
        vals = [0.0, 0.0, 0.0]

    return np.asarray(vals, dtype=np.float64)


def split_sentences_max_words(text, max_words=100):
    sentences = split_sentences(text)
    chunks = []
    for sent in sentences:
        words = sent.split()
        if len(words) <= max_words:
            chunks.append(sent)
        else:
            # split long sentence only\n",
            for i in range(0, len(words), max_words):
                chunks.append(" ".join(words[i:i+max_words]))
    return chunks
    
def extract_word_features(word_log_probs):
    log_probs = np.asarray(word_log_probs, dtype=np.float64)

    if log_probs.size == 0:
        return np.zeros(16, dtype=np.float64)

    # clip insane values
    log_probs = np.clip(log_probs, -50, 0)

    probs = np.exp(log_probs)

    # ----- core stats -----
    mean_lp   = np.mean(log_probs)
    var_lp    = np.var(log_probs)
    std_lp    = np.std(log_probs)
    min_lp    = np.min(log_probs)
    max_lp    = np.max(log_probs)
    median_lp = np.median(log_probs)

    q25_lp = np.percentile(log_probs, 25)
    q75_lp = np.percentile(log_probs, 75)
    iqr_lp = q75_lp - q25_lp

    per10_lp = np.percentile(log_probs, 10)
    per90_lp = np.percentile(log_probs, 90)

    skew_val     = skew(log_probs)
    kurtosis_val = kurtosis(log_probs)

    # ----- perplexity -----
    mean_ppl = np.exp(-mean_lp)
    var_ppl  = np.var(np.exp(-log_probs))

    # ----- entropy -----
    entropy = -np.mean(probs * log_probs)

    # ----- normalized dispersion -----
    cv_lp = std_lp / (abs(mean_lp) + 1e-8)

    return np.array([
        mean_lp,
        std_lp,
        var_lp,
        min_lp,
        max_lp,
        median_lp,
        q25_lp,
        q75_lp,
        iqr_lp,
        per10_lp,
        per90_lp,
        skew_val,
        kurtosis_val,
        mean_ppl,
        entropy,
        cv_lp
    ], dtype=np.float64)

def sentence_stats_features(sentence):
    """
    Compute aggregate sentence-level stats for a list of sentences.
    
    Returns:
        [mean_len, std_len, mean_rep, max_rep]
    """
    lengths = []
    rep_counts = []

    word_lengths = []

    
    words = sentence.split()

    if len(words) == 0:
        # fallback for empty input
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    for word in words:
        word_lengths.append(len(word))

    lengths.append(len(words))

    counts = list(Counter(words).values())

    #mean_len = float(np.mean(lengths))
    #std_len = float(np.std(lengths))
    mean_rep = float(np.mean(counts))
    max_rep = float(np.max(counts))
    #word_len = float(np.mean(word_lengths))
    max_len = float(np.max(word_lengths))
    min_len = float(np.min(word_lengths))
    word_std = float(np.std(word_lengths))
    word_var = float(np.var(word_lengths))

    return np.array([mean_rep, max_rep, max_len, min_len, word_std, word_var])



def skew(x):
    x = np.asarray(x)
    mean = np.mean(x)
    std = np.std(x)

    if std == 0:
        return 0.0

    return np.mean(((x - mean) / std) ** 3)

def kurtosis(x):
    x = np.asarray(x)
    mean = np.mean(x)
    std = np.std(x)

    if std == 0:
        return 0.0

    return np.mean(((x - mean) / std) ** 4) - 3