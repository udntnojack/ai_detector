import re
import torch
import numpy as np
import os

# Prevent transformers from importing every architecture
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from transformers import GPT2TokenizerFast
from transformers import GPT2LMHeadModel

class LMLogProbs:
    def __init__(self, model_path="distilgpt2"):
        self.device = "cpu"
        model_path = os.path.abspath(model_path).replace("\\", "/")

        self.tokenizer = GPT2TokenizerFast.from_pretrained(
            model_path,
            local_files_only=True
        )

        self.model = GPT2LMHeadModel.from_pretrained(
            model_path,
            local_files_only=True
        ).to(self.device)

        self.model.eval()
        torch.set_grad_enabled(False)

    def tokenize(self, text):
        return self.tokenizer(text, return_tensors="pt")["input_ids"].to(self.device) 

    def get_log_probs(self, text): 
        with torch.no_grad(): 
            outputs = self.model(text) 
            logits = outputs.logits 
            shift_logits = logits[:, :-1, :] 
            shift_labels = text[:, 1:] 
            log_probs = torch.log_softmax(shift_logits, dim=-1) 
            token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1) 
            return token_log_probs.cpu().numpy().flatten()
    
    
    def get_chunk_intra_sentence_variance(self, log_probs, sentence_lengths):
        """
        Computes intra-sentence token variance for a chunk.

        log_probs: np.array of token log-probs for the chunk
        sentence_lengths: list of token lengths per sentence in this chunk
        """
        if len(sentence_lengths) == 0:
            return 0.0

        idx = 0
        vars_ = []

        for n_tokens in sentence_lengths:
            if idx + n_tokens <= len(log_probs) and n_tokens > 1:
                v = np.var(log_probs[idx:idx+n_tokens])
                vars_.append(v)
                idx += n_tokens

        if len(vars_) == 0:
            return 0.0

        raw_mean_var = np.mean(vars_)

        # stabilize large variations
        return np.log1p(raw_mean_var)


    def get_burstiness(self, log_probs, window_size=5):
        """
        token_ids: torch tensor of shape [1, seq_len] or [seq_len]
        log_probs: 1D numpy array of per-token log-probs for this chunk
        window_size: number of tokens per window to compute local mean
        """
        lp = log_probs.flatten()
        n = len(lp)
        if n < 2:
            return 0.0

        # sliding window means
        means = []
        for start in range(0, n, window_size):
            end = min(start + window_size, n)
            window = lp[start:end]
            if len(window) > 0:
                means.append(np.mean(window))

        return float(np.std(means)) if len(means) > 1 else 0.0
    

    

def cross_model_disagreement(lp_a, lp_b):
    if len(lp_a) == 0 or len(lp_b) == 0:
        return 0.0

    # match lengths
    n = min(len(lp_a), len(lp_b))
    a = lp_a[:n]
    b = lp_b[:n]

    # normalize to comparable scale
    a = (a - np.mean(a)) / (np.std(a) + 1e-8)
    b = (b - np.mean(b)) / (np.std(b) + 1e-8)

    # disagreement = average absolute difference
    return float(np.mean(np.abs(a - b)))

    def get_repeating_words(self, text):
        words = text.split()
        if not words:
            return 0.0
        seen = set()
        repeats = 0
        for w in words:
            if w in seen:
                repeats += 1
            else:
                seen.add(w)
        return repeats / len(words) if words else 0.0

    def get_sentence_length(self, text):
        sentences = re.split(r'[.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0.0
        lengths = [len(s.split()) for s in sentences]
        return np.mean(lengths)