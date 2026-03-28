"""
Compute per-token importance weights based on Pointwise Mutual Information (PMI).

Based on: "Training LLMs Beyond Next Token Prediction — Filling the Mutual
Information Gap" (arXiv:2511.00198).

Tokens that are rare/informative get higher weight in the loss; common filler
tokens (articles, punctuation, whitespace) get lower weight. Weights are
normalised so the mean weight equals 1.0, keeping the loss scale comparable
to the unweighted baseline.

Usage:
    python data/rocstories/compute_token_weights.py

Outputs:
    data/rocstories/token_weights.pt  — tensor of shape (vocab_size,)
"""

import os
import numpy as np
import torch

BASE_DIR = os.path.dirname(__file__)
VOCAB_SIZE = 50257  # GPT-2 BPE
SMOOTHING = 1.0     # Laplace smoothing to avoid log(0)

# Load training tokens
train_path = os.path.join(BASE_DIR, "train.bin")
data = np.memmap(train_path, dtype=np.uint16, mode="r")
print(f"loaded {len(data):,} tokens from {train_path}")

# Count token frequencies
counts = np.bincount(data, minlength=VOCAB_SIZE).astype(np.float64)

# Laplace-smoothed probabilities
counts_smoothed = counts + SMOOTHING
probs = counts_smoothed / counts_smoothed.sum()

# PMI-inspired weight: -log(p(token))
# High for rare tokens, low for common tokens (self-information / surprisal)
weights = -np.log(probs)

# Normalise so mean weight = 1.0 (preserves loss scale)
weights = weights / weights.mean()

# Clamp extremes to avoid instability from ultra-rare tokens
weights = np.clip(weights, 0.1, 10.0)

weights_tensor = torch.tensor(weights, dtype=torch.float32)
out_path = os.path.join(BASE_DIR, "token_weights.pt")
torch.save(weights_tensor, out_path)

print(f"saved token weights to {out_path}")
print(f"  shape: {weights_tensor.shape}")
print(f"  mean:  {weights_tensor.mean().item():.4f}")
print(f"  min:   {weights_tensor.min().item():.4f}")
print(f"  max:   {weights_tensor.max().item():.4f}")

# Show a few example tokens and their weights
try:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    # Sort by weight descending — show top/bottom 10
    sorted_idx = torch.argsort(weights_tensor, descending=True)
    print("\nTop 10 highest-weighted tokens (most informative):")
    for i in sorted_idx[:10]:
        tok = enc.decode([i.item()])
        print(f"  {i.item():5d}  w={weights_tensor[i]:.3f}  '{tok}'")
    print("\nTop 10 lowest-weighted tokens (most common/filler):")
    for i in sorted_idx[-10:]:
        tok = enc.decode([i.item()])
        print(f"  {i.item():5d}  w={weights_tensor[i]:.3f}  '{tok}'")
except ImportError:
    pass