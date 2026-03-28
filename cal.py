"""
model_calculation.py — Parameter count and memory estimate for nanoGPT configs.

Usage:
    python model_calculation.py                              # uses default train_rocstories.py
    python model_calculation.py config/train_rocstories.py
    python model_calculation.py --n_layer=8 --n_head=8 --n_embd=512

Reads a config file (same way train.py does), then:
  1. Shows an analytical breakdown of where params come from
  2. Instantiates the real GPT model for ground-truth verification
  3. Estimates VRAM usage
  4. Checks against the 32M assignment limit
"""

import sys
import os
import math

# ── defaults matching GPTConfig + train_rocstories.py ──────────────────────────
# Architecture
n_layer    = 6
n_head     = 6
n_embd     = 384
block_size = 256
dropout    = 0.2
bias       = False

# vocab_size is NOT set in train_rocstories.py → nanoGPT falls back to GPTConfig
# default of 50304 (50257 padded to nearest multiple of 64 for efficiency).
# ROCStories has no meta.pkl (prepare.py removes it), so 50304 is the real value.
vocab_size = 50304

# Assignment limit
PARAM_LIMIT = 32_000_000

# ── load config file and/or CLI overrides (same mechanism as train.py) ─────────
config_file = None
cli_overrides = []

for arg in sys.argv[1:]:
    if not arg.startswith('--') and arg.endswith('.py'):
        config_file = arg
    elif arg.startswith('--'):
        cli_overrides.append(arg)

if config_file:
    print(f"Loading config: {config_file}")
    with open(config_file) as f:
        exec(f.read())          # sets variables in local scope (same as configurator.py)
else:
    print("No config file given — using defaults matching config/train_rocstories.py")

for override in cli_overrides:
    key, _, val = override.lstrip('-').partition('=')
    try:
        exec(f"{key} = {val}")
    except Exception:
        exec(f"{key} = '{val}'")

# ── analytical breakdown ───────────────────────────────────────────────────────
bias_mult = 2 if bias else 1    # weight + bias vs weight only

# Token embedding (wte) — also serves as lm_head due to weight tying
p_wte = vocab_size * n_embd

# Position embedding (wpe) — subtracted from the "reported" count by get_num_params()
p_wpe = block_size * n_embd

# Per transformer block
p_ln1    = n_embd * bias_mult           # LayerNorm 1
p_attn   = n_embd * (3 * n_embd)       # Q, K, V projections (combined c_attn)
p_attn  += n_embd * bias_mult * 3 if bias else 0  # c_attn bias (3*n_embd)
p_cproj  = n_embd * n_embd             # attention output projection
p_cproj += n_embd * bias_mult if bias else 0
p_ln2    = n_embd * bias_mult           # LayerNorm 2
p_mlp_fc = n_embd * (4 * n_embd)       # MLP fc  (n_embd → 4*n_embd)
p_mlp_fc += (4 * n_embd) * bias_mult if bias else 0
p_mlp_pr = (4 * n_embd) * n_embd       # MLP proj (4*n_embd → n_embd)
p_mlp_pr += n_embd * bias_mult if bias else 0

p_block = p_ln1 + p_attn + p_cproj + p_ln2 + p_mlp_fc + p_mlp_pr
p_all_blocks = p_block * n_layer

# Final LayerNorm
p_ln_f = n_embd * bias_mult

# lm_head is tied to wte → 0 extra params
p_lm_head = 0

p_total_raw = p_wte + p_wpe + p_all_blocks + p_ln_f
p_reported  = p_total_raw - p_wpe   # what get_num_params(non_embedding=True) returns

# ── print breakdown ────────────────────────────────────────────────────────────
SEP = "─" * 58

print()
print(SEP)
print("  Config")
print(SEP)
print(f"  n_layer    = {n_layer}")
print(f"  n_head     = {n_head}")
print(f"  n_embd     = {n_embd}")
print(f"  block_size = {block_size}")
print(f"  vocab_size = {vocab_size}  (50257 padded to nearest 64)")
print(f"  bias       = {bias}")
print(f"  dropout    = {dropout}")

print()
print(SEP)
print("  Analytical Parameter Breakdown")
print(SEP)
print(f"  Token embedding  (wte)       {p_wte:>12,}  ({p_wte/1e6:.2f}M)")
print(f"  Position emb.    (wpe)       {p_wpe:>12,}  [{p_wpe/1e6:.2f}M — excluded from reported count]")
print()
print(f"  Per transformer block:       {p_block:>12,}")
print(f"    LayerNorm 1                {p_ln1:>12,}")
print(f"    Attention Q/K/V (c_attn)   {p_attn:>12,}")
print(f"    Attention output (c_proj)  {p_cproj:>12,}")
print(f"    LayerNorm 2                {p_ln2:>12,}")
print(f"    MLP fc  (→ 4×n_embd)       {p_mlp_fc:>12,}")
print(f"    MLP proj (→ n_embd)        {p_mlp_pr:>12,}")
print()
print(f"  {n_layer} blocks total               {p_all_blocks:>12,}  ({p_all_blocks/1e6:.2f}M)")
print(f"  Final LayerNorm  (ln_f)      {p_ln_f:>12,}")
print(f"  LM head (tied to wte)        {p_lm_head:>12,}  [weight tying — 0 extra]")
print()
print(f"  Total (raw, incl. wpe)       {p_total_raw:>12,}  ({p_total_raw/1e6:.2f}M)")
print(f"  Reported (non_embedding=True){p_reported:>12,}  ({p_reported/1e6:.2f}M)")

# ── assignment limit check ─────────────────────────────────────────────────────
print()
print(SEP)
margin = PARAM_LIMIT - p_reported
status = "PASS ✓" if p_reported <= PARAM_LIMIT else "FAIL ✗"
print(f"  Assignment limit: {PARAM_LIMIT/1e6:.0f}M params")
print(f"  Reported params:  {p_reported/1e6:.2f}M")
print(f"  Margin:           {margin/1e6:+.2f}M")
print(f"  Status: {status}")

# ── memory estimate ────────────────────────────────────────────────────────────
# Model weights (float32 = 4 bytes, bfloat16 = 2 bytes)
bytes_f32   = p_total_raw * 4
bytes_bf16  = p_total_raw * 2
# Optimizer (AdamW stores 2 moment tensors per param, float32)
bytes_optim = p_total_raw * 4 * 2
# Activations (rough estimate: batch_size * block_size * n_embd * n_layer * 2 bytes)
batch_size = 64  # from config default; update if changed
bytes_act   = batch_size * block_size * n_embd * n_layer * 2

total_training_bf16 = bytes_bf16 + bytes_optim + bytes_act

print()
print(SEP)
print("  Memory Estimates")
print(SEP)
print(f"  Model weights  (float32)     {bytes_f32/1e9:.3f} GB")
print(f"  Model weights  (bfloat16)    {bytes_bf16/1e9:.3f} GB")
print(f"  AdamW optimizer states       {bytes_optim/1e9:.3f} GB  (2 moments, float32)")
print(f"  Activations    (approx)      {bytes_act/1e9:.3f} GB  (batch={batch_size}, bf16)")
print(f"  Training total (bf16 model)  {total_training_bf16/1e9:.2f} GB  [rough estimate]")
print(f"  → Any GPU with ≥8GB VRAM is comfortable for this config.")

# ── ground-truth verification using actual model ───────────────────────────────
print()
print(SEP)
print("  Ground-Truth: instantiating real GPT model...")
print(SEP)
try:
    import torch
    from model import GPT, GPTConfig

    gpt_config = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        bias=bias,
    )
    # suppress the print inside GPT.__init__ by redirecting stdout briefly
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    model = GPT(gpt_config)
    sys.stdout = old_stdout

    real_total    = sum(p.numel() for p in model.parameters())
    real_reported = model.get_num_params(non_embedding=True)

    print(f"  Real total params:           {real_total:>12,}  ({real_total/1e6:.2f}M)")
    print(f"  Real reported (non_emb=True){real_reported:>12,}  ({real_reported/1e6:.2f}M)")

    match = "✓ match" if real_reported == p_reported else "✗ mismatch — check formula"
    print(f"  Analytical vs real:          {match}")
    limit_ok = "PASS ✓" if real_reported <= PARAM_LIMIT else "FAIL ✗"
    print(f"  Assignment limit check:      {limit_ok}  ({real_reported/1e6:.2f}M ≤ {PARAM_LIMIT/1e6:.0f}M)")

except ImportError:
    print("  (torch not available — skipping model instantiation)")

print(SEP)
print()
