# =============================================================================
# CONFIG: ROCStories - Stochastic Weight Averaging (SWA) Strategy
# =============================================================================
# Based on analysis of previous runs (w1, w2, w3):
# - Validation loss plateaus around step 3000-4000
# - Model could benefit from weight averaging in plateau region
# - Strategy: Extended training with SWA-like approach using checkpoint averaging
#
# Key Changes from Baseline:
# - weight_decay: 0.1 -> 0.6 (6x increase)
# - dropout: 0.2 -> 0.25 (moderate increase)
# - max_iters: 8000 -> 15000 (extended for SWA-style averaging)
# - warmup: 750 -> 400 (moderate warmup)
# - lr_decay_iters: 8000 -> 10000 (longer decay to maintain learning)
# - min_lr: 6e-5 -> 5e-6 (10x lower for ultra-fine convergence)
# - always_save_checkpoint: True (save all checkpoints for later averaging)
# =============================================================================

out_dir = 'outs/out-rocstories-swa'
eval_interval = 250
eval_iters = 100
log_interval = 25
always_save_checkpoint = True  # Critical for SWA-style checkpoint averaging

wandb_log = False
wandb_project = 'rocstories-nanogpt'
wandb_run_name = 'swa-style'

dataset = 'rocstories'
gradient_accumulation_steps = 4
batch_size = 32

# Architecture (baseline - consistent with previous runs)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.25  # Moderate increase from baseline 0.2
bias = False

# Context size
block_size = 256

# Training hyperparameters - SWA Focus
learning_rate = 6e-4  # Same as baseline
max_iters = 15000  # Extended training for SWA-style averaging
weight_decay = 0.6  # 6x baseline
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule - Extended for SWA
decay_lr = True
warmup_iters = 400  # Moderate warmup
lr_decay_iters = 10000  # Longer decay schedule
min_lr = 5e-6  # Ultra-low minimum for fine convergence

# Early stopping configuration
early_stop_patience = 3000  # Very patient (allow extended training)
early_stop_threshold = 3.15  # Ambitious target

# GPU settings
device = 'cuda'
dtype = 'bfloat16'
compile = True
