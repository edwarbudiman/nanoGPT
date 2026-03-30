# =============================================================================
# CONFIG: ROCStories - Larger Model Capacity Strategy
# =============================================================================
# Based on analysis of previous runs (w1, w2, w3):
# - Current 29.94M model may be undersized for narrative complexity
# - Larger models can capture more complex patterns with proper regularization
# - Strategy: Increase model capacity with matched regularization
#
# Key Changes from Baseline:
# - n_embd: 384 -> 512 (33% increase in embedding dimension)
# - n_layer: 6 -> 8 (33% increase in depth)
# - weight_decay: 0.1 -> 1.2 (12x increase for larger model)
# - dropout: 0.2 -> 0.35 (75% increase for more capacity)
# - learning_rate: 6e-4 -> 5e-4 (reduced for larger model stability)
# - warmup: 750 -> 800 (slightly longer warmup for larger model)
# =============================================================================

out_dir = 'outs/out-rocstories-large'
eval_interval = 250
eval_iters = 100
log_interval = 25
always_save_checkpoint = True

wandb_log = False
wandb_project = 'rocstories-nanogpt'
wandb_run_name = 'large-model'

dataset = 'rocstories'
gradient_accumulation_steps = 4
batch_size = 32

# Architecture - Increased Capacity
n_layer = 8  # 33% increase from 6
n_head = 8   # Match n_layer increase (512/64 = 8 heads)
n_embd = 512  # 33% increase from 384
dropout = 0.35  # Strong regularization for larger model
bias = False

# Context size
block_size = 256

# Training hyperparameters - Matched to larger model
learning_rate = 5e-4  # Reduced for stability with larger model
max_iters = 10000
weight_decay = 1.2  # Strong regularization for larger capacity
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule
decay_lr = True
warmup_iters = 800  # Slightly longer for larger model
lr_decay_iters = 10000
min_lr = 1e-5

# Early stopping configuration
early_stop_patience = 1500
early_stop_threshold = 3.15  # Target for larger model

# GPU settings
device = 'cuda'
dtype = 'bfloat16'
compile = True
