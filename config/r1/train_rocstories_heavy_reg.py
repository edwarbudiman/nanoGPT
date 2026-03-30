# =============================================================================
# CONFIG: ROCStories - Heavy Regularization Strategy
# =============================================================================
# Based on analysis of previous runs (w1, w2, w3):
# - Main issue: severe overfitting (train-val gap > 1.0 by step 8000)
# - Best baseline: w2 with weight_decay=0.5 achieved val_loss=3.31
# - Strategy: Aggressive regularization to combat memorization
#
# Key Changes from Baseline:
# - weight_decay: 0.1 -> 1.0 (10x increase for strong L2 regularization)
# - dropout: 0.2 -> 0.3 (50% increase for stronger regularization)
# - min_lr: 6e-5 -> 1e-5 (6x lower for finer convergence)
# - warmup: 750 -> 500 (faster initial learning)
# - Added early stopping configuration
# =============================================================================

out_dir = 'outs/out-rocstories-heavy-reg'
eval_interval = 250
eval_iters = 100
log_interval = 25
always_save_checkpoint = True

wandb_log = False
wandb_project = 'rocstories-nanogpt'
wandb_run_name = 'heavy-reg'

dataset = 'rocstories'
gradient_accumulation_steps = 4
batch_size = 32

# Architecture (baseline - consistent with previous runs)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.3  # 50% increase from baseline 0.2
bias = False

# Context size
block_size = 256

# Training hyperparameters - Heavy Regularization Focus
learning_rate = 6e-4  # Same peak LR
max_iters = 10000  # Extended training with early stopping
weight_decay = 1.0  # 10x increase from baseline 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule - refined for stability
decay_lr = True
warmup_iters = 500  # Reduced from 750 for faster initial learning
lr_decay_iters = 10000  # Match extended max_iters
min_lr = 1e-5  # 6x lower than baseline for finer convergence

# Early stopping configuration
early_stop_patience = 1500  # Stop if no improvement for 1500 steps
early_stop_threshold = 3.25  # Target validation loss

# GPU settings
device = 'cuda'
dtype = 'bfloat16'
compile = True
