# =============================================================================
# CONFIG: ROCStories - Balanced Regularization Strategy
# =============================================================================
# Based on analysis of previous runs (w1, w2, w3):
# - w2 with weight_decay=0.5 performed best at val_loss=3.31
# - Overfitting became significant after step 4000
# - Strategy: Moderate regularization with optimized learning rate schedule
#
# Key Changes from Baseline:
# - weight_decay: 0.1 -> 0.7 (7x increase, between w1 and w3 optimal)
# - dropout: 0.2 -> 0.25 (moderate increase)
# - learning_rate: 6e-4 -> 8e-4 (33% higher for faster convergence)
# - warmup: 750 -> 600 (slightly faster warmup)
# - lr_decay_iters: 8000 -> 12000 (longer decay for extended training)
# - min_lr: 6e-5 -> 3e-5 (2x lower for finer final convergence)
# =============================================================================

out_dir = 'outs/out-rocstories-balanced'
eval_interval = 250
eval_iters = 100
log_interval = 25
always_save_checkpoint = True

wandb_log = False
wandb_project = 'rocstories-nanogpt'
wandb_run_name = 'balanced'

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

# Training hyperparameters - Balanced Configuration
learning_rate = 8e-4  # 33% higher for faster initial convergence
max_iters = 12000  # Extended training to find optimal point
weight_decay = 0.7  # Moderate increase, 7x baseline
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule - optimized for longer training
decay_lr = True
warmup_iters = 600  # Slightly faster warmup
lr_decay_iters = 12000  # Extended decay schedule
min_lr = 3e-5  # Lower minimum for finer final convergence

# Early stopping configuration
early_stop_patience = 2000  # Allow more patience for gradual improvement
early_stop_threshold = 3.20  # Ambitious target

# GPU settings
device = 'cuda'
dtype = 'bfloat16'
compile = True
