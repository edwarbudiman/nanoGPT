# ============================================================================
# EXPERIMENT B1: Stronger Regularization
# ============================================================================
# Hypothesis: Your previous runs showed overfitting (train loss ~2.0, val ~3.3)
# This config increases regularization to combat overfitting
#
# Key changes from baseline:
# - dropout: 0.2 -> 0.25 (slightly stronger)
# - weight_decay: 0.1 -> 0.15 (moderate increase)
# - Everything else stays the same
# ============================================================================

out_dir = 'outs/out-b1-regularization'
eval_interval = 250
eval_iters = 100
log_interval = 25
always_save_checkpoint = False

wandb_log = False
wandb_project = 'rocstories-experiments'
wandb_run_name = 'b1-regularization'

# Dataset
dataset = 'rocstories'
gradient_accumulation_steps = 4
batch_size = 32

# ARCHITECTURE (baseline - must not change)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.25           # CHANGED: 0.2 -> 0.25
bias = False

# Context size
block_size = 256

# Training settings - stronger regularization
learning_rate = 1e-3      # Keep baseline
max_iters = 5000
weight_decay = 0.15      # CHANGED: 0.1 -> 0.15
beta1 = 0.9
beta2 = 0.99             # Keep baseline for stability
grad_clip = 1.0

decay_lr = True
warmup_iters = 100       # Keep baseline
lr_decay_iters = 5000
min_lr = 1e-4

# GPU
device = 'cuda'
dtype = 'bfloat16'
compile = True

# Early stopping
early_stop_val_loss = 0.0
early_stop_patience = 3
early_stop_threshold = 3.3
high_freq_eval_interval = 100
