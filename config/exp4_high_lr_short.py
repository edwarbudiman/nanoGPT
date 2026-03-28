# ============================================================
# EXPERIMENT 4: Higher LR + Aggressive Early Stop
# ============================================================
# Reasoning:
# - Higher LR for faster initial convergence
# - Short training to prevent overfitting
# - Let model converge quickly, then stop
# - Lower dropout since we're stopping early
#
# Key Differences from Baseline:
# - learning_rate: 1e-3 -> 1.2e-3 (higher)
# - max_iters: 10000 -> 2000 (short)
# - dropout: 0.1 -> 0.1 (keep low since short training)
# ============================================================

out_dir = 'out-exp4-high-lr'
eval_interval = 200
eval_iters = 100
log_interval = 25
always_save_checkpoint = True

wandb_log = False
wandb_project = 'rocstories-exp'
wandb_run_name = 'exp4-high-lr-short'

dataset = 'rocstories'
gradient_accumulation_steps = 4
batch_size = 64

# BASELINE ARCHITECTURE (MUST NOT CHANGE)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1  # Lower dropout - short training
bias = False

block_size = 256

# Training settings - Fast convergence
learning_rate = 1.2e-3  # Higher LR for fast convergence
max_iters = 2000  # Short training - stop early
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

decay_lr = True
warmup_iters = 150  # Short warmup
lr_decay_iters = 2000
min_lr = 1e-4

# GPU settings
device = 'cuda'
dtype = 'bfloat16'
compile = True
