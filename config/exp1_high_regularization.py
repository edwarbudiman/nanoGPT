# ============================================================
# EXPERIMENT 1: High Regularization Strategy
# ============================================================
# Reasoning:
# - Previous run showed overfitting after step 1000
# - Increase dropout significantly to force generalization
# - Combine with weight decay for strong regularization
# - Lower learning rate for smoother convergence
#
# Key Differences from Baseline:
# - dropout: 0.1 -> 0.3 (3x increase)
# - learning_rate: 1e-3 -> 5e-4
# - weight_decay: 1e-1 -> 2e-1
# ============================================================

out_dir = 'out-exp1-high-reg'
eval_interval = 250
eval_iters = 100
log_interval = 25
always_save_checkpoint = True

wandb_log = False
wandb_project = 'rocstories-exp'
wandb_run_name = 'exp1-high-regularization'

dataset = 'rocstories'
gradient_accumulation_steps = 4
batch_size = 64

# BASELINE ARCHITECTURE (MUST NOT CHANGE)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.3  # HIGH dropout - aggressive regularization
bias = False

block_size = 256

# Training settings - Regularization focused
learning_rate = 5e-4  # Lower LR for smoother convergence
max_iters = 8000  # Moderate training length
weight_decay = 2e-1  # Stronger weight decay
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

decay_lr = True
warmup_iters = 800
lr_decay_iters = 8000
min_lr = 5e-5

# GPU settings
device = 'cuda'
dtype = 'bfloat16'
compile = True
