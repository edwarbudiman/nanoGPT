# ============================================================
# EXPERIMENT 2: Large Batch + Low LR Strategy
# ============================================================
# Reasoning:
# - Larger effective batch sizes lead to more stable gradients
# - Lower learning rate with larger batch prevents oscillation
# - More gradient updates per epoch = smoother convergence
# - Extended training for better generalization
#
# Key Differences from Baseline:
# - gradient_accumulation_steps: 4 -> 8 (batch 512)
# - learning_rate: 1e-3 -> 4e-4
# - max_iters: 10000 -> 6000
# ============================================================

out_dir = 'out-exp2-large-batch'
eval_interval = 250
eval_iters = 100
log_interval = 25
always_save_checkpoint = True

wandb_log = False
wandb_project = 'rocstories-exp'
wandb_run_name = 'exp2-large-batch'

dataset = 'rocstories'
gradient_accumulation_steps = 8  # Larger effective batch
batch_size = 64

# BASELINE ARCHITECTURE (MUST NOT CHANGE)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.15  # Moderate dropout
bias = False

block_size = 256

# Training settings - Large batch focused
learning_rate = 4e-4  # Lower LR for larger batch
max_iters = 6000  # Extended training
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

decay_lr = True
warmup_iters = 400  # Longer warmup for stability
lr_decay_iters = 6000
min_lr = 4e-5

# GPU settings
device = 'cuda'
dtype = 'bfloat16'
compile = True
