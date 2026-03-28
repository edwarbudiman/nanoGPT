# ============================================================
# FINAL RECOMMENDATION CONFIG
# ============================================================
# Based on all experiments + research findings
#
# BEST RESULTS SO FAR:
# - V1 config: PPL ~25.2 with simple \n\n separator
# - Val loss still decreasing at step 3000
# - Gap (train-val) ~0.7 suggests room for improvement
#
# STRATEGY:
# 1. Use simple \n\n separator (proven better for PPL)
# 2. Lower learning rate for better convergence
# 3. More iterations since val loss still decreasing
# 4. Slightly higher weight decay for regularization
#
# TARGET: PPL < 25
# ============================================================

out_dir = 'out-final-recommendation'
eval_interval = 250
eval_iters = 100
log_interval = 25
always_save_checkpoint = True

wandb_log = False
wandb_project = 'rocstories-exp'
wandb_run_name = 'final-recommendation'

dataset = 'rocstories'
gradient_accumulation_steps = 4
batch_size = 64

# BASELINE ARCHITECTURE (MUST NOT CHANGE)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2  # Good balance from experiments
bias = False

block_size = 256

# OPTIMIZED training settings
learning_rate = 4e-4  # Lower LR for better convergence
max_iters = 10000  # More training - val still decreasing
weight_decay = 1.5e-1  # Moderate regularization
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

decay_lr = True
warmup_iters = 500
lr_decay_iters = 10000
min_lr = 4e-5

# GPU settings
device = 'cuda'
dtype = 'bfloat16'
compile = True
