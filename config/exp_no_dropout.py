# ============================================================
# EXPERIMENT: Research-Based Optimized Config
# ============================================================
# Based on Giles Thomas's experiments:
# https://www.gilesthomas.com
#
# KEY FINDINGS:
# 1. Removing dropout improved test loss by 0.051 (3x gradient clipping)
# 2. Learning rate scheduling was most impactful (0.09 improvement)
# 3. Cosine decay with warmup is recommended
#
# CHANGES:
# - dropout = 0.0 (REMOVED - research shows this helps!)
# - Lower LR for stability with no dropout
# - Longer training since model learns faster without dropout
#
# Target: PPL < 25
# ============================================================

out_dir = 'out-exp-no-dropout'
eval_interval = 250
eval_iters = 100
log_interval = 25
always_save_checkpoint = True

wandb_log = False
wandb_project = 'rocstories-exp'
wandb_run_name = 'exp-no-dropout'

dataset = 'rocstories'
gradient_accumulation_steps = 4
batch_size = 64

# BASELINE ARCHITECTURE (MUST NOT CHANGE)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.0  # REMOVED based on research!
bias = False

block_size = 256

# Training settings - optimized
learning_rate = 4e-4  # Lower LR for stability without dropout
max_iters = 6000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

decay_lr = True
warmup_iters = 300
lr_decay_iters = 6000
min_lr = 4e-5

# GPU settings
device = 'cuda'
dtype = 'bfloat16'
compile = True
