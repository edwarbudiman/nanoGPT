# ============================================================
# EXPERIMENT: No Dropout + Higher LR
# ============================================================
# Based on Giles Thomas's finding:
# - Best intervention was LR scheduling (0.09 improvement)
# - Removing dropout was second best (0.051 improvement)
#
# Combine both for maximum effect!
#
# Changes:
# - dropout = 0.0 (REMOVED)
# - learning_rate = 6e-4 (higher peak)
# - More warmup for stability
#
# Target: PPL < 25
# ============================================================

out_dir = 'out-exp-no-dropout-highlr'
eval_interval = 250
eval_iters = 100
log_interval = 25
always_save_checkpoint = True

wandb_log = False
wandb_project = 'rocstories-exp'
wandb_run_name = 'exp-no-dropout-highlr'

dataset = 'rocstories'
gradient_accumulation_steps = 4
batch_size = 64

# BASELINE ARCHITECTURE (MUST NOT CHANGE)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.0  # REMOVED
bias = False

block_size = 256

# Training settings - maximum optimization
learning_rate = 6e-4  # Higher LR (but with warmup)
max_iters = 6000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

decay_lr = True
warmup_iters = 500  # Longer warmup for stability
lr_decay_iters = 6000
min_lr = 6e-5

# GPU settings
device = 'cuda'
dtype = 'bfloat16'
compile = True
