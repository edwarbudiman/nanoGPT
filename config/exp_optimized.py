# ============================================================
# EXPERIMENT: Optimized Combined Settings
# ============================================================
# Best settings from experiments 1 and V2:
# - dropout: 0.25 (between 0.2 and 0.3)
# - learning_rate: 5.5e-4 (between 5e-4 and 6e-4)
# - weight_decay: 1.5e-1 (between 1e-1 and 2e-1)
# - More iterations since val loss was still decreasing
#
# Previous best: PPL ~25.2
# Target: PPL < 25
# ============================================================

out_dir = 'out-optimized'
eval_interval = 250
eval_iters = 100
log_interval = 25
always_save_checkpoint = True

wandb_log = False
wandb_project = 'rocstories-exp'
wandb_run_name = 'optimized-combined'

dataset = 'rocstories'
gradient_accumulation_steps = 4
batch_size = 64

# BASELINE ARCHITECTURE (MUST NOT CHANGE)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.25  # OPTIMIZED: between 0.2 and 0.3
bias = False

block_size = 256

# OPTIMIZED training settings
learning_rate = 5.5e-4  # OPTIMIZED: between 5e-4 and 6e-4
max_iters = 7000  # More iterations - val still decreasing
weight_decay = 1.5e-1  # OPTIMIZED: between 1e-1 and 2e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

decay_lr = True
warmup_iters = 350  # Slightly longer warmup
lr_decay_iters = 7000  # Match max_iters
min_lr = 5.5e-5  # Learning rate / 10

# GPU settings
device = 'cuda'
dtype = 'bfloat16'
compile = True
