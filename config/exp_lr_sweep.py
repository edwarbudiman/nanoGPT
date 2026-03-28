# ============================================================
# EXPERIMENT: Lower Learning Rate + More Iterations
# ============================================================
# Strategy: Lower LR for better convergence
# Sometimes the last bit of improvement needs slower, more careful updates
#
# Settings:
# - learning_rate: 3e-4 (lower than previous)
# - max_iters: 10000 (more iterations)
# - dropout: 0.25 (moderate regularization)
# ============================================================

out_dir = 'out-exp-lr-sweep'
eval_interval = 250
eval_iters = 100
log_interval = 25
always_save_checkpoint = True

wandb_log = False
wandb_project = 'rocstories-exp'
wandb_run_name = 'lr-sweep-low'

dataset = 'rocstories'
gradient_accumulation_steps = 4
batch_size = 64

# BASELINE ARCHITECTURE (MUST NOT CHANGE)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.25  # Moderate
bias = False

block_size = 256

# Lower LR settings
learning_rate = 3e-4  # LOWER - more careful updates
max_iters = 10000  # More iterations for slow LR
weight_decay = 1.5e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

decay_lr = True
warmup_iters = 500  # Longer warmup for stability
lr_decay_iters = 10000  # Match max_iters
min_lr = 3e-5  # Learning rate / 10

# GPU settings
device = 'cuda'
dtype = 'bfloat16'
compile = True
