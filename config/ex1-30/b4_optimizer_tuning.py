# ============================================================================
# EXPERIMENT B4: Sharper Optimizer (Lower Beta2)
# ============================================================================
# Hypothesis: Lower beta2 (0.95) gives sharper, more responsive updates
# This might help the model adapt faster to story patterns
#
# Key changes from baseline:
# - beta2: 0.99 -> 0.95 (sharper updates)
# - learning_rate: slightly increased to compensate
# ============================================================================

out_dir = 'outs/out-b4-optimizer'
eval_interval = 250
eval_iters = 100
log_interval = 25
always_save_checkpoint = False

wandb_log = False
wandb_project = 'rocstories-experiments'
wandb_run_name = 'b4-optimizer'

# Dataset
dataset = 'rocstories'
gradient_accumulation_steps = 4
batch_size = 32

# ARCHITECTURE (baseline - must not change)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
bias = False

# Context size
block_size = 256

# Training settings - optimizer tuning
learning_rate = 1.2e-3    # CHANGED: slightly higher to compensate for sharper updates
max_iters = 5000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95             # CHANGED: 0.99 -> 0.95 (KEY CHANGE!)
grad_clip = 1.0

decay_lr = True
warmup_iters = 100
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
