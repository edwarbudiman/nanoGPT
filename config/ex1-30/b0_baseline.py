# ============================================================================
# EXPERIMENT B0: Official Baseline (as-is from nanoGPT)
# ============================================================================
# This is the exact configuration from nanoGPT's train_shakespeare_char.py
# Purpose: Establish baseline performance without any modifications
#
# Key hyperparameters from official baseline:
# - learning_rate = 1e-3 (higher for small networks)
# - warmup_iters = 100 (short warmup)
# - beta2 = 0.99 (stable updates)
# - dropout = 0.2 (moderate regularization)
# ============================================================================

out_dir = 'outs/out-b0-baseline'
eval_interval = 250
eval_iters = 100
log_interval = 25
always_save_checkpoint = False

wandb_log = False
wandb_project = 'rocstories-experiments'
wandb_run_name = 'b0-baseline'

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

# ===== OFFICIAL BASELINE SETTINGS =====
learning_rate = 1e-3      # Official: 1e-3
max_iters = 5000          # Official: 5000
weight_decay = 0.1        # Official: 0.1
beta1 = 0.9               # Official: 0.9
beta2 = 0.99              # Official: 0.99 (KEY DIFFERENCE!)
grad_clip = 1.0

decay_lr = True
warmup_iters = 100        # Official: 100 (KEY DIFFERENCE!)
lr_decay_iters = 5000     # Official: matches max_iters
min_lr = 1e-4             # Official: 1e-4

# GPU
device = 'cuda'
dtype = 'bfloat16'
compile = True

# Early stopping
early_stop_val_loss = 0.0
early_stop_patience = 3
early_stop_threshold = 3.3
high_freq_eval_interval = 100
