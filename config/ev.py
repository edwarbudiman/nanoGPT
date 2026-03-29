# Train nanoGPT on ROCStories - Friend's Config + Early Stopping
# Based on friend's optimized hyperparameters
#
# KEY IMPROVEMENTS from friend:
# - Longer warmup (750 vs 300)
# - Proper LR decay iters = max_iters (crucial!)
# - Lower min_lr for better convergence
# - Higher weight_decay for regularization
#
# Architecture: n_layer=6, n_head=6, n_embd=384 (~30.2M params, under 32M)

out_dir = 'out-rocstories-friend'
eval_interval = 250  # Frequent evaluation
eval_iters = 100
log_interval = 25
always_save_checkpoint = True

wandb_log = False
wandb_project = 'rocstories-nanogpt'
wandb_run_name = 'rocstories-friend'

# Dataset
dataset = 'rocstories'
gradient_accumulation_steps = 4  # Effective batch = 128
batch_size = 32

# BASELINE ARCHITECTURE (MUST NOT CHANGE per assignment)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
bias = False

# Context size
block_size = 256

# AdamW optimizer - FRIEND'S OPTIMIZED SETTINGS
learning_rate = 6e-4
max_iters = 15000  # Friend's recommendation
weight_decay = 0.2  # Friend's higher weight decay
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

# Learning rate schedule - FRIEND'S KEY FIX
decay_lr = True
warmup_iters = 750  # Friend's longer warmup
lr_decay_iters = 15000  # MUST equal max_iters for proper LR schedule!
min_lr = 6e-5  # Friend's lower min_lr

# GPU settings
device = 'cuda'
dtype = 'bfloat16'
compile = True

# EARLY STOPPING CONFIG (optional - enable via CLI)
# Run: python train.py config/train_rocstories_friend.py --early_stop_val_loss=3.2
# early_stop_val_loss = 0.0  # 0.0 = disabled, set > 0 to enable
# early_stop_patience = 3
# early_stop_threshold = 3.3  # Switch to high_freq when below this
# high_freq_eval_interval = 100  # Eval every 100 steps when close to target
