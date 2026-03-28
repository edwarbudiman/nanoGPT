# Train nanoGPT on ROCStories dataset - Anti-Overfitting Config
# Following official baseline: n_layer=6, n_head=6, n_embd=384 (~30.2M params)
#
# FIXED ISSUES:
# - Previous config overfit after step 1000 (val loss increased)
# - Lower learning rate for better generalization
# - Stronger dropout to prevent overfitting
# - Shorter training to avoid overfitting

out_dir = 'out-rocstories-v2'
eval_interval = 250  # More frequent evaluation
eval_iters = 100
log_interval = 25
always_save_checkpoint = True

wandb_log = False
wandb_project = 'rocstories-nanogpt'
wandb_run_name = 'rocstories-anti-overfit'

# Dataset
dataset = 'rocstories'
gradient_accumulation_steps = 4  # Effective batch = 256
batch_size = 64

# BASELINE ARCHITECTURE (MUST NOT CHANGE)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2  # INCREASED from 0.1 - stronger regularization
bias = False

# Context size
block_size = 256

# AdamW optimizer - LOWER learning rate for generalization
learning_rate = 6e-4  # REDUCED from 1e-3 - prevents overfitting
max_iters = 6000  # REDUCED - prevent overfitting (best val at ~1000-1500)
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

# Learning rate schedule
decay_lr = True
warmup_iters = 600  # Shorter warmup
lr_decay_iters = 6000  # Match max_iters
min_lr = 6e-5

# GPU settings
device = 'cuda'
dtype = 'bfloat16'
compile = True
