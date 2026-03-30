# ============================================================================
# PHASE 1: Initial Training - Find the best checkpoint
# ============================================================================
# Goal: Train from scratch until we find a good point (before overfitting)
#
# Strategy:
# - Use friend's recommended hyperparameters
# - Train until val_loss stops improving
# - Checkpoint will be saved automatically
#
# Expected outcome:
# - Find optimal stopping point around step 3000-8000
# - Best val_loss typically around 3.2-3.5
#
# Command:
#   python train.py config/phase1_initial_training.py --early_stop_val_loss=3.2
# ============================================================================

out_dir = 'outs/out-w1'
eval_interval = 250
eval_iters = 100
log_interval = 25
always_save_checkpoint = False

wandb_log = False
wandb_project = 'rocstories-nanogpt'
wandb_run_name = 'phase1-initial'

# Dataset
dataset = 'rocstories'
gradient_accumulation_steps = 4  # Effective batch = 128
batch_size = 32

# ARCHITECTURE (baseline - must not change)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
bias = False

# Context size
block_size = 256

learning_rate = 6e-4
max_iters = 8000  # Enough to find optimal point
weight_decay = 0.5
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 750
lr_decay_iters = 8000  # Match max_iters for proper schedule
min_lr = 6e-5

# GPU
device = 'cuda'
dtype = 'bfloat16'
compile = True

# Early stopping - stop when we hit target
early_stop_val_loss = 0.0  # Change to 3.2 to enable early stopping
early_stop_patience = 3
early_stop_threshold = 3.3
high_freq_eval_interval = 100
