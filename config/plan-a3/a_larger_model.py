# =============================================================================
# Plan A Experiment 8: Larger Model (n_embd 512)
# =============================================================================
# Hypothesis: Increase model capacity to capture more complex patterns.
#
# Changes from heavyknorm (25.12 PPL @ 14750):
#   - n_embd: 384 -> 512 (+33% embedding dimension)
#   - n_head: 6 -> 8 (adjust for new embd dim, 512/64=8 heads)
#   - learning_rate: 5e-4 -> 4e-4 (lower for larger model)
#   - max_iters: 15000 -> 20000 (longer training for larger model)
#   - weight_decay: 1.0 (keep same - proven)
#   - dropout: 0.3 (keep same)
#
# Strategy: Larger model has more capacity but needs more regularization
#          and longer training to converge properly.
# =============================================================================

out_dir = 'out-plan-a/k2'
eval_interval = 250
eval_iters = 100
log_interval = 25
always_save_checkpoint = False

wandb_log = False
wandb_project = 'rocstories-plan-a'
wandb_run_name = 'a-larger-model'

dataset = 'rocstories'
gradient_accumulation_steps = 4
batch_size = 32

# Architecture - LARGER
n_layer = 6
n_head = 8  # Increased from 6 (512/64=8)
n_embd = 512  # Increased from 384 (+33%)
dropout = 0.3  # Keep same - proven
bias = False
block_size = 256

# A1: QK-Norm ONLY (proven)
qk_norm = True
v_dropout = 0.0

# Optimizer - adjusted for larger model
learning_rate = 4e-4  # Lower LR for larger model
max_iters = 20000  # Longer training
weight_decay = 1.0  # FIXED, no schedule - proven
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# LR schedule
decay_lr = True
warmup_iters = 0
lr_decay_iters = 20000
min_lr = 5e-6

# Weight decay schedule - DISABLED
wd_schedule = False

# Early stopping - conservative
early_stop_threshold = 3.15
early_stop_patience = 4000

# System
device = 'cuda'
dtype = 'bfloat16'
compile = True