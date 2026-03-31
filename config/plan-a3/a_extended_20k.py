# =============================================================================
# Plan A Experiment 7: Extended Training (20k iterations)
# =============================================================================
# Hypothesis: Since heavyknorm was still improving at 15k, extend training
#             to 20k iterations to capture more improvement.
#
# Changes from heavyknorm (25.12 PPL @ 14750):
#   - max_iters: 15000 -> 20000 (extend by 5k)
#   - lr_decay_iters: 15000 -> 20000 (extend LR decay)
#   - learning_rate: 5e-4 -> 4e-4 (lower for longer training)
#   - warmup_iters: 500 -> 667 (proportional warmup)
#
# Strategy: Extend training horizon with slightly lower LR to allow
#          the model to find better minima.
# =============================================================================

out_dir = 'out-plan-a/k1'
eval_interval = 250
eval_iters = 100
log_interval = 25
always_save_checkpoint = False

wandb_log = False
wandb_project = 'rocstories-plan-a'
wandb_run_name = 'k1'

dataset = 'rocstories'
gradient_accumulation_steps = 4
batch_size = 32

# Architecture - keep same as heavyknorm
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.3
bias = False
block_size = 256

# A1: QK-Norm ONLY (proven)
qk_norm = True
v_dropout = 0.0

# Optimizer - extended training with lower LR
learning_rate = 4e-4  # Lower LR for longer training
max_iters = 30000  # Extended training
weight_decay = 1.0  # FIXED, no schedule
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# LR schedule - extended
decay_lr = True
warmup_iters = 0  # Proportional warmup (500/15000 = 667/20000)
lr_decay_iters = 30000
min_lr = 5e-6

# Weight decay schedule - DISABLED
wd_schedule = False

# Early stopping - extended patience
early_stop_threshold = 3.15
early_stop_patience = 4000  # Increased for longer training

# System
device = 'cuda'
dtype = 'bfloat16'
compile = True