# =============================================================================
# Plan A Experiment 5: Learning Rate Focus
# =============================================================================
# Hypothesis: Lower learning rate allows finer-grained convergence.
# Combined with moderate regularization for stability.
#
# Changes from a-combinedheavy:
#   - dropout: 0.3 -> 0.15
#   - v_dropout: 0.1 -> 0.05
#   - weight_decay: 0.8 -> 0.4
#   - learning_rate: 6e-4 -> 4e-4 (lower, key change)
#   - beta2: 0.95 -> 0.98
#   - warmup_iters: 500 -> 750 (longer warmup for lower LR)
#   - min_weight_decay: 0.05 -> 0.1
#
# Strategy: Lower LR needs longer training + warmup to be effective.
# =============================================================================

out_dir = 'out-plan-a/a-lr-focus'
eval_interval = 250
eval_iters = 100
log_interval = 25
always_save_checkpoint = False

wandb_log = False
wandb_project = 'rocstories-plan-a'
wandb_run_name = 'a-lr-focus'

dataset = 'rocstories'
gradient_accumulation_steps = 4
batch_size = 32

# Architecture
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.15  # Reduced
bias = False
block_size = 256

# A1: QK-Norm with learnable scalar
qk_norm = True
# A2: Value dropout - reduced
v_dropout = 0.05

# Optimizer
learning_rate = 4e-4  # Lower LR (key change)
max_iters = 15000  # Longer training for lower LR
weight_decay = 0.4  # Moderate
beta1 = 0.9
beta2 = 0.98
grad_clip = 1.0

# A3: Weight decay schedule
wd_schedule = True
min_weight_decay = 0.1

# LR schedule - extended for lower LR
decay_lr = True
warmup_iters = 750  # Longer warmup for lower LR
lr_decay_iters = 15000
min_lr = 5e-6  # Even lower min for fine-tuning

# Early stopping
early_stop_threshold = 3.15
early_stop_patience = 2500

# System
device = 'cuda'
dtype = 'bfloat16'
compile = True
