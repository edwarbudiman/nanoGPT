# =============================================================================
# Plan A Experiment 4: Beta2 Focus (Aggressive vs Conservative)
# =============================================================================
# Hypothesis: AdamW beta2 has significant impact on convergence.
# Testing both extremes to find optimal balance.
#
# This config uses more conservative regularization with beta2=0.99.
#
# Changes from a-combinedheavy:
#   - dropout: 0.3 -> 0.25 (slight reduction)
#   - v_dropout: 0.1 -> 0.05
#   - weight_decay: 0.8 -> 0.6
#   - beta2: 0.95 -> 0.99 (conservative/smooth)
#   - min_weight_decay: 0.05 -> 0.15
# =============================================================================

out_dir = 'out-plan-a/a-beta-focus'
eval_interval = 250
eval_iters = 100
log_interval = 25
always_save_checkpoint = False

wandb_log = False
wandb_project = 'rocstories-plan-a'
wandb_run_name = 'a-beta-focus'

dataset = 'rocstories'
gradient_accumulation_steps = 4
batch_size = 32

# Architecture
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.25  # Slight reduction
bias = False
block_size = 256

# A1: QK-Norm with learnable scalar
qk_norm = True
# A2: Value dropout - reduced
v_dropout = 0.05

# Optimizer
learning_rate = 6e-4  # Keep original
max_iters = 12000
weight_decay = 0.6  # Moderate-high
beta1 = 0.9
beta2 = 0.99  # Conservative/smooth updates (key change)
grad_clip = 1.0

# A3: Weight decay schedule
wd_schedule = True
min_weight_decay = 0.15  # Less restrictive

# LR schedule
decay_lr = True
warmup_iters = 500
lr_decay_iters = 12000
min_lr = 1e-5

# Early stopping
early_stop_threshold = 3.15
early_stop_patience = 2000

# System
device = 'cuda'
dtype = 'bfloat16'
compile = True
