# =============================================================================
# Plan A Experiment 2: Very Light Regularization
# =============================================================================
# Hypothesis: Minimal regularization to maximize fitting capability.
# The QK-Norm alone provides sufficient stabilization.
#
# Changes from a-combinedheavy:
#   - dropout: 0.3 -> 0.1 (minimal dropout)
#   - v_dropout: 0.1 -> 0.0 (remove value dropout)
#   - weight_decay: 0.8 -> 0.3 (very light decay)
#   - beta2: 0.95 -> 0.99 (smoothest updates)
#   - min_weight_decay: 0.05 -> 0.05
#   - learning_rate: 6e-4 -> 5e-4 (compensate for less regularization)
#
# Risk: May overfit faster. Monitor train-val gap closely.
# =============================================================================

out_dir = 'out-plan-a/a-very-light'
eval_interval = 250
eval_iters = 100
log_interval = 25
always_save_checkpoint = False

wandb_log = False
wandb_project = 'rocstories-plan-a'
wandb_run_name = 'a-very-light'

dataset = 'rocstories'
gradient_accumulation_steps = 4
batch_size = 32

# Architecture
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1  # Minimal dropout
bias = False
block_size = 256

# A1: QK-Norm with learnable scalar (main stability source)
qk_norm = True
# A2: No extra value dropout
v_dropout = 0.0

# Optimizer
learning_rate = 5e-4
max_iters = 15000  # Longer training to compensate
weight_decay = 0.3  # Very light decay
beta1 = 0.9
beta2 = 0.99  # Smoothest updates
grad_clip = 1.0

# A3: Weight decay schedule
wd_schedule = True
min_weight_decay = 0.05

# LR schedule
decay_lr = True
warmup_iters = 500
lr_decay_iters = 15000
min_lr = 1e-5

# Early stopping - tighter threshold since we're less regularized
early_stop_threshold = 3.1
early_stop_patience = 1500

# System
device = 'cuda'
dtype = 'bfloat16'
compile = True
