# =============================================================================
# Plan A Experiment 1: Balanced Light Regularization
# =============================================================================
# Hypothesis: Moderate reduction in regularization will allow better fit
# while maintaining generalization capability.
#
# Changes from a-combinedheavy:
#   - dropout: 0.3 -> 0.15 (reduce by half)
#   - v_dropout: 0.1 -> 0.05 (reduce by half)
#   - weight_decay: 0.8 -> 0.5 (less aggressive)
#   - beta2: 0.95 -> 0.98 (smoother updates for late-stage fine-tuning)
#   - min_weight_decay: 0.05 -> 0.1 (less restrictive late decay)
# =============================================================================

out_dir = 'out-plan-a/a-balanced-light'
eval_interval = 250
eval_iters = 100
log_interval = 25
always_save_checkpoint = False

wandb_log = False
wandb_project = 'rocstories-plan-a'
wandb_run_name = 'a-balanced-light'

dataset = 'rocstories'
gradient_accumulation_steps = 4
batch_size = 32

# Architecture
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.15  # Reduced from 0.3
bias = False
block_size = 256

# A1: QK-Norm with learnable scalar
qk_norm = True
# A2: Value dropout - reduced from 0.1
v_dropout = 0.05

# Optimizer
learning_rate = 5e-4  # Slightly lower than 6e-4
max_iters = 12000
weight_decay = 0.5  # Reduced from 0.8
beta1 = 0.9
beta2 = 0.98  # Smoother updates
grad_clip = 1.0

# A3: Weight decay schedule
wd_schedule = True
min_weight_decay = 0.1  # Less restrictive late decay

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
