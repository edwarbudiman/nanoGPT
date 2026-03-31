# =============================================================================
# Plan A Experiment 3: Extended Training with Moderate Regularization
# =============================================================================
# Hypothesis: The model needs more training iterations to converge properly.
# Extended training with moderate regularization should find a better minimum.
#
# Changes from a-combinedheavy:
#   - max_iters: 10000 -> 20000 (double training)
#   - dropout: 0.3 -> 0.2 (moderate reduction)
#   - v_dropout: 0.1 -> 0.05
#   - weight_decay: 0.8 -> 0.5
#   - beta2: 0.95 -> 0.98
#   - min_weight_decay: 0.05 -> 0.1
#   - lr_decay_iters: 10000 -> 20000
#
# Strategy: Train longer, let the model find a better valley in loss landscape.
# =============================================================================

out_dir = 'out-plan-a/a-extended'
eval_interval = 250
eval_iters = 100
log_interval = 25
always_save_checkpoint = False

wandb_log = False
wandb_project = 'rocstories-plan-a'
wandb_run_name = 'a-extended'

dataset = 'rocstories'
gradient_accumulation_steps = 4
batch_size = 32

# Architecture
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2  # Moderate dropout
bias = False
block_size = 256

# A1: QK-Norm with learnable scalar
qk_norm = True
# A2: Value dropout - reduced
v_dropout = 0.05

# Optimizer
learning_rate = 6e-4  # Keep original
max_iters = 20000  # Extended training
weight_decay = 0.5  # Moderate
beta1 = 0.9
beta2 = 0.98  # Smoother updates
grad_clip = 1.0

# A3: Weight decay schedule
wd_schedule = True
min_weight_decay = 0.1

# LR schedule - extended to match training
decay_lr = True
warmup_iters = 1000  # Slightly longer warmup
lr_decay_iters = 20000
min_lr = 1e-5

# Early stopping
early_stop_threshold = 3.15
early_stop_patience = 3000

# System
device = 'cuda'
dtype = 'bfloat16'
compile = True
