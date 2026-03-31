# =============================================================================
# Plan A Experiment 9: Longer Context (block_size 384)
# =============================================================================
# Hypothesis: Longer context allows model to understand story better.
#
# Changes from heavyknorm (25.12 PPL @ 14750):
#   - block_size: 256 -> 384 (+50% context)
#   - learning_rate: 5e-4 -> 4e-4 (lower for different data patterns)
#   - max_iters: 15000 -> 20000 (longer training)
#   - batch_size: 32 -> 24 (more memory for longer context)
#
# Strategy: Story understanding benefits from longer context windows.
#          The model can see more of the story at once.
# =============================================================================

out_dir = 'out-plan-a/k3'
eval_interval = 250
eval_iters = 100
log_interval = 25
always_save_checkpoint = False

wandb_log = False
wandb_project = 'rocstories-plan-a'
wandb_run_name = 'k3'

dataset = 'rocstories'
gradient_accumulation_steps = 4
batch_size = 24  # Reduced for longer context (more memory)

# Architecture - longer context
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.3
bias = False
block_size = 384  # Increased from 256 (+50%)

# A1: QK-Norm ONLY (proven)
qk_norm = True
v_dropout = 0.0

# Optimizer
learning_rate = 4e-4  # Slightly lower for different context
max_iters = 20000  # Longer training
weight_decay = 1.0  # FIXED, no schedule
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

# Early stopping
early_stop_threshold = 3.15
early_stop_patience = 4000

# System
device = 'cuda'
dtype = 'bfloat16'
compile = True