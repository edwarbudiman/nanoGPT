# =============================================================================
# Plan A Experiment 6: Heavy + QK-Norm Only (Recommended)
# =============================================================================
# Hypothesis: Keep proven heavy regularization, add QK-Norm benefits only.
#
# Changes from original heavy.log (25.30 PPL):
#   - qk_norm: False -> True (add QK-Norm only)
#   - v_dropout: 0.0 -> 0.0 (keep at 0, don't add v_dropout)
#   - weight_decay: 1.0 -> 1.0 (FIXED, no schedule - key to heavy's success)
#   - dropout: 0.3 -> 0.3 (keep same)
#   - beta2: 0.95 -> 0.95 (keep same)
#   - max_iters: 10000 -> 15000 (extend for potential improvement)
#   - learning_rate: 6e-4 -> 5e-4 (slightly lower for longer training)
#
# Strategy: Don't fix what isn't broken. Add QK-Norm without disrupting
# the proven regularization balance from original heavy config.
# =============================================================================

out_dir = 'out-plan-a/a-heavy-qknorm'
eval_interval = 250
eval_iters = 100
log_interval = 25
always_save_checkpoint = False

wandb_log = False
wandb_project = 'rocstories-plan-a'
wandb_run_name = 'a-heavy-qknorm'

dataset = 'rocstories'
gradient_accumulation_steps = 4
batch_size = 32

# Architecture
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.3  # Keep from heavy - proven optimal
bias = False
block_size = 256

# A1: QK-Norm ONLY (no v_dropout)
qk_norm = True  # Add QK-Norm
v_dropout = 0.0  # Keep at 0 - no value dropout

# Optimizer - keep heavy's proven settings
learning_rate = 5e-4  # Slightly lower for longer training
max_iters = 15000  # Extended training
weight_decay = 1.0  # FIXED, no schedule - KEY to heavy's success
beta1 = 0.9
beta2 = 0.95  # Keep heavy's beta2
grad_clip = 1.0

# Weight decay schedule - DISABLED (critical fix from analysis)
wd_schedule = False
# No min_weight_decay needed when wd_schedule is False

# LR schedule - standard
decay_lr = True
warmup_iters = 500
lr_decay_iters = 15000
min_lr = 5e-6

# Early stopping - conservative
early_stop_threshold = 3.15
early_stop_patience = 3000

# System
device = 'cuda'
dtype = 'bfloat16'
compile = True