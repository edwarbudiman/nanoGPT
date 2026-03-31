# =============================================================================
# Plan A Experiment 10: Very Extended Training (25k iterations)
# =============================================================================
# Hypothesis: Since heavyknorm improved until 14750, push training further
#             with even longer training and very low learning rate.
#
# Changes from heavyknorm (25.12 PPL @ 14750):
#   - max_iters: 15000 -> 25000 (extend significantly)
#   - lr_decay_iters: 15000 -> 25000
#   - learning_rate: 5e-4 -> 3e-4 (very low LR for fine-tuning)
#   - warmup_iters: 500 -> 833 (proportional)
#   - min_lr: 5e-6 -> 1e-6 (allow finer convergence)
#
# Strategy: Very long training with very low LR to squeeze out
#          maximum performance through extensive optimization.
# =============================================================================

out_dir = 'out-plan-a/k4'
eval_interval = 250
eval_iters = 100
log_interval = 25
always_save_checkpoint = False

wandb_log = False
wandb_project = 'rocstories-plan-a'
wandb_run_name = 'k4'

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

# Optimizer - very extended with very low LR
learning_rate = 3e-4  # Very low LR
max_iters = 25000  # Very long training
weight_decay = 1.0  # FIXED, no schedule
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# LR schedule - very extended
decay_lr = True
warmup_iters = 0  # Proportional warmup
lr_decay_iters = 25000
min_lr = 1e-6  # Allow finer convergence

# Weight decay schedule - DISABLED
wd_schedule = False

# Early stopping - very extended patience
early_stop_threshold = 3.15
early_stop_patience = 5000

# System
device = 'cuda'
dtype = 'bfloat16'
compile = True