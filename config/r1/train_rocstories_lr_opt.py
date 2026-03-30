# =============================================================================
# CONFIG: ROCStories - Learning Rate Optimization Strategy
# =============================================================================
# Based on analysis of previous runs (w1, w2, w3):
# - Learning rate schedule may be suboptimal for ROCStories
# - Shorter warmup and different decay pattern could help
# - Strategy: Optimize learning rate schedule for faster, more stable convergence
#
# Key Changes from Baseline:
# - learning_rate: 6e-4 -> 1e-3 (67% higher peak LR)
# - warmup: 750 -> 200 (75% shorter warmup)
# - min_lr: 6e-5 -> 1e-5 (6x lower for extended fine-tuning)
# - lr_decay_iters: 8000 -> 6000 (faster decay to minimum)
# - weight_decay: 0.1 -> 0.8 (8x increase for stability)
# - dropout: 0.2 -> 0.25 (moderate increase)
# - beta2: 0.95 -> 0.98 (more stable optimization)
# =============================================================================

out_dir = 'outs/out-rocstories-lr-opt'
eval_interval = 250
eval_iters = 100
log_interval = 25
always_save_checkpoint = True

wandb_log = False
wandb_project = 'rocstories-nanogpt'
wandb_run_name = 'lr-optimized'

dataset = 'rocstories'
gradient_accumulation_steps = 4
batch_size = 32

# Architecture (baseline - consistent with previous runs)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.25  # Moderate increase from baseline 0.2
bias = False

# Context size
block_size = 256

# Training hyperparameters - LR Optimization Focus
learning_rate = 1e-3  # 67% higher peak LR for faster initial convergence
max_iters = 8000
weight_decay = 0.8  # 8x baseline for optimization stability
beta1 = 0.9
beta2 = 0.98  # More stable optimization
grad_clip = 1.0

# Learning rate schedule - Optimized schedule
decay_lr = True
warmup_iters = 200  # Much shorter warmup (reach optimal LR faster)
lr_decay_iters = 6000  # Faster decay to minimum
min_lr = 1e-5  # Much lower minimum for extended fine-tuning

# Early stopping configuration
early_stop_patience = 1000
early_stop_threshold = 3.20

# GPU settings
device = 'cuda'
dtype = 'bfloat16'
compile = True
