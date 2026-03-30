# =============================================================================
# CONFIG: ROCStories - Engram (both layers 1 and 4)
# =============================================================================
# Based on optimal Task 1 config (heavy_reg: PPL ~24.8)
# Adds Engram n-gram memory module after layers 1 and 4
# Engram params: ~1.1M -> total ~31.0M (under 32M limit)
# =============================================================================

out_dir = 'outs/out-rocstories-engram-both'
eval_interval = 250
eval_iters = 100
log_interval = 25
always_save_checkpoint = True

wandb_log = False
wandb_project = 'rocstories-nanogpt'
wandb_run_name = 'engram-both-1-4'

dataset = 'rocstories'
gradient_accumulation_steps = 4
batch_size = 32

# Architecture (same as optimal Task 1 baseline)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.3
bias = False
block_size = 256

# Engram module — n-gram memory after layers 1 and 4
engram_enabled = True
engram_table_size = 8192
engram_dim = 64
engram_n_hash = 4
engram_layers = (1, 4)

# Training hyperparameters (proven optimal from Task 1)
learning_rate = 6e-4
max_iters = 10000
weight_decay = 1.0
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# LR schedule
decay_lr = True
warmup_iters = 500
lr_decay_iters = 10000
min_lr = 1e-5

# Early stopping
early_stop_patience = 1500
early_stop_threshold = 3.25

# GPU
device = 'cuda'
dtype = 'bfloat16'
compile = True
