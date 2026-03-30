# Train nanoGPT + Engram on ROCStories
# Based on train_rocstories.py baseline with Engram n-gram memory enabled
# Engram adds ~1.1M params -> total ~31.3M (under 32M limit)

out_dir = 'out-rocstories-engram'
eval_interval = 500
eval_iters = 100
log_interval = 50
always_save_checkpoint = False

wandb_log = False
wandb_project = 'rocstories-nanogpt'
wandb_run_name = 'rocstories-engram'

# Dataset
dataset = 'rocstories'
gradient_accumulation_steps = 4  # Effective batch = 4 * 64 = 256
batch_size = 64

# Architecture (same baseline)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1
bias = False
block_size = 256

# Engram module — n-gram memory lookup after layers 1 and 4
engram_enabled = True
engram_table_size = 8192  # 8K entries in hash table
engram_dim = 64           # embedding dim per entry (projected to 384)
engram_n_hash = 4         # 4 independent hash functions to reduce collisions
engram_layers = (1, 4)    # insert after transformer layers 1 and 4

# Optimizer (same as baseline)
learning_rate = 1e-3
max_iters = 10000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

# LR schedule
decay_lr = True
warmup_iters = 500
lr_decay_iters = 10000
min_lr = 1e-4

# GPU
device = 'cuda'
dtype = 'bfloat16'
compile = True
