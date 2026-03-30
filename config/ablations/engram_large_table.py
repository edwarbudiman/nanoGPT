# =============================================================================
# ABLATION: Engram with larger table (16K entries)
# =============================================================================
# Tests whether more hash table capacity reduces collisions and improves
# n-gram recall. Larger table = fewer hash collisions but more params.
# Engram params: ~2.1M -> total ~32.0M (at the limit)
# =============================================================================

out_dir = 'outs/out-rocstories-engram-16K'
eval_interval = 250
eval_iters = 100
log_interval = 25
always_save_checkpoint = True

wandb_log = False
wandb_project = 'rocstories-nanogpt'
wandb_run_name = 'engram-16K-table'

dataset = 'rocstories'
gradient_accumulation_steps = 4
batch_size = 32

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.3
bias = False
block_size = 256

# Engram — larger table
engram_enabled = True
engram_table_size = 16384  # 16K entries (2x default)
engram_dim = 64
engram_n_hash = 4
engram_layers = (1, 4)

learning_rate = 6e-4
max_iters = 10000
weight_decay = 1.0
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 500
lr_decay_iters = 10000
min_lr = 1e-5

early_stop_patience = 1500
early_stop_threshold = 3.25

device = 'cuda'
dtype = 'bfloat16'
compile = True
