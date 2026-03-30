# =============================================================================
# B. Small Table — 2K entries + embed_dropout 0.3
# =============================================================================
# 4x smaller table forces hash collisions, acting as implicit regularization.
# Different n-grams share entries -> embeddings capture general patterns.
# Params: ~30.25M
# =============================================================================

out_dir = 'outs/plan-b-r2/engram-small-2K'
eval_interval = 250
eval_iters = 100
log_interval = 25
always_save_checkpoint = False

wandb_log = False
wandb_project = 'rocstories-nanogpt'
wandb_run_name = 'engram-small-2K'

dataset = 'rocstories'
gradient_accumulation_steps = 4
batch_size = 32

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.3
bias = False
block_size = 256

engram_enabled = True
engram_table_size = 2048
engram_dim = 64
engram_n_hash = 4
engram_layers = (1, 4)
engram_embed_dropout = 0.3

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
