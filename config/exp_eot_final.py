# ============================================================
# EXPERIMENT: EOT Token Optimized (Final)
# ============================================================
# Dataset: Each story ends with <|endoftext|> token
# Goal: Model learns story endings + generates proper 5-sentence stories
#
# Key insights for EOT learning:
# - EOT is a rare token (1 per story ~98K stories = 98K EOT)
# - Need more iterations to learn EOT well
# - Lower dropout might help (EOT is hard to learn with high dropout)
# - Continue training longer
#
# Previous results: PPL ~27 with default settings
# Target: PPL < 25 with optimized settings
# ============================================================

out_dir = 'out-exp-eot-final'
eval_interval = 250
eval_iters = 100
log_interval = 25
always_save_checkpoint = True

wandb_log = False
wandb_project = 'rocstories-exp'
wandb_run_name = 'exp-eot-final'

dataset = 'rocstories'
gradient_accumulation_steps = 4
batch_size = 64

# BASELINE ARCHITECTURE (MUST NOT CHANGE)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2  # Lower dropout - EOT is rare, needs less regularization
bias = False

block_size = 256

# Training settings - OPTIMIZED for EOT learning
learning_rate = 6e-4  # Good balance
max_iters = 10000  # More iterations for EOT learning
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

decay_lr = True
warmup_iters = 500  # Longer warmup for EOT stability
lr_decay_iters = 10000  # Match max_iters
min_lr = 6e-5

# GPU settings
device = 'cuda'
dtype = 'bfloat16'
compile = True
