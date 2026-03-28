# ============================================================
# EXPERIMENT 3: Smaller Context (ROCStories Specific)
# ============================================================
# Reasoning:
# - ROCStories are 5 sentences, ~100-150 tokens each
# - block_size=128 is sufficient for single story context
# - Smaller context = model focuses on local story coherence
# - Less overfitting risk with shorter context window
#
# Key Differences from Baseline:
# - block_size: 256 -> 128 (matches story length)
# - learning_rate: 1e-3 -> 6e-4
# - More updates per epoch
# ============================================================

out_dir = 'out-exp3-small-block'
eval_interval = 250
eval_iters = 100
log_interval = 25
always_save_checkpoint = True

wandb_log = False
wandb_project = 'rocstories-exp'
wandb_run_name = 'exp3-small-context'

dataset = 'rocstories'
gradient_accumulation_steps = 4
batch_size = 64

# BASELINE ARCHITECTURE (MUST NOT CHANGE)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.15
bias = False

block_size = 128  # Smaller - matches story length

# Training settings
learning_rate = 6e-4  # Good for this model size
max_iters = 5000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

decay_lr = True
warmup_iters = 300
lr_decay_iters = 5000
min_lr = 6e-5

# GPU settings
device = 'cuda'
dtype = 'bfloat16'
compile = True
