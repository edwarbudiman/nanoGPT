# Train nanoGPT on ROCStories dataset
# Following official baseline: n_layer=6, n_head=6, n_embd=384 (~30.2M params)
# Model must stay under 32M parameters (cannot change architecture)

out_dir = 'out-rocstories'
eval_interval = 500
eval_iters = 100
log_interval = 50
always_save_checkpoint = True

wandb_log = False
wandb_project = 'rocstories-nanogpt'
wandb_run_name = 'rocstories-baseline'

# Dataset
dataset = 'rocstories'
gradient_accumulation_steps = 4  # Effective batch = 4 * 64 = 256
batch_size = 64

# BASELINE ARCHITECTURE (MUST NOT CHANGE)
# These are fixed to match official nanoGPT baby GPT model (~30.2M params)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1  # Slightly more regularization
bias = False

# Context size - can adjust (stories are ~100-150 tokens)
block_size = 256  # Longer context for better story coherence

# AdamW optimizer - OPTIMIZED training settings
learning_rate = 1e-3  # Slightly higher for faster convergence
max_iters = 10000  # More training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99  # More stable for longer training
grad_clip = 1.0

# Learning rate schedule - OPTIMIZED
decay_lr = True
warmup_iters = 500  # Proper warmup
lr_decay_iters = 10000
min_lr = 1e-4  # learning_rate / 10

# GPU settings - OPTIMIZED
device = 'cuda'
dtype = 'bfloat16'
compile = True
