# ============================================================================
# EXPERIMENT B3: Larger Context Window
# ============================================================================
# Hypothesis: Stories might benefit from larger context (512 vs 256)
# This allows the model to see more of the story before generating
#
# Key changes from baseline:
# - block_size: 256 -> 512 (larger context)
# - Effective batch size doubles in terms of tokens
# ============================================================================

out_dir = 'outs/out-b3-larger-context'
eval_interval = 250
eval_iters = 100
log_interval = 25
always_save_checkpoint = False

wandb_log = False
wandb_project = 'rocstories-experiments'
wandb_run_name = 'b3-larger-context'

# Dataset
dataset = 'rocstories'
gradient_accumulation_steps = 4
batch_size = 32

# ARCHITECTURE (baseline - must not change)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
bias = False

# Context size - KEY CHANGE
block_size = 512          # CHANGED: 256 -> 512

# Training settings
learning_rate = 1e-3      # Keep baseline
max_iters = 5000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

decay_lr = True
warmup_iters = 100
lr_decay_iters = 5000
min_lr = 1e-4

# GPU
device = 'cuda'
dtype = 'bfloat16'
compile = True

# Early stopping
early_stop_val_loss = 0.0
early_stop_patience = 3
early_stop_threshold = 3.3
high_freq_eval_interval = 100
