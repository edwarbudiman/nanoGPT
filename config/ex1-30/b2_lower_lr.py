# ============================================================================
# EXPERIMENT B2: Lower Learning Rate + Longer Warmup
# ============================================================================
# Hypothesis: ROCStories might benefit from smoother convergence
# with more careful warmup phase
#
# Key changes from baseline:
# - learning_rate: 1e-3 -> 7e-4 (moderate reduction)
# - warmup_iters: 100 -> 300 (longer warmup)
# - min_lr: 1e-4 -> 7e-5 (proportional reduction)
# ============================================================================

out_dir = 'outs/out-b2-lower-lr'
eval_interval = 250
eval_iters = 100
log_interval = 25
always_save_checkpoint = False

wandb_log = False
wandb_project = 'rocstories-experiments'
wandb_run_name = 'b2-lower-lr'

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

# Context size
block_size = 256

# Training settings - lower LR
learning_rate = 7e-4     # CHANGED: 1e-3 -> 7e-4
max_iters = 5000
weight_decay = 0.1        # Keep baseline
beta1 = 0.9
beta2 = 0.99             # Keep baseline
grad_clip = 1.0

decay_lr = True
warmup_iters = 300       # CHANGED: 100 -> 300
lr_decay_iters = 5000
min_lr = 7e-5            # CHANGED: proportional to LR

# GPU
device = 'cuda'
dtype = 'bfloat16'
compile = True

# Early stopping
early_stop_val_loss = 0.0
early_stop_patience = 3
early_stop_threshold = 3.3
high_freq_eval_interval = 100
