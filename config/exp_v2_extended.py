# ============================================================
# EXPERIMENT V2: Anti-Overfit - EXTENDED
# ============================================================
# Extended from previous run (val loss still decreasing at step 3000)
# Previous: val_loss=3.2272, PPL=25.2
# Continue training to push PPL below 25
# ============================================================

out_dir = 'out-v2-extended'
eval_interval = 250
eval_iters = 100
log_interval = 25
always_save_checkpoint = True

wandb_log = False
wandb_project = 'rocstories-exp'
wandb_run_name = 'v2-extended'

dataset = 'rocstories'
gradient_accumulation_steps = 4
batch_size = 64

# BASELINE ARCHITECTURE (MUST NOT CHANGE)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2  # Same as v2
bias = False

block_size = 256

# Training settings - Same as v2, just more iterations
learning_rate = 6e-4
max_iters = 6000  # EXTENDED from 3000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

decay_lr = True
warmup_iters = 300
lr_decay_iters = 6000  # Match max_iters
min_lr = 6e-5

# GPU settings
device = 'cuda'
dtype = 'bfloat16'
compile = True
