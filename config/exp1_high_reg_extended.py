# ============================================================
# EXPERIMENT 1: High Regularization - EXTENDED
# ============================================================
# Extended from previous run (val loss still decreasing at step 4000)
# Previous: val_loss=3.2338, PPL=25.4
# Continue training to push PPL below 25
# ============================================================

out_dir = 'out-exp1-extended'
eval_interval = 250
eval_iters = 100
log_interval = 25
always_save_checkpoint = True

wandb_log = False
wandb_project = 'rocstories-exp'
wandb_run_name = 'exp1-extended'

dataset = 'rocstories'
gradient_accumulation_steps = 4
batch_size = 64

# BASELINE ARCHITECTURE (MUST NOT CHANGE)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.3  # Same as exp1
bias = False

block_size = 256

# Training settings - Same as exp1, just more iterations
learning_rate = 5e-4
max_iters = 7000  # EXTENDED from 4000
weight_decay = 2e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

decay_lr = True
warmup_iters = 300
lr_decay_iters = 7000  # Match max_iters
min_lr = 5e-5

# GPU settings
device = 'cuda'
dtype = 'bfloat16'
compile = True
