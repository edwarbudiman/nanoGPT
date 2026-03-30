# Plan A Combined: All training pipeline improvements (A1 + A2 + A3)
# QK-Norm + Value Dropout + Weight Decay Schedule

out_dir = 'out-plan-a/a-default'
eval_interval = 500
eval_iters = 100
log_interval = 50
always_save_checkpoint = False

wandb_log = False
wandb_project = 'rocstories-plan-a'
wandb_run_name = 'a-combined'

dataset = 'rocstories'
gradient_accumulation_steps = 4
batch_size = 64

# Architecture
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1
bias = False
block_size = 256

# A1: QK-Norm
qk_norm = True
# A2: Value dropout
v_dropout = 0.15

# Optimizer
learning_rate = 1e-3
max_iters = 10000
weight_decay = 0.5
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

# A3: Weight decay schedule
wd_schedule = True
min_weight_decay = 0.01

# LR schedule
decay_lr = True
warmup_iters = 500
lr_decay_iters = 10000
min_lr = 1e-4

# System
device = 'cuda'
dtype = 'bfloat16'
compile = True
