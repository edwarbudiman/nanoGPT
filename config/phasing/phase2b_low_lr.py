# ============================================================================
# PHASE 2B: Low Learning Rate Finetune
# ============================================================================
# Goal: Resume from Phase 1, try significantly lower LR
#
# Hypothesis: Lower LR for more careful fine-tuning
#
# Command:
#   python train.py config/phase2b_low_lr.py \
#       --init_from=resume --out_dir=out-rocstories-phase1
# ============================================================================

out_dir = 'out-rocstories-phase2b'
eval_interval = 100
eval_iters = 100
log_interval = 25
always_save_checkpoint = False

wandb_log = False
wandb_project = 'rocstories-nanogpt'
wandb_run_name = 'phase2b-low-lr'

dataset = 'rocstories'
gradient_accumulation_steps = 4
batch_size = 32

# ARCHITECTURE (same as Phase 1)
n_layer = 6
n_head = 6
n_embd = 384
bias = False
block_size = 256

# PHASE 2B: Lower LR for careful fine-tuning
dropout = 0.15  # Slightly lower
learning_rate = 3e-4  # Half of Phase 1's LR
max_iters = 5000
weight_decay = 0.2
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

# LR Schedule
decay_lr = True
warmup_iters = 150  # Shorter warmup
lr_decay_iters = 5000
min_lr = 3e-5  # Lower min_lr for longer decay

device = 'cuda'
dtype = 'bfloat16'
compile = True

# Early stopping
early_stop_val_loss = 0.0
early_stop_patience = 3
early_stop_threshold = 3.2
high_freq_eval_interval = 50
