# ============================================================================
# PHASE 2A: Low Dropout Finetune
# ============================================================================
# Goal: Resume from Phase 1, try lower dropout
#
# Hypothesis: Lower dropout might help if Phase 1 had too much regularization
#
# Command:
#   python train.py config/phase2a_low_dropout.py \
#       --init_from=resume --out_dir=out-rocstories-phase1
# ============================================================================

out_dir = 'out-rocstories-phase2a'
eval_interval = 100
eval_iters = 100
log_interval = 25
always_save_checkpoint = True

wandb_log = False
wandb_project = 'rocstories-nanogpt'
wandb_run_name = 'phase2a-low-dropout'

dataset = 'rocstories'
gradient_accumulation_steps = 4
batch_size = 32

# ARCHITECTURE (same as Phase 1)
n_layer = 6
n_head = 6
n_embd = 384
bias = False
block_size = 256

# PHASE 2A: Lower dropout - try less regularization
dropout = 0.1  # Down from 0.2
learning_rate = 4e-4  # Slightly lower LR
max_iters = 5000
weight_decay = 0.2
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

# LR Schedule
decay_lr = True
warmup_iters = 200
lr_decay_iters = 5000
min_lr = 6e-5

device = 'cuda'
dtype = 'bfloat16'
compile = True

# Early stopping
early_stop_val_loss = 0.0
early_stop_patience = 3
early_stop_threshold = 3.2
high_freq_eval_interval = 50
