# ============================================================================
# PHASE 3: Careful Extension from Best Phase 2
# ============================================================================
# Goal: Extend from best Phase 2 variant with very conservative settings
#
# When to use:
# - After finding best Phase 2 variant (2a, 2b, 2c, or 2d)
# - Extend training with lower LR and more regularization
#
# IMPORTANT: Change out_dir to match your best Phase 2 run
# Example: --out_dir=out-rocstories-phase2b
#
# Command:
#   python train.py config/phase3_careful_extend.py \
#       --init_from=resume --out_dir=out-rocstories-phase2b
# ============================================================================

out_dir = 'out-rocstories-phase2b'  # CHANGE THIS to your best Phase 2 result
eval_interval = 100
eval_iters = 100
log_interval = 25
always_save_checkpoint = True

wandb_log = False
wandb_project = 'rocstories-nanogpt'
wandb_run_name = 'phase3-careful'

dataset = 'rocstories'
gradient_accumulation_steps = 4
batch_size = 32

# ARCHITECTURE (same as Phase 1)
n_layer = 6
n_head = 6
n_embd = 384
bias = False
block_size = 256

# PHASE 3: Very conservative settings for final polish
dropout = 0.1  # Lower dropout for final tuning
learning_rate = 2e-4  # Very low LR for careful extension
max_iters = 3000  # Short extension
weight_decay = 0.2
beta1 = 0.9
beta2 = 0.99
grad_clip = 0.5  # Lower gradient clipping for stability

# LR Schedule - Very slow decay
decay_lr = True
warmup_iters = 100  # Minimal warmup (already warmed)
lr_decay_iters = 3000
min_lr = 1e-5  # Lower min_lr for smoother decay

device = 'cuda'
dtype = 'bfloat16'
compile = True

# Early stopping
early_stop_val_loss = 0.0
early_stop_patience = 3
early_stop_threshold = 3.15  # Tighter target
high_freq_eval_interval = 50
