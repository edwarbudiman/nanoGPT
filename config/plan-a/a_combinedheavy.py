# =============================================================================
# Plan A Combined: Task 1 best config (heavy_reg, PPL 24.8) + A1 + A2 + A3
# =============================================================================
# Base: train_rocstories_heavy_reg.py — our best Task 1 result
# Added: QK-Norm (A1), Value Dropout (A2), Weight Decay Schedule (A3)
#
# Reasoning for each choice:
#
# KEPT from heavy_reg (proven to work):
#   - batch_size=32: smaller batches acted as implicit regularization
#   - dropout=0.3: heavy_reg's main weapon against overfitting
#   - beta2=0.95: sharper optimizer updates worked better than 0.99
#   - learning_rate=6e-4: stable peak LR that didn't diverge
#   - min_lr=1e-5: fine-grained convergence in late training
#
# CHANGED with reasoning:
#   - weight_decay 1.0 -> 0.8 start (with A3 cosine schedule to 0.05):
#     Heavy_reg used fixed 1.0 throughout. The schedule lets us start
#     aggressive (preventing early memorization) but relax late so the
#     model can fit finer patterns. Net effect: same early regularization
#     strength but better late-stage convergence.
#   - v_dropout=0.1 (A2, new): heavy_reg already has dropout=0.3 globally.
#     Adding moderate extra dropout specifically on V projections targets
#     value embedding overfitting (autoresearch finding) without
#     over-regularizing the whole model. Kept conservative since base
#     dropout is already high.
#   - qk_norm=True (A1, new): normalizing Q,K stabilizes attention
#     distributions. The learnable scalar lets each head find its own
#     sharpness, which should help with the attention-too-diffuse problem
#     autoresearch identified.
# =============================================================================

out_dir = 'out-plan-a/a-combined'
eval_interval = 250
eval_iters = 100
log_interval = 25
always_save_checkpoint = True

wandb_log = False
wandb_project = 'rocstories-plan-a'
wandb_run_name = 'a-combined'

dataset = 'rocstories'
gradient_accumulation_steps = 4
batch_size = 32  # from heavy_reg — smaller batch = implicit regularization

# Architecture
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.3  # from heavy_reg — proven effective against overfitting
bias = False
block_size = 256

# A1: QK-Norm with learnable scalar
qk_norm = True
# A2: Value dropout — conservative since base dropout is already 0.3
v_dropout = 0.1

# Optimizer — from heavy_reg base
learning_rate = 6e-4
max_iters = 10000
weight_decay = 0.8  # slightly lower start since A3 schedule handles decay
beta1 = 0.9
beta2 = 0.95  # from heavy_reg — sharper updates
grad_clip = 1.0

# A3: Weight decay schedule — cosine from 0.8 down to 0.05
wd_schedule = True
min_weight_decay = 0.05

# LR schedule — from heavy_reg
decay_lr = True
warmup_iters = 500
lr_decay_iters = 10000
min_lr = 1e-5  # from heavy_reg — fine-grained late convergence

# Early stopping
early_stop_threshold = 3.25
early_stop_patience = 1500

# System
device = 'cuda'
dtype = 'bfloat16'
compile = True
