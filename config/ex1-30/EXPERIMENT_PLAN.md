# ROCStories Hyperparameter Tuning - Experimental Plan

## Objective
Systematically find optimal hyperparameters for training nanoGPT on ROCStories dataset, following the official baseline configuration as a starting point.

## Constraints (per assignment)
- Model: n_layer=6, n_head=6, n_embd=384 (~30M parameters)
- No pretrained models - train from scratch
- Max 32M parameters

---

## Experimental Design

### B0 - Baseline (Official nanoGPT config)
**Config:** `config/experiment/b0_baseline.py`

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| learning_rate | 1e-3 | Official baseline - higher for small networks |
| warmup_iters | 100 | Official baseline - short warmup |
| beta2 | 0.99 | Official baseline - stable updates |
| dropout | 0.2 | Official baseline |
| weight_decay | 0.1 | Official baseline |
| block_size | 256 | Official baseline |

**Expected:** Establish reference performance. May show overfitting similar to previous runs.

---

### B1 - Stronger Regularization
**Config:** `config/experiment/b1_regularization.py`

| Parameter | Value | Change from B0 |
|-----------|-------|----------------|
| dropout | 0.25 | +0.05 |
| weight_decay | 0.15 | +0.05 |

**Hypothesis:** Previous runs showed train loss ~2.0 vs val loss ~3.3 (gap of 1.3). Increasing regularization should reduce this gap and potentially improve validation loss.

**Expected:** Lower val loss than B0, smaller train-val gap, but possibly higher train loss.

---

### B2 - Lower Learning Rate
**Config:** `config/experiment/b2_lower_lr.py`

| Parameter | Value | Change from B0 |
|-----------|-------|----------------|
| learning_rate | 7e-4 | -3e-4 |
| warmup_iters | 300 | +200 |
| min_lr | 7e-5 | proportional |

**Hypothesis:** ROCStories is a different domain than Shakespeare. Lower LR with longer warmup may provide more stable convergence for story-like structures.

**Expected:** Slower initial progress, but potentially better final performance and less fluctuation in val loss.

---

### B3 - Larger Context Window
**Config:** `config/experiment/b3_larger_context.py`

| Parameter | Value | Change from B0 |
|-----------|-------|----------------|
| block_size | 512 | +256 |

**Hypothesis:** Stories have narrative structure. Larger context (512 vs 256 tokens) allows the model to see more of the story before generating, potentially capturing better story coherence.

**Expected:** More memory usage, potentially better story completion performance. Val loss may be lower if narrative understanding improves.

**Note:** This doubles the effective receptive field.

---

### B4 - Sharper Optimizer (Lower Beta2)
**Config:** `config/experiment/b4_optimizer_tuning.py`

| Parameter | Value | Change from B0 |
|-----------|-------|----------------|
| beta2 | 0.95 | -0.04 |
| learning_rate | 1.2e-3 | +2e-4 |

**Hypothesis:** Lower beta2 (0.95 vs 0.99) gives sharper, more responsive gradient updates. Combined with slightly higher LR, this may help the model adapt faster to story patterns.

**Expected:** More responsive training, potentially faster initial convergence, but may be less stable.

---

## Running the Experiments

### Parallel Training Commands

```bash
# Terminal 1 - Baseline B0
python train.py config/experiment/b0_baseline.py 2>&1 | tee train-b0.log

# Terminal 2 - Regularization B1
python train.py config/experiment/b1_regularization.py 2>&1 | tee train-b1.log

# Terminal 3 - Lower LR B2
python train.py config/experiment/b2_lower_lr.py 2>&1 | tee train-b2.log

# Terminal 4 - Larger Context B3
python train.py config/experiment/b3_larger_context.py 2>&1 | tee train-b3.log

# Terminal 5 - Optimizer B4
python train.py config/experiment/b4_optimizer_tuning.py 2>&1 | tee train-b4.log
```

### Alternative: Background Training
```bash
nohup python train.py config/experiment/b0_baseline.py > train-b0.log 2>&1 &
nohup python train.py config/experiment/b1_regularization.py > train-b1.log 2>&1 &
nohup python train.py config/experiment/b2_lower_lr.py > train-b2.log 2>&1 &
nohup python train.py config/experiment/b3_larger_context.py > train-b3.log 2>&1 &
nohup python train.py config/experiment/b4_optimizer_tuning.py > train-b4.log 2>&1 &
```

---

## Evaluation Criteria

After training completes (5000 iterations each), compare:

1. **Final Validation Loss** (lower is better)
   - Best val loss achieved across all iterations

2. **Best Checkpoint Step**
   - Which iteration achieved the best val loss

3. **Train-Val Gap**
   - (Train loss - Val loss) at best checkpoint
   - Smaller gap = better generalization

4. **Loss Stability**
   - Variance in val loss over last 1000 iterations
   - More stable = more confident generalization

5. **Perplexity (PPL)**
   - exp(val_loss)
   - Lower is better

---

## Success Criteria

| Metric | Target | Notes |
|--------|--------|-------|
| Best Val Loss | < 3.2 | Current best from your runs was ~3.3 |
| Train-Val Gap | < 1.0 | Your runs showed ~1.3 gap |
| Best Iteration | < 4000 | Earlier = more efficient |

---

## Next Steps After Results

Based on results, we will:

1. **If B0 (baseline) performs well** → Fine-tune within that region
2. **If B1 (regularization) wins** → Try even stronger regularization
3. **If B2 (lower LR) wins** → Try 5e-4 or 6e-4
4. **If B3 (context) wins** → Try block_size=384 or 256 with better LR
5. **If B4 (optimizer) wins** → Explore beta2=0.97 or 0.98

---

## Key Differences from Your Previous Runs

| Aspect | Your Previous Runs | This Experiment |
|--------|-------------------|-----------------|
| learning_rate | 6e-4 | 1e-3 (baseline) or variations |
| warmup_iters | 750 | 100 (baseline) or 300 |
| beta2 | 0.95 | 0.99 (baseline) or 0.95 |
| max_iters | 8000 | 5000 (matches baseline) |
| lr_decay_iters | 8000 | 5000 (matches baseline) |

The key insight is that your previous configs deviated from the baseline in several ways. This experiment starts from baseline and explores one direction at a time.

---

## Expected Timeline

- **Training time**: ~2-4 hours per config (depending on GPU)
- **All configs complete**: ~4 hours if run in parallel
- **Analysis**: 30 minutes

**Total estimated time**: 4-5 hours for one full experimental round
