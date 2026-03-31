# Plan A Experiments - Configuration Summary

## Overview
5 configs to systematically explore regularization-optimizer space.
All configs use QK-Norm (A1) as the primary stability mechanism.

## Quick Reference Table

| Config | File | Key Change | dropout | v_dropout | beta2 | lr | max_iters |
|--------|------|------------|---------|-----------|-------|-----|-----------|
| Baseline | a_combinedheavy | N/A | 0.3 | 0.1 | 0.95 | 6e-4 | 10000 |
| Exp 1 | a_balanced_light | Moderate reduction | 0.15 | 0.05 | 0.98 | 5e-4 | 12000 |
| Exp 2 | a_very_light | Minimal regularization | 0.1 | 0.0 | 0.99 | 5e-4 | 15000 |
| Exp 3 | a_extended | Long training | 0.2 | 0.05 | 0.98 | 6e-4 | 20000 |
| Exp 4 | a_beta_focus | Beta2=0.99 focus | 0.25 | 0.05 | 0.99 | 6e-4 | 12000 |
| Exp 5 | a_lr_focus | Lower LR focus | 0.15 | 0.05 | 0.98 | 4e-4 | 15000 |

## Detailed Comparison

### Key Parameters Across All Configs (Fixed)

| Parameter | Value |
|-----------|-------|
| n_layer | 6 |
| n_head | 6 |
| n_embd | 384 |
| block_size | 256 |
| batch_size | 32 |
| qk_norm | True |
| weight_decay start | 0.5-0.6 |
| warmup_iters | 500-750 |

### Variable Parameters

| Config | dropout | v_dropout | beta2 | learning_rate | max_iters | min_wd |
|--------|---------|-----------|-------|---------------|-----------|--------|
| Baseline | 0.30 | 0.10 | 0.95 | 6e-4 | 10000 | 0.05 |
| Exp 1 | 0.15 | 0.05 | 0.98 | 5e-4 | 12000 | 0.10 |
| Exp 2 | 0.10 | 0.00 | 0.99 | 5e-4 | 15000 | 0.05 |
| Exp 3 | 0.20 | 0.05 | 0.98 | 6e-4 | 20000 | 0.10 |
| Exp 4 | 0.25 | 0.05 | 0.99 | 6e-4 | 12000 | 0.15 |
| Exp 5 | 0.15 | 0.05 | 0.98 | 4e-4 | 15000 | 0.10 |

## Hypotheses

### Exp 1 (Balanced Light)
- **Hypothesis**: Reducing regularization by ~50% will allow better fitting while QK-Norm provides stability
- **Expected**: PPL ~24-26, stable training
- **Risk**: Moderate overfitting possible

### Exp 2 (Very Light)
- **Hypothesis**: Minimal regularization maximizes fitting capability
- **Expected**: PPL ~22-25, but may overfit
- **Risk**: High - watch train-val gap closely

### Exp 3 (Extended Training)
- **Hypothesis**: Current best config just needs more time to converge
- **Expected**: PPL ~23-25, steady improvement
- **Risk**: Low - safe bet

### Exp 4 (Beta Focus)
- **Hypothesis**: Higher beta2 (smoother updates) prevents overshooting
- **Expected**: PPL ~24-26, very stable
- **Risk**: Very Low

### Exp 5 (LR Focus)
- **Hypothesis**: Lower learning rate allows finer convergence
- **Expected**: PPL ~23-25, slower but steadier
- **Risk**: Medium - needs longer training

## Running Instructions

Run each config in parallel:
```bash
# Terminal 1
python train.py config/plan-a/a_balanced_light.py

# Terminal 2
python train.py config/plan-a/a_very_light.py

# Terminal 3
python train.py config/plan-a/a_extended.py

# Terminal 4
python train.py config/plan-a/a_beta_focus.py

# Terminal 5
python train.py config/plan-a/a_lr_focus.py
```

## Expected Results (Ranked by Likelihood of Success)

1. **Exp 1 (Balanced Light)** - Most likely to succeed
   - Systematic reduction of all regularization factors
   - Best balance of fitting vs generalization

2. **Exp 3 (Extended Training)** - Safe bet
   - Current best with longer training
   - May reach similar or slightly better results

3. **Exp 5 (LR Focus)** - Good alternative
   - Lower LR often helps convergence
   - May find different local minimum

4. **Exp 4 (Beta Focus)** - Conservative
   - Highest stability
   - May not improve much over baseline

5. **Exp 2 (Very Light)** - High risk, high reward
   - Could be best OR worst
   - Monitor train-val gap closely

## Success Criteria

- **Target PPL**: ~20 (Task 3 requirement)
- **Acceptable PPL**: ~22-24 (better than Task 1 baseline)
- **Train-val gap**: < 1.0 indicates good generalization
- **Train-val gap**: > 1.5 indicates overfitting

## After First Run

Once you have results, identify the best config and create a "combined best" config that:
1. Takes the best dropout/v_dropout from winning config
2. Uses extended training (max_iters: 20000)
3. Applies early stopping based on best checkpoint
