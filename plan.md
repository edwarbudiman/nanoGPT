# NanoGPT Task 2 + 3 Plan

## Goal

Improve story generation perplexity from ~24.8 (Task 1 baseline) to ~20 (Task 3 target) while documenting novel explorations for Task 2.

---

## A. Training Pipeline Improvements (from Karpathy's Autoresearch/NanoChat)

Karpathy's autoresearch system ran ~100 autonomous experiments on nanochat and discovered ~20 additive improvements that reduced GPT-2 training time by 11%. These generalized from depth-12 to depth-24 models.

### A1. QK-Norm with Learnable Scalar

**Idea:** The original QK-Norm implementation makes attention too diffuse. Adding a learnable scalar per head sharpens attention distribution.

**How:** In `model.py`, after applying QK-Norm (RMSNorm on Q and K), multiply by a learnable scalar parameter initialized to 1.0. This lets each head learn its own attention sharpness.

**Ground statement:** *"The agent discovered that the QK-Norm implementation was missing a scalar multiplier, making attention too diffuse across heads."* — Karpathy on X, describing autoresearch findings after ~700 experiments over 2 days.

**Reference:** [karpathy/autoresearch](https://github.com/karpathy/autoresearch), [Karpathy's X post](https://x.com/karpathy/status/2031135152349524125).

### A2. Value Embedding Regularization

**Idea:** Apply additional regularization specifically to value projections to combat overfitting (our main problem — train-val gap > 1.0 by step 8000).

**How:** Add a small auxiliary loss term that penalizes the norm of value embeddings, or apply separate dropout to value projections.

**Ground statement:** *"The agent found that the Value Embeddings really like regularization and Karpathy wasn't applying any."* — DataCamp autoresearch guide, summarizing one of ~20 discovered improvements.

**Reference:** [karpathy/autoresearch](https://github.com/karpathy/autoresearch), [DataCamp autoresearch guide](https://www.datacamp.com/tutorial/guide-to-autoresearch).

### A3. Weight Decay Schedule

**Idea:** Use dynamic weight decay rather than a fixed value. Current config uses fixed `weight_decay=1.0`.

**How:** Tie weight decay to learning rate schedule (e.g., cosine decay weight decay alongside LR), or use a warmup-then-decay pattern for weight decay.

**Ground statement:** *"Stacking up all of these changes [including weight decay schedule tuning], the leaderboard's 'Time to GPT-2' metric reduced from 2.02 hours to 1.80 hours — an 11% speedup on code that one of the best ML researchers had already optimized."* — Karpathy on X / Analytics Vidhya analysis.

**Reference:** [karpathy/autoresearch](https://github.com/karpathy/autoresearch), [karpathy/nanochat](https://github.com/karpathy/nanochat), [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2026/03/nanochat-gpt-2-training/).

### A4. Better Initialization

**Idea:** Improved weight initialization for transformer blocks improves training stability and final performance.

**How:** Review nanochat's init scheme. Common improvements include scaled init for residual connections (scale by `1/sqrt(2*n_layer)`) and muP-style parametrization.

**Ground statement:** *"The agent tuned the network initialization [...] all of [the changes] were additive and transferred to larger (depth=24) models."* — Karpathy on X.

**Reference:** [karpathy/nanochat](https://github.com/karpathy/nanochat), [Karpathy's X post](https://x.com/karpathy/status/2031135152349524125).

### A5. AdamW Beta Corrections

**Idea:** Default AdamW betas (0.9, 0.95) may not be optimal for small models on story data.

**How:** Experiment with beta2 values (0.95 vs 0.98 vs 0.99). The autoresearch system found that correcting betas improved convergence.

**Ground statement:** *"The agent found that [...] AdamW betas were all messed up"* — describing how the automated system discovered suboptimal optimizer settings that a human researcher had overlooked.

**Reference:** [karpathy/autoresearch](https://github.com/karpathy/autoresearch), [Data Science Dojo](https://datasciencedojo.com/blog/karpathy-autoresearch-explained/).

---

## B. Architecture Exploration — Simplified Engram Module (Task 2 Novelty)

DeepSeek released Engram, a conditional memory module that modernizes classic n-gram embeddings for O(1) lookup inside transformers.

### B1. Core Idea

Add an n-gram lookup table to specific transformer layers. When the model sees a sequence of tokens, it hashes 2-gram and 3-gram contexts into an embedding table and fuses retrieved vectors into the residual stream via context-aware gating.

### B2. Implementation Plan

1. **N-gram hash table:** Create a fixed-size embedding table (e.g., 64K entries, small dim like 128). Hash bigrams and trigrams of token IDs into table indices.
2. **Multi-head hashing:** Use 4-8 independent hash functions to reduce collisions.
3. **Context-aware gating:** Compute a gate value from the current hidden state: `gate = sigmoid(W_gate @ hidden_state)`. Fuse: `output = hidden + gate * n_gram_embedding`.
4. **Placement:** Insert at layers 1 and 4 (adapted from DeepSeek's layers 2 and 15 for our 6-layer model).
5. **Parameter budget:** Keep Engram params under ~2M to stay within 32M total.

### B3. Why This Is Good for Task 2

- Novel architectural modification (not just hyperparameter tuning)
- Directly relevant to story generation — n-grams capture phrases, names, and common patterns
- DeepSeek's sparsity allocation law suggests 20-25% of sparse params should go to memory
- Can provide interesting ablation results (with/without Engram, different n-gram sizes)

**Ground statements:**

- *"Mechanistic analyses reveal that Engram relieves the backbone's early layers from static reconstruction, effectively deepening the network for complex reasoning."* — Engram paper, Section 5.
- *"While the memory module is expected to aid knowledge retrieval (e.g., MMLU +3.4; CMMLU +4.0), even larger gains were observed in general reasoning (e.g., BBH +5.0; ARC-Challenge +3.7) and code/math domains (HumanEval +3.0; MATH +2.4)."* — Engram paper, main results.
- *"By delegating local dependencies to lookups, it frees up attention capacity for global context, substantially boosting long-context retrieval (e.g., Multi-Query NIAH: 84.2 to 97.0)."* — Engram paper.
- *"A U-shaped scaling law [...] optimizes the trade-off between neural computation (MoE) and static memory (Engram). 20-25% of sparse parameters should go to memory."* — Engram paper, Sparsity Allocation Law.

**Reference:** [DeepSeek Engram paper (arXiv:2601.07372)](https://arxiv.org/abs/2601.07372), [deepseek-ai/Engram GitHub](https://github.com/deepseek-ai/Engram).

---

## C. Multi-Token Prediction with Curriculum (Task 2 Exploration)

### C1. The Problem

Multi-token prediction (MTP) improves large models but **hurts small models** when applied naively.

**Ground statement:** *"Multi-token prediction models are worse than the baseline for small model sizes, but outperform the baseline at scale."* — Gloeckle et al. (Meta), arXiv:2404.19737. However: *"Gains are especially pronounced on generative benchmarks like coding, where [13B] models solve 12% more problems on HumanEval and 17% more on MBPP than comparable next-token models."*

### C2. Curriculum Fix

Start training with standard next-token prediction (NTP), then gradually introduce MTP heads as training progresses. This lets the model build strong representations before adding the harder MTP objective.

**How:**

1. Add D=2 extra prediction heads (shared backbone, separate linear heads)
2. For first 50% of training: only NTP loss
3. Gradually blend in MTP loss: `loss = NTP_loss + alpha * MTP_loss` where alpha ramps from 0 to 0.3
4. MTP heads are discarded at inference (only used as training regularizer)

**Ground statement:** *"[We] propose a curriculum learning strategy for MTP training that gradually increases the complexity from next-token prediction (NTP) to MTP, and enables small language models to better leverage the MTP objective during pre-training, improving downstream NTP performance and generative output quality."* — Aynetdinov & Akbik (ACL 2025), arXiv:2505.22757. Tested on 1.3B and 3B models. The *forward curriculum* (NTP→MTP) retained self-speculative decoding benefits.

### C3. Alternative: Future Summary Prediction (FSP)

Instead of predicting exact next tokens, predict a compact representation of future context.

**Ground statement:** *"Future summary prediction with learned summaries yields substantial improvements over NTP and MTP baselines, with gains of up to 5% on math and coding benchmarks."* — arXiv:2510.14751.

**Reference:** [Meta MTP paper (arXiv:2404.19737)](https://arxiv.org/abs/2404.19737), [MTP Curriculum — ACL 2025 (arXiv:2505.22757)](https://aclanthology.org/2025.acl-long.1243/), [FSP paper (arXiv:2510.14751)](https://arxiv.org/html/2510.14751v1).

---

## D. Data Improvements

### D1. Data Quality Filtering

Filter ROCStories for higher-quality stories (remove duplicates, very short stories, stories with encoding issues).

**Ground statement:** *"High-Quality Training Data matters more for SLMs than sheer data quantity [...] Phi-3 [was] trained on 'textbook-quality' synthetic data, carefully filtered to remove noise and redundancy."* — Machine Learning Mastery, SLM guide 2026.

### D2. External Data — TinyStories (Allowed per Assignment Rules)

Supplement ROCStories with TinyStories, a synthetic dataset of short stories generated by GPT-3.5/GPT-4 using vocabulary a 3-4 year old would understand.

**Ground statement:** *"TinyStories can be used to train and evaluate small language models that are much smaller than state-of-the-art models (below 10 million total parameters) [...] [these models] can produce fluent and consistent stories with several paragraphs that are diverse and have almost perfect grammar, and demonstrate reasoning capabilities."* — Eldan & Li (Microsoft), arXiv:2305.07759.

Keep final model at 32M params. External data for training only.

**Reference:** [TinyStories paper (arXiv:2305.07759)](https://arxiv.org/abs/2305.07759).

### D3. Story-Specific Tokenization

Review if current tokenization handles story boundaries, sentence structure, and special tokens optimally. Consider adding `<|story|>` separator tokens if not already present.

---

## E. Knowledge Distillation (Task 2/3)

### E1. Self-Distillation

Train multiple 32M models with different configs, then distill the ensemble into a single model.

**How:**
1. Train 3-5 models with different random seeds and hyperparams
2. Use their averaged logits as soft targets
3. Train a student model with: `loss = alpha * hard_loss + (1-alpha) * KL(student, teacher_avg)`

### E2. Why This Works

Allowed by assignment rules. The final model stays at 32M params. Ensemble distillation captures diverse learned patterns without increasing model size.

**Ground statements:**

- *"When a smaller net was regularized by matching soft targets produced by a large net at a temperature of 20, it achieved 74 test errors [vs 146 without distillation]."* — Hinton et al., "Distilling the Knowledge in a Neural Network" (arXiv:1503.02531).
- *"[We] obtained the best results when setting the weight on hard targets to be much smaller than the weight on soft targets."* — Hinton et al., on balancing the distillation loss.

**Reference:** [Hinton et al. (arXiv:1503.02531)](https://arxiv.org/abs/1503.02531). Assignment explicitly allows knowledge distillation.

---

## Execution Order

| Priority | Task | Effort | Expected Impact |
|----------|------|--------|-----------------|
| 1 | A1-A5: Port nanochat training improvements | Medium | High — directly improves convergence and reduces overfitting |
| 2 | D1-D3: Data quality and augmentation | Low | Medium — cleaner data = better generalization |
| 3 | B1-B3: Implement simplified Engram | High | High — architectural novelty for Task 2 marks |
| 4 | C1-C3: MTP curriculum experiment | Medium | Medium — good exploration content for Task 2 report |
| 5 | E1-E2: Knowledge distillation | Medium | Medium — final polish for Task 3 checkpoint |

---

## Success Metrics

- **Task 2:** Document at least 3 explorations with ablations, quantitative results, and qualitative samples
- **Task 3:** Achieve PPL ~20 on ROCStories test set (down from 24.8 baseline)
- **Qwen score:** Target average score of 3.5+ on generated stories
