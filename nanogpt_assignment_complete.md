# NanoGPT Assignment: Complete Task and Rules Document

## COMP4680/8650 Advanced Topics in Machine Learning

**Mini-Project 1: NanoGPT**

**Release Date:** March 5, 2026

**Codebase:** https://github.com/karpathy/nanoGPT

**Dataset:** https://huggingface.co/datasets/mintujupally/ROCStories

---

## Overview

This assignment focuses on training a small language model for story generation using nanoGPT. The assignment consists of four components: a warm-up exercise (not graded), training on ROCStories dataset, exploration of improvements, and an optional story generation competition. Students will develop practical experience with language model training, evaluation, and iterative improvement while adhering to computational constraints designed for fair assessment across different resource levels.

The primary objective is to build a functional story generation system from scratch, understand the training pipeline, and explore methods to improve model performance. Students are encouraged to think carefully about architecture choices, training strategies, and data processing techniques that can improve performance in principled and efficient ways.

---

## Warm-up: Setup and First Run (0 Points)

This component is not graded and serves as a foundational step to ensure you can properly set up your development environment and understand the nanoGPT codebase structure. The warm-up should be completed before attempting the graded tasks.

### Objectives

The warm-up phase has three main goals. First, you need to install all required dependencies following the official nanoGPT README and verify that your environment is configured correctly. Second, you should run a provided or official training or sampling command from nanoGPT to confirm your setup works end-to-end. Third, you will use the provided eval.py script to evaluate a checkpoint and gain a basic understanding of model quality assessment.

### Required Steps

Begin by cloning the nanoGPT repository and following the installation instructions provided in the README file. Ensure that all Python packages, CUDA drivers (if available), and other dependencies are properly installed and compatible with each other. After installation, execute a simple training command using the default configuration to verify that the training loop functions correctly. Document any issues encountered during setup and their solutions, as this process will help you understand the codebase structure better.

Next, run the evaluation script on a sample checkpoint to observe how model performance is measured. Pay attention to the metrics reported and the format of the output. Finally, experiment with the sampling functionality to generate text and understand how decoding parameters affect the generated output.

### Documentation Recommendations

Write brief notes covering the following areas: the overall code structure and main components, the location where datasets are prepared and loaded, the training loop implementation and key functions, and the sampling mechanism including temperature and top-k parameters. These notes will serve as a reference for the graded tasks and help you identify where modifications may be needed.

---

## Task 1: Train NanoGPT on ROCStories (7 Points)

Train your own nanoGPT model on the ROCStories training set. You may use the ROCStories test set for local evaluation and model selection. A private held-out dataset will be used for final grading, so avoid overfitting to the public test set.

### Task 1 Clarification

The goal of Task 1 is simply to train your own nanoGPT model on the ROCStories training set. Because this is meant to be a basic task, you do not have to make improvements on data or architecture. Your goal is just to get the base model training correctly and achieve reasonable performance. However, if you would like to make improvements in Task 1, they will also be welcomed but will not earn additional marks.

### Requirements

**Data Processing (2 Marks):** Convert ROCStories into a training corpus suitable for nanoGPT. You must document the preprocessing steps you used, including tokenization method, special tokens added, concatenation strategy for stories, and story separators if any. The pipeline correctness will be assessed based on whether the dataset is processed correctly and the training setup is sound.

**Training (1 Mark):** Train a model from scratch. You must report your key hyperparameters including model size (number of layers, attention heads, embedding dimensions), context length, batch size, learning rate schedule, number of training steps or epochs, random seed used, and any other relevant parameters. Experimental rigor will be evaluated through clear hyperparameter reporting and sensible baseline or ablation choices.

**Evaluation (2 Marks):** Evaluate your model on the ROCStories test split using at least one quantitative metric such as loss or perplexity. Include qualitative samples of generated stories to demonstrate model capability. The test performance component assesses both quantitative metrics and qualitative story samples.

### Baseline Configuration

The baseline should follow the official nanoGPT configuration as specified in `config/train_shakespeare_char.py`. This corresponds to the default "baby GPT model" with the following parameters:

| Parameter | Value |
|-----------|-------|
| n_layer | 6 |
| n_head | 6 |
| n_embd | 384 |
| Context Length | 256 |
| Approximate Parameters | 30.2M |
| Physical Size | ~340MB |

### Important Constraints

It is not allowed to fine-tune existing pre-trained models. All model parameters must be trained by yourself from scratch. You may do pre-training on your own data if you would like to do so, but you cannot start from external pre-trained checkpoints. For Task 1, the model size constraint of 32M total parameters applies. This ensures fair comparison across students with different computational resources.

### Performance Expectations

Task 1 marking criteria requires achieving a perplexity (PPL) value of approximately 25 or close to it (such as 25.1 or 25.2). Marks will be deducted if results do not reach this threshold. You will need to tune hyperparameters to achieve acceptable performance, so plan your experiments accordingly.

---

## Task 2: Exploration for Story Generation (8 Points)

This task is open-ended and encourages exploration beyond the base setup. The goal is to foster creativity and investigation into different approaches for improving story generation capability.

### Task 2 Clarification

While the assignment encourages exploration beyond the base setup and mentions trying different datasets like Q&A, dialogue, or even SVG icons, please note that the core focus remains on story generation. The intention behind trying different task settings is to explore whether co-training on them might increase the model's story generation ability. Task 2 is not meant for exploring tasks completely unrelated to story generation.

### Possible Exploration Directions

**Dataset Ideas:** You may explore using different datasets such as Q&A pairs, dialogue data, or structured text like recipes and instructions. The goal is to find complementary data that might improve the model's story generation capabilities through multi-task learning or transfer learning effects.

**Model Ideas:** You may experiment with alternative model architectures such as LLaMA-style blocks or other transformer variants. Consider how architectural changes might affect training dynamics, memory usage, and generation quality.

**Creative Experiments:** One example idea is to try a dataset of SVG icons or simple vector drawings and train the model to generate SVG code for basic shapes, logos, or glyphs. This explores whether the model can learn to generate structured output beyond natural language.

### Assessment Criteria

Task 2 will be assessed based on the following aspects:

**Novelty and Usefulness (3 Marks):** Full marks are awarded if your idea is completely novel and has not been proposed before. Full marks are also given if your method is demonstrated to be very useful across different tasks. Full marks are available if you can spot interesting insights from new datasets or new architectures. Only 1 mark will be awarded if you only switch a dataset or architecture without demonstrating clear benefits.

**Comprehensive Trials (3 Marks):** Full marks are given if your experiments are comprehensive and sensible. Important details of your experiments should be discussed, and evidence and results should be summarized. Missing important experiments will result in mark deductions.

**Writing Clarity (2 Marks):** Same as Task 1, assessed based on clear communication and proper documentation.

### What to Include in Your Report

For each exploration attempt, clearly state what you tried, why you chose that approach, and provide a short summary of what worked or did not work. Include quantitative evidence (metrics) and qualitative evidence (sample outputs) to support your conclusions. Be honest about failures and negative results, as these are valuable learning experiences.

---

## Task 3: Best Checkpoint Submission (15 Points)

Submit your best checkpoint for story generation. We will use private test prompts to evaluate your model using a combination of perplexity (PPL) and Qwen scores for evaluation.

### Task 3 Clarification

For this task, you are required to submit your best checkpoint for story generation. You should not just submit the basic checkpoint from Task 1. We expect to see the best model you produced after applying the various improvements explored in Task 2.

### Dataset and Size Constraints

You are allowed to use external datasets beyond ROCStories to improve your model, provided that your final model size does not exceed 32M parameters. This constraint applies to the model you submit to HuggingFace. In other words, do not evaluate models larger than 32M in Task 2 or Task 3.

### Architecture and Training Flexibility

You are allowed to change your training method and may modify the architecture by including your own `model.py` file in your submission. This allows for significant experimentation with different architectures and training strategies.

### Evaluation Script Constraints

You cannot change the evaluation script (`eval.py`) or the upload script (`hf_upload.py` / `hf_load.py`). Because these scripts must remain untouched, you are not allowed to change the pre-processing of the input text. Your final model must be perfectly loadable with your `model.py` and testable on our private dataset using the default text-preprocessing pipeline.

### Allowed Training Methods

The following training methods are explicitly allowed for Task 2 and Task 3:

- **Knowledge Distillation:** Allowed. The final submitted model must still be within 32M parameters.
- **RLHF (Reinforcement Learning from Human Feedback):** Allowed.
- **DPO (Direct Preference Optimization):** Allowed.
- **Pre-training:** Allowed on your own data.
- **Fine-tuning:** Allowed on custom datasets, but NOT from existing pre-trained models.

### Marking Criteria

Your mark for Task 3 will be based on two factors: whether you achieved a basic PPL value (to be published, expected to be around 20) and your ranking in the class relative to other students.

---

## Optional Task 4: Story Generation Competition (2 Extra Points)

We will hold lab competitions to compare models by story quality. Note that you can continue improving your model until the day before your lab session.

### Competition Timeline

By the day before your Week 6 lab, you can choose to submit your trained checkpoint(s) and a short description of your best configuration including data and model settings. We will provide prompts, and your model will generate stories based on these prompts for head-to-head comparison.

### Evaluation Method

All submissions will be evaluated with human judgment, focusing on four criteria: coherence (logical flow and consistency), creativity (originality and interesting ideas), consistency (maintaining narrative throughout), and grammar (proper language use). Student voting may be used as part of the evaluation process.

### Reward

The winning student will receive 2 extra marks for this course, with the total mark capped at 100.

---

## Report Guidelines

Your report should summarize Task 1 and Task 2 in a clear and organized manner. The main report should be no longer than 2 pages, excluding references. References and Appendix can extend beyond this limit.

### Required Sections

**Data Pipeline and Cleaning:** Clearly describe your preprocessing approach, including tokenization, special tokens, concatenation strategy, and any cleaning decisions made. Explain why these choices were made and how they affect model performance.

**Training Setup and Compute Budget:** Document your hardware configuration and approximate training time. Include all relevant hyperparameters and justify your choices where appropriate.

**Learning Curves:** Include training and validation loss curves and explain what you learned from these curves. Identify patterns such as overfitting, underfitting, or unstable training.

**Test Set Evaluation:** Report numerical metrics (perplexity, loss) and provide a brief error analysis discussing common failure modes observed in generated stories.

**Generated Samples:** Include samples generated under different decoding settings such as temperature and top-k values. Discuss how these parameters affect output quality and diversity.

### Appendix Policy

You may choose to put only a few generated samples in the main report and leave additional samples in the Appendix. Do not expect evaluators to read the Appendix carefully, so do not put important findings there. The Appendix should contain only supplementary generation samples.

### Writing Standards

Your report should be well-structured, concise, and readable with enough detail for someone to follow your methodology and results. Use plain English and plain grammar. Over-length reports will be penalized by 1-2 marks depending on the extent.

### AI Usage Declaration

You should declare whenever GPT or other AI tools are used for writing or coding. Any sentence that has been polished or written by AI should be highlighted in blue. Failure to declare AI-assisted writing will result in 1-2 marks deduction per undeclared sentence.

---

## Marking Criteria Summary

### Task 1 Assessment (7 Points Total)

| Component | Marks | Description |
|-----------|-------|-------------|
| Pipeline Correctness | 2 | Dataset processed correctly, training setup sound, system runs end-to-end |
| Experimental Rigor | 1 | Clear hyperparameter reporting, sensible baselines/ablation choices |
| Test Performance | 2 | Quantitative metrics and qualitative story samples |
| Writing Clarity | 2 | Well-structured, concise, readable report |

### Task 2 Assessment (8 Points Total)

| Component | Marks | Description |
|-----------|-------|-------------|
| Novelty/Usefulness | 3 | Original ideas, demonstrated usefulness, interesting insights |
| Comprehensive Trials | 3 | Sensible experiments, documented details, summarized evidence |
| Writing Clarity | 2 | Same criteria as Task 1 |

### Task 3 Assessment (15 Points Total)

Based on achieving basic PPL threshold (~20) and class ranking.

### Task 4 Assessment (2 Extra Points)

Based on story quality judged by human evaluation focusing on coherence, creativity, consistency, and grammar.

---

## Evaluation Methodology

### Quantitative Evaluation

For quantitative evaluation, we primarily use perplexity (PPL) as a metric. Lower perplexity indicates better model performance on predicting the next token. Your model should achieve a perplexity close to the target threshold (approximately 20 for Task 3, approximately 25 for Task 1).

### Qualitative Evaluation (Qwen Scoring)

For qualitative story evaluation, we use the following scoring rubric administered through Qwen. Your generated stories will be scored on a scale of 1 to 5:

| Score | Description |
|-------|-------------|
| 1 | Incoherent, highly repetitive, or completely ignores the opening sentence |
| 2 | Loosely follows the prompt with major issues (logic gaps, abrupt cutoff, repetition) |
| 3 | Adequate story following prompt with conclusion, but bland or has minor flaws |
| 4 | Good story; coherent, natural language, satisfying ending |
| 5 | Excellent; creative, engaging, all sentences connect, natural conclusion |

**Evaluation Prompt for Qwen:**

```
You are a creative writing evaluator. Score the following short story on a scale of 1 to 5 using this rubric:

1 — Incoherent, highly repetitive, or completely ignores the opening sentence
2 — Loosely follows the prompt; major issues (logic gaps, abrupt cutoff, repetition)
3 — Adequate story; follows prompt and has a conclusion, but bland or has minor flaws
4 — Good story; coherent, natural language, satisfying ending
5 — Excellent; creative, engaging, all sentences connect, natural conclusion

Respond ONLY with a JSON object in this exact format (no extra text):
{"score": N, "reason": "brief reason"}

Where N is an integer from 1 to 5.
```

---

## Submission Instructions

### Upload Method

Use `hf_load.py` to upload your model folder to HuggingFace under the namespace `<username>/nanoGPT_hw`. Ensure your HuggingFace authentication token is properly configured.

### Required Files

Your submitted repository must contain the following files:

| File | Description |
|------|-------------|
| `ckpt.pt` | Your trained model checkpoint. Only include the final checkpoint, no intermediate results. |
| `model.py` | Required if you modified the architecture. Must be loadable with the evaluation script. |
| `sample_params.json` | Optional JSON file recording your sampling parameters for generation. Will be loaded with `sample_params = json.load(f)` for sampling but not for PPL evaluation. |

### Repository Requirements

- The repository should contain only the final checkpoint and necessary files for loading and running inference.
- Remove any intermediate checkpoints or large temporary files.
- Ensure your model is properly saved and can be loaded using the standard nanoGPT loading mechanism.

### Canvas Submission

Under "Mini-Project 1" on Canvas, enter your repository information in the Text field. Type your `<username>/nanoGPT_hw` and your HuggingFace authentication token in two different rows as specified.

---

## Key Rules and Constraints Summary

### Model Size Constraint

The total number of parameters must not exceed 32M for all tasks (Task 1, Task 2, and the submitted model for Task 3). The baseline baby GPT model has approximately 30.2M parameters and occupies about 340MB. Your submitted model should be within a similar scale.

### Pre-training Requirement

All model parameters must be trained by yourself. Fine-tuning existing pre-trained models is not allowed. However, you may do pre-training on your own data if desired.

### Evaluation Script Integrity

You cannot modify `eval.py`, `hf_upload.py`, or `hf_load.py`. Your model must work with the default text-preprocessing pipeline used by these scripts.

### Architecture Modification

You may modify the architecture by including your own `model.py` file, but the modified model must remain loadable and testable with the evaluation infrastructure.

### Training Method Flexibility

All training methods are allowed including knowledge distillation, RLHF, and DPO, provided the final model size remains under 32M.

### External Datasets

You may use external datasets beyond ROCStories for training, as long as your final model does not exceed the size constraint.

### Larger Models Option

For those interested in experimenting with larger models, you are welcome to do so for exploration purposes. In the LLM arena, you can submit larger models, but they must be completely trained by you and not derived from external pre-trained models.

---

## Timeline and Deadlines

| Task | Deadline | Points |
|------|----------|--------|
| Warm-up | Before Task 1 | 0 |
| Task 1 | As scheduled | 7 |
| Task 2 | As scheduled | 8 |
| Task 3 (Checkpoint) | As scheduled | 15 |
| Task 4 (Competition) | Day before Week 6 lab | 2 (extra) |

---

## Important Reminders

Read all instructions carefully before beginning your implementation. Plan your experiments to ensure you can complete all required tasks within the available time. Document your work thoroughly as you proceed, as this will make report writing much easier. Test your model locally using the provided evaluation scripts before submission to avoid technical issues during grading.

Remember that the goal of Task 1 is to establish a working baseline, Task 2 is for exploration and improvement, and Task 3 is to submit your best model after applying lessons learned from Task 2. Start early, iterate often, and do not hesitate to experiment with different approaches.
