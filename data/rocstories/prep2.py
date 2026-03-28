"""
Prepare ROCStories dataset for nanoGPT training.

This script:
1. Downloads ROCStories dataset from HuggingFace
2. Uses ONLY official train split (split into 90% train, 10% val)
3. Test split is held out for final evaluation
4. Adds GPT-2 <|endoftext|> token between stories for clear boundaries
5. Tokenizes using GPT-2 BPE tokenizer (tiktoken)
6. Saves as train.bin and val.bin files
"""

import os
import json
import numpy as np
from datasets import load_dataset

# Set tiktoken cache directory to writable location
os.environ['TIKTOKEN_CACHE_DIR'] = '/tmp/tiktoken_cache'

import tiktoken

# Configuration
DATASET_NAME = "mintujupally/ROCStories"
TRAIN_RATIO = 0.9  # 90% for training, 10% for validation

def prepare_rocstories():
    """Download and prepare ROCStories dataset."""

    print("=" * 60)
    print("ROCStories Dataset Preparation for nanoGPT")
    print("=" * 60)

    # Load dataset from HuggingFace
    print(f"\n[1/5] Loading dataset '{DATASET_NAME}' from HuggingFace...")
    dataset = load_dataset(DATASET_NAME)

    print(f"Dataset splits: {list(dataset.keys())}")

    # Get official train and test splits
    train_split = dataset['train']
    test_split = dataset['test']

    print(f"Official train stories: {len(train_split):,}")
    print(f"Official test stories (held out for evaluation): {len(test_split):,}")

    # Format ALL train stories
    print("\n[2/5] Formatting stories...")
    all_train_stories = [example['text'].strip() for example in train_split]

    # Split train into train/val (90/10)
    print("\n[3/5] Splitting train into train/val (90/10)...")
    n = len(all_train_stories)
    np.random.seed(42)  # For reproducibility
    indices = np.random.permutation(n)

    train_size = int(n * TRAIN_RATIO)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_stories = [all_train_stories[i] for i in train_indices]
    val_stories = [all_train_stories[i] for i in val_indices]

    print(f"Training stories: {len(train_stories):,}")
    print(f"Validation stories: {len(val_stories):,}")

    # Tokenize using GPT-2 BPE tokenizer
    print("\n[4/5] Tokenizing with GPT-2 BPE tokenizer...")
    enc = tiktoken.get_encoding("gpt2")

    # GPT-2 special tokens
    EOT_TOKEN = enc.eot_token  # <|endoftext|>, typically token ID 50256
    print(f"GPT-2 tokenizer loaded (vocab_size={enc.n_vocab})")
    print(f"End-of-text token ID: {EOT_TOKEN}")

    # Tokenize stories and add EOT token between each story
    def tokenize_with_eot(stories, tokenizer, eot_token):
        """Tokenize stories and add EOT token between each story."""
        all_ids = []
        for i, story in enumerate(stories):
            # Encode the story
            story_ids = tokenizer.encode(story, allowed_special="all")
            all_ids.extend(story_ids)

            # Add EOT token between stories (not after the last one)
            if i < len(stories) - 1:
                all_ids.append(eot_token)

        return all_ids

    print("Adding <|endoftext|> tokens between stories...")
    train_ids = tokenize_with_eot(train_stories, enc, EOT_TOKEN)
    val_ids = tokenize_with_eot(val_stories, enc, EOT_TOKEN)

    print(f"Train tokens (with EOT): {len(train_ids):,}")
    print(f"Val tokens (with EOT): {len(val_ids):,}")

    # Count EOT tokens
    eot_count_train = train_ids.count(EOT_TOKEN)
    eot_count_val = val_ids.count(EOT_TOKEN)
    print(f"EOT tokens in train: {eot_count_train:,}")
    print(f"EOT tokens in val: {eot_count_val:,}")

    # Save to binary files
    print("\n[5/5] Saving to binary files...")
    data_dir = os.path.dirname(os.path.abspath(__file__))

    train_ids_np = np.array(train_ids, dtype=np.uint16)
    val_ids_np = np.array(val_ids, dtype=np.uint16)

    train_path = os.path.join(data_dir, 'train.bin')
    val_path = os.path.join(data_dir, 'val.bin')

    train_ids_np.tofile(train_path)
    val_ids_np.tofile(val_path)

    print(f"Saved train.bin ({len(train_ids_np):,} tokens) to: {train_path}")
    print(f"Saved val.bin ({len(val_ids_np):,} tokens) to: {val_path}")

    # Save metadata
    meta = {
        'dataset': DATASET_NAME,
        'num_train_stories': len(train_stories),
        'num_val_stories': len(val_stories),
        'num_test_stories': len(test_split),
        'train_tokens': len(train_ids),
        'val_tokens': len(val_ids),
        'vocab_size': enc.n_vocab,
        'eot_token_id': EOT_TOKEN,
        'separator': '<|endoftext|>',
        'tokenizer': 'gpt2 (tiktoken)',
        'note': 'Test split held out for final evaluation'
    }

    meta_path = os.path.join(data_dir, 'meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved metadata to: {meta_path}")
    print("\n" + "=" * 60)
    print("ROCStories preparation complete!")
    print("=" * 60)
    print("\n[IMPORTANT] Test split is held out for final evaluation!")
    print(f"  - Training: {len(train_stories):,} stories ({len(train_ids):,} tokens)")
    print(f"  - Validation: {len(val_stories):,} stories ({len(val_ids):,} tokens)")
    print(f"  - Test (held out): {len(test_split):,} stories")
    print(f"\n[INFO] Added {eot_count_train:,} <|endoftext|> tokens to training data")
    print(f"[INFO] Added {eot_count_val:,} <|endoftext|> tokens to validation data")

    return meta

if __name__ == "__main__":
    meta = prepare_rocstories()

    # Print summary
    print("\n--- Dataset Summary ---")
    print(f"Dataset: {meta['dataset']}")
    print(f"Train stories: {meta['num_train_stories']:,}")
    print(f"Validation stories: {meta['num_val_stories']:,}")
    print(f"Test stories (held out): {meta['num_test_stories']:,}")
    print(f"Train tokens: {meta['train_tokens']:,}")
    print(f"Val tokens: {meta['val_tokens']:,}")
    print(f"Vocab size: {meta['vocab_size']}")
    print(f"EOT token ID: {meta['eot_token_id']}")
