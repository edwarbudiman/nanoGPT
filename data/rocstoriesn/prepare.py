"""
Prepare ROCStories dataset for nanoGPT training.

This script:
1. Downloads ROCStories dataset from HuggingFace
2. Uses official train/test splits (test = validation)
3. Adds GPT-2 <|endoftext|> token AFTER each story
4. This helps model learn when to STOP generating a story
5. Tokenizes using GPT-2 BPE tokenizer (tiktoken)
6. Saves as train.bin and val.bin files

IMPORTANT: EOT token helps model learn story endings for generation!
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
STORY_SEPARATOR = "\n\n"  # Separator between stories (EOT token will be added after this)

def prepare_rocstories():
    """Download and prepare ROCStories dataset."""

    print("=" * 60)
    print("ROCStories Dataset Preparation for nanoGPT")
    print("=" * 60)

    # Load dataset from HuggingFace
    print(f"\n[1/4] Loading dataset '{DATASET_NAME}' from HuggingFace...")
    dataset = load_dataset(DATASET_NAME)

    print(f"Dataset splits: {list(dataset.keys())}")

    # Get official train and test splits
    train_split = dataset['train']
    test_split = dataset['test']

    print(f"Train stories: {len(train_split):,}")
    print(f"Test stories (used as validation): {len(test_split):,}")

    # Format stories
    print("\n[2/4] Formatting stories...")
    train_stories = [example['text'].strip() for example in train_split]
    val_stories = [example['text'].strip() for example in test_split]


    # Join with story separator
    train_text = STORY_SEPARATOR.join(train_stories)
    val_text = STORY_SEPARATOR.join(val_stories)

    print(f"Train text length: {len(train_text):,} characters")
    print(f"Val text length: {len(val_text):,} characters")
    # Tokenize using GPT-2 BPE tokenizer
    print("\n[3/4] Tokenizing with GPT-2 BPE tokenizer...")
    enc = tiktoken.get_encoding("gpt2")

    train_ids = enc.encode_ordinary(train_text)
    val_ids = enc.encode_ordinary(val_text)

    print(f"Train tokens: {len(train_ids):,}")
    print(f"Val tokens: {len(val_ids):,}")

    # Save to binary files
    print("\n[4/4] Saving to binary files...")
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
        'train_tokens': len(train_ids),
        'val_tokens': len(val_ids),
        'vocab_size': enc.n_vocab,
        'eot_token_id': STORY_SEPARATOR,
        'separator': '<|endoftext|> (after each story)',
        'tokenizer': 'gpt2 (tiktoken)',
        'note': 'EOT added AFTER each story for story generation'
    }

    meta_path = os.path.join(data_dir, 'meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved metadata to: {meta_path}")
    print("\n" + "=" * 60)
    print("ROCStories preparation complete!")
    print("=" * 60)
    print("\n[INFO] EOT token added AFTER each story")
    print(f"[INFO] Model will learn: story ends -> output EOT")
    print(f"[INFO] During generation: stop when model outputs EOT")

    return meta

if __name__ == "__main__":
    meta = prepare_rocstories()

    # Print summary
    print("\n--- Dataset Summary ---")
    print(f"Dataset: {meta['dataset']}")
    print(f"Train stories: {meta['num_train_stories']:,}")
    print(f"Validation stories: {meta['num_val_stories']:,}")
    print(f"Train tokens: {meta['train_tokens']:,}")
    print(f"Val tokens: {meta['val_tokens']:,}")
    print(f"Vocab size: {meta['vocab_size']}")
    print(f"EOT token ID: {meta['eot_token_id']}")
