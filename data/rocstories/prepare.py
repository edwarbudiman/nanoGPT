"""
Prepare ROCStories dataset for nanoGPT training.

This script:
1. Downloads ROCStories dataset from HuggingFace
2. Uses official train/test splits
3. Tokenizes using GPT-2 BPE tokenizer (tiktoken)
4. Saves as train.bin and val.bin files (test split used as validation)
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
STORY_SEPARATOR = "<|endoftext|>"  # GPT-2 special token as story boundary

def prepare_rocstories():
    """Download and prepare ROCStories dataset using official splits."""

    print("=" * 50)
    print("ROCStories Dataset Preparation for nanoGPT")
    print("=" * 50)

    # Load dataset from HuggingFace
    print(f"\n[1/4] Loading dataset '{DATASET_NAME}' from HuggingFace...")
    dataset = load_dataset(DATASET_NAME)

    print(f"Dataset splits: {list(dataset.keys())}")

    # Get official train and test splits
    # train_split = dataset['train']
    validation_split = dataset['test']
    
    # split dataset['train'] 90:10 for train and validation
    full_train_split = dataset['train']
    train_size = int(0.9 * len(full_train_split))
    train_split = full_train_split.select(range(train_size))
    test_split = full_train_split.select(range(train_size, len(full_train_split)))
    

    print(f"Train stories: {len(train_split):,}")
    print(f"Test stories (used as validation): {len(test_split):,}")

    # Save raw text files
    data_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(data_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for example in train_split:
            f.write(example['text'] + '\n')
    with open(os.path.join(data_dir, 'val.txt'), 'w', encoding='utf-8') as f:
        for example in validation_split:
            f.write(example['text'] + '\n')
    print("Saved train.txt and val.txt")

    # Format stories
    print("\n[2/4] Formatting stories...")
    train_stories = [example['text'].strip() for example in train_split]
    val_stories = [example['text'].strip() for example in test_split]

    # Join with <|endoftext|> as story boundary
    train_text = STORY_SEPARATOR.join(train_stories)
    val_text = STORY_SEPARATOR.join(val_stories)

    print(f"Train text length: {len(train_text):,} characters")
    print(f"Val text length: {len(val_text):,} characters")

    # Tokenize using GPT-2 BPE tokenizer
    print("\n[3/4] Tokenizing with GPT-2 BPE tokenizer...")
    enc = tiktoken.get_encoding("gpt2")

    train_ids = enc.encode(train_text, allowed_special={"<|endoftext|>"})
    val_ids = enc.encode(val_text, allowed_special={"<|endoftext|>"})

    print(f"Train tokens: {len(train_ids):,}")
    print(f"Val tokens: {len(val_ids):,}")

    # # Save output in text files too
    # check_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'check')
    # os.makedirs(check_dir, exist_ok=True)
    # with open(os.path.join(check_dir, 'train.txt'), 'w', encoding='utf-8') as f:
    #     f.write(enc.decode(train_ids))
    # with open(os.path.join(check_dir, 'val.txt'), 'w', encoding='utf-8') as f:
    #     f.write(enc.decode(val_ids))

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
        'separator': STORY_SEPARATOR,
        'tokenizer': 'gpt2 (tiktoken)'
    }

    meta_path = os.path.join(data_dir, 'meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved metadata to: {meta_path}")
    print("\n" + "=" * 50)
    print("ROCStories preparation complete!")
    print("=" * 50)

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
