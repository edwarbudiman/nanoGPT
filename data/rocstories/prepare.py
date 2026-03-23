import os
import re

import numpy as np
import tiktoken
from huggingface_hub import snapshot_download

HF_REPO_ID = "mintujupally/ROCStories"
EOT_TOKEN_ID = 50256  # <|endoftext|> in GPT-2 tokenizer
BASE_DIR = os.path.dirname(__file__)


def split_into_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def normalize_story(raw_story):
    sentences = split_into_sentences(raw_story)
    if len(sentences) != 5:
        return None
    return " ".join(sentences)


def load_stories_from_hf_text(path):
    stories = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            story = normalize_story(line)
            if story is not None:
                stories.append(story)
    return stories


print(f"downloading dataset files from Hugging Face Hub: {HF_REPO_ID}")
local_repo_dir = snapshot_download(
    repo_id=HF_REPO_ID,
    repo_type="dataset",
    allow_patterns=["train.txt", "test.txt"],
)

train_txt = os.path.join(local_repo_dir, "train.txt")
test_txt = os.path.join(local_repo_dir, "test.txt")
if not os.path.exists(train_txt) or not os.path.exists(test_txt):
    raise FileNotFoundError(
        "Expected train.txt and test.txt in Hugging Face dataset repo."
    )

train_stories = load_stories_from_hf_text(train_txt)
test_stories = load_stories_from_hf_text(test_txt)

if not train_stories or not test_stories:
    raise ValueError(
        "Failed to load 5-sentence stories from train.txt/test.txt. "
        "Please check dataset format."
    )

print(f"loaded train stories: {len(train_stories):,}")
print(f"loaded val stories: {len(test_stories):,}")

enc = tiktoken.get_encoding("gpt2")


def tokenize_stories(stories_list):
    ids = []
    for story in stories_list:
        ids.extend(enc.encode_ordinary(story))
        ids.append(EOT_TOKEN_ID)
    return ids


train_ids = tokenize_stories(train_stories)
val_ids = tokenize_stories(test_stories)

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")
print(f"total has {len(train_ids) + len(val_ids):,} tokens")

train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(BASE_DIR, "train.bin"))
val_ids.tofile(os.path.join(BASE_DIR, "val.bin"))
print("wrote train.bin and val.bin")

# Ensure GPT-2 BPE path is used in training (no char-level meta.pkl)
meta_path = os.path.join(BASE_DIR, "meta.pkl")
if os.path.exists(meta_path):
    os.remove(meta_path)
    print("removed stale meta.pkl")
