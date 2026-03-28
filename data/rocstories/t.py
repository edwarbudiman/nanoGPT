"""Decode train.bin and val.bin back to plain text files."""
import os
import numpy as np
import tiktoken

BASE_DIR = os.path.dirname(__file__)
EOT_TOKEN_ID = 50256

enc = tiktoken.get_encoding("gpt2")

for split in ["train", "val"]:
    bin_path = os.path.join(BASE_DIR, f"{split}.bin")
    txt_path = os.path.join(BASE_DIR, f"{split}.txt")

    ids = np.fromfile(bin_path, dtype=np.uint16).tolist()

    # Split on EOT tokens to recover individual stories
    stories = []
    current = []
    for tok in ids:
        if tok == EOT_TOKEN_ID:
            if current:
                stories.append(enc.decode(current))
                current = []
        else:
            current.append(tok)
    if current:
        stories.append(enc.decode(current))

    with open(txt_path, "w", encoding="utf-8") as f:
        for story in stories:
            f.write(story.strip() + "\n\n")

    print(f"wrote {len(stories):,} stories to {txt_path}")
