"""
Evaluate GPT on the full ROCStories test set (19,633 stories) and report average loss and perplexity.

Identical PPL calculation to eval.py, but loads the 19,633-story test set
from HuggingFace (mintujupally/ROCStories test.txt) — one story per line.

On first run, downloads test.txt and caches normalized stories locally at
data/rocstories/test_stories.txt. Subsequent runs load from cache.

Usage:
    python eval_rocstories.py --init_from=resume --out_dir=out-rocstories
    python eval_rocstories.py --init_from=resume --out_dir=out-rocstories --device=cpu --compile=False
"""

import math
import os
import pickle
import re
import sys
from contextlib import nullcontext

import torch
import tiktoken

from model import GPT, GPTConfig

# -----------------------------------------------------------------------------
# model/load config (same pattern as sample.py)
init_from = 'resume'  # 'resume' or a GPT-2 variant (e.g. 'gpt2-medium')
out_dir = 'out-rocstories'  # used when init_from == 'resume'
device = 'cuda'  # 'cpu', 'cuda', 'cuda:0', ...
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
seed = 1337

# data/eval config
max_paragraphs = -1  # -1 means all
print_first_n = 3  # preview first N loaded paragraphs

exec(open('configurator.py').read())  # allows overrides from CLI / config file
# -----------------------------------------------------------------------------

HF_REPO_ID = "mintujupally/ROCStories"
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "rocstories", "test_stories.txt")


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


def load_stories_from_cache(path):
    stories = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                stories.append(line)
    return stories


# Load test stories: use local cache if available, otherwise download from HF
if os.path.exists(CACHE_PATH):
    print(f"Loading cached test stories from {CACHE_PATH}")
    paragraphs = load_stories_from_cache(CACHE_PATH)
else:
    from huggingface_hub import snapshot_download
    print(f"Downloading dataset from Hugging Face Hub: {HF_REPO_ID}")
    local_repo_dir = snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        allow_patterns=["test.txt"],
    )
    test_txt = os.path.join(local_repo_dir, "test.txt")
    if not os.path.exists(test_txt):
        raise FileNotFoundError("Expected test.txt in Hugging Face dataset repo.")

    paragraphs = load_stories_from_hf_text(test_txt)

    # Cache locally for future runs
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        for story in paragraphs:
            f.write(story + "\n")
    print(f"Cached {len(paragraphs)} stories to {CACHE_PATH}")

if max_paragraphs is not None and max_paragraphs >= 0:
    paragraphs = paragraphs[:max_paragraphs]

if len(paragraphs) == 0:
    raise ValueError("No stories found")

print(f"Loaded {len(paragraphs)} stories")
for i, p in enumerate(paragraphs[:max(0, int(print_first_n))]):
    preview = p.replace('\n', ' ')[:120]
    print(f"[preview {i}] {preview}{'...' if len(p) > 120 else ''}")

# -----------------------------------------------------------------------------

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
else:
    raise ValueError(f"Unsupported init_from: {init_from}")

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

# tokenizer (same behavior as sample.py)
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi = meta['stoi']
    encode = lambda s: [stoi[c] for c in s]
else:
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})

total_nll = 0.0
total_tokens = 0
used_paragraphs = 0
skipped_short = 0
block_size = model.config.block_size
eval_batch_size = 64  # stories per forward pass (~50-80 tokens each, very light on VRAM)

# Pre-tokenize all stories
print("Tokenizing all stories...")
all_token_ids = []
for para in paragraphs:
    token_ids = encode(para)
    if len(token_ids) < 2:
        skipped_short += 1
        continue
    # Truncate to block_size+1 if needed (stories are short, unlikely)
    all_token_ids.append(token_ids[:block_size + 1])
    used_paragraphs += 1

n_stories = len(all_token_ids)
print(f"Evaluating {n_stories} stories in batches of {eval_batch_size}...")

with torch.no_grad():
    with ctx:
        for batch_start in range(0, n_stories, eval_batch_size):
            batch = all_token_ids[batch_start:batch_start + eval_batch_size]

            # Pad all stories in batch to same length (pad targets with -1 to ignore)
            max_len = max(len(t) for t in batch)
            xs = []
            ys = []
            lengths = []

            for token_ids in batch:
                inp = token_ids[:-1]
                tgt = token_ids[1:]
                n_tok = len(tgt)
                pad_len = (max_len - 1) - n_tok
                if pad_len > 0:
                    inp = inp + [0] * pad_len
                    tgt = tgt + [-1] * pad_len
                xs.append(inp)
                ys.append(tgt)
                lengths.append(n_tok)

            x = torch.tensor(xs, dtype=torch.long, device=device)
            y = torch.tensor(ys, dtype=torch.long, device=device)

            logits, _ = model(x, y)

            # Sum loss only over real tokens (ignore_index=-1 skips padding)
            loss_sum = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1),
                ignore_index=-1, reduction='sum'
            )

            total_nll += loss_sum.item()
            total_tokens += sum(lengths)

            done = min(batch_start + eval_batch_size, n_stories)
            if done % (eval_batch_size * 20) == 0 or done == n_stories:
                running_ppl = math.exp(total_nll / total_tokens)
                print(f"  [{done}/{n_stories}] running ppl: {running_ppl:.2f}")

if total_tokens == 0:
    raise ValueError("No valid tokens to evaluate. Check your input text.")

avg_loss = total_nll / total_tokens
ppl = math.exp(avg_loss)

print("----- Evaluation Results -----")
print(f"model           : {init_from}")
print(f"paragraphs_used : {used_paragraphs}")
print(f"paragraphs_skip : {skipped_short}")
print(f"pred_tokens     : {total_tokens}")
print(f"avg_loss        : {avg_loss:.3f}")
print(f"ppl             : {ppl:.2f}")
