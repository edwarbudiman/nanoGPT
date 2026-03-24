"""
Generate a story completion that ends at 5 sentences total (prompt included).

Usage:
python generate_rocstories_5sent.py --prompt="Once upon a time"
"""

import os
import pickle
import re
from contextlib import nullcontext

import torch
import tiktoken

from model import GPT, GPTConfig

# -----------------------------------------------------------------------------
init_from = 'resume'  # 'resume' or 'gpt2*'
out_dir = 'out-rocstories'
prompt = ''
max_total_new_tokens = 220
temperature = 0.8
top_k = 200
min_first_sentence_tokens = 12
retry_on_short_first_sentence = True
enforce_exact_five = True  # truncate back to exactly 5 sentences if overshot
seed = 1337
device = 'cuda'  # 'cpu', 'cuda', 'cuda:0', ...
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
exec(open('configurator.py').read())  # allows CLI overrides
# -----------------------------------------------------------------------------


def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]


def extract_complete_sentences(text):
    return [m.group(0).strip() for m in re.finditer(r'[^.!?]*[.!?]', text) if m.group(0).strip()]


def count_complete_sentences(text):
    return len(extract_complete_sentences(text))


def truncate_to_n_sentences(text, n):
    return " ".join(extract_complete_sentences(text)[:n])


def complete_sentence_after_prompt(prompt_text, full_text):
    if not full_text.startswith(prompt_text):
        return ""
    tail = full_text[len(prompt_text):].lstrip()
    m = re.search(r'[.!?]', tail)
    if m is None:
        return ""
    return tail[:m.end()].strip()


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

# tokenizer
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)

if load_meta:
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

if not isinstance(prompt, str):
    raise TypeError("`prompt` must be a string. Example: --prompt=\"Once upon a time\"")
prompt = prompt.strip()

prompt_sentence_count = len(split_sentences(prompt))
if prompt_sentence_count >= 5:
    output_text = truncate_to_n_sentences(prompt, 5) if enforce_exact_five else prompt
    print("Prompt already has >=5 sentences; returning prompt-only result.")
    print("----- Story (5 sentences target) -----")
    print(output_text)
    raise SystemExit(0)

x = torch.tensor(encode(prompt), dtype=torch.long, device=device)[None, ...]
token_ids = x[0].tolist() if x.numel() > 0 else []


def decode_without_eot(ids):
    return decode([t for t in ids if t != 50256])


generated_text = decode_without_eot(token_ids) if token_ids else prompt
new_tokens = 0

with torch.no_grad():
    with ctx:
        while count_complete_sentences(generated_text) < 5 and new_tokens < max_total_new_tokens:
            y = model.generate(x, 1, temperature=temperature, top_k=top_k)
            x = y
            new_tokens += 1
            token_ids = y[0].tolist()
            generated_text = decode_without_eot(token_ids)

        # If prompt is very short (1 word), the model can produce a very short first completion.
        # Optionally retry once with lower randomness to encourage a longer first sentence.
        if retry_on_short_first_sentence:
            first_after_prompt = complete_sentence_after_prompt(prompt, generated_text)
            first_word_count = len(first_after_prompt.split()) if first_after_prompt else 0
            if first_word_count > 0 and first_word_count < min_first_sentence_tokens:
                x = torch.tensor(encode(prompt), dtype=torch.long, device=device)[None, ...]
                token_ids = x[0].tolist() if x.numel() > 0 else []
                generated_text = decode_without_eot(token_ids) if token_ids else prompt
                new_tokens = 0
                retry_temperature = max(0.65, temperature - 0.15)
                retry_top_k = min(top_k, 100)
                while count_complete_sentences(generated_text) < 5 and new_tokens < max_total_new_tokens:
                    y = model.generate(x, 1, temperature=retry_temperature, top_k=retry_top_k)
                    x = y
                    new_tokens += 1
                    token_ids = y[0].tolist()
                    generated_text = decode_without_eot(token_ids)

if enforce_exact_five and count_complete_sentences(generated_text) > 5:
    generated_text = truncate_to_n_sentences(generated_text, 5)

print("----- Generation Summary -----")
print(f"prompt_sentences : {prompt_sentence_count}")
print(f"final_sentences  : {count_complete_sentences(generated_text)}")
print(f"new_tokens       : {new_tokens}")
print("----- Story (5 sentences target) -----")
print(generated_text)
