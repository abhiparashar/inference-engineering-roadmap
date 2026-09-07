# 9 — Build: GPT-2's Forward Pass From Scratch (With a KV-Cache)

> **Goal:** implement GPT-2 inference in ~150 lines of NumPy — embeddings, LayerNorm, causal multi-head attention, MLP, logits, sampling, and a hand-rolled KV-cache — load the *real* pretrained weights into it, and prove it matches HuggingFace token for token.

This is the Phase 1 build ([project 03](../../projects/README.md)). Nothing internalizes lessons 1-8 like watching your own tensors produce coherent English. Afterwards, no part of a transformer is magic to you — which is the entire point of this phase.

**Rules:** you may use `numpy` for math, `transformers` **only** to download the pretrained weights, and `tiktoken` for the tokenizer. You may **not** use `torch.nn` layers, `model.generate()`, or any attention implementation you didn't write.

---

## What you're building

```
projects/03-gpt2-from-scratch/
├── gpt2.py         # the model: params → logits, in NumPy
├── generate.py     # the decode loop + KV-cache + sampling
├── verify.py       # compares your logits/tokens against HuggingFace
├── bench.py        # cache vs no-cache tokens/s, and the O(N²) curve
└── README.md       # your numbers, your shape table, what broke
```

Work in five milestones. **Do not skip the verification milestone** — an unverified reimplementation teaches you the wrong things confidently.

---

## Milestone 1 — Get the weights and see their shapes

```python
import numpy as np
from transformers import GPT2LMHeadModel

hf = GPT2LMHeadModel.from_pretrained("gpt2")            # 124M, downloads ~550 MB
params = {k: v.detach().numpy() for k, v in hf.state_dict().items()}
for k, v in list(params.items())[:12]:
    print(f"{k:40s} {v.shape}")
```

You'll see exactly the structure from [lesson 3](03-transformer-block.md):

| Key | Shape | What it is |
|---|---|---|
| `transformer.wte.weight` | (50257, 768) | token embedding table ([lesson 1](01-tokens-and-embeddings.md)) |
| `transformer.wpe.weight` | (1024, 768) | learned positional embeddings; 1024 = max context |
| `transformer.h.{i}.ln_1.{weight,bias}` | (768,) | pre-attention LayerNorm |
| `transformer.h.{i}.attn.c_attn.{weight,bias}` | (768, 2304), (2304,) | **fused** Q,K,V projection (3 × 768) |
| `transformer.h.{i}.attn.c_proj.{weight,bias}` | (768, 768) | the `W_O` output projection |
| `transformer.h.{i}.ln_2.{weight,bias}` | (768,) | pre-MLP LayerNorm |
| `transformer.h.{i}.mlp.c_fc.{weight,bias}` | (768, 3072) | MLP expand (4×) |
| `transformer.h.{i}.mlp.c_proj.{weight,bias}` | (3072, 768) | MLP project back |
| `transformer.ln_f.{weight,bias}` | (768,) | final LayerNorm |
| `lm_head.weight` | (50257, 768) | unembedding — **tied to `wte`** (same tensor) |

> **Two gotchas up front.** (1) GPT-2's linear layers are `Conv1D`, whose weight is stored as `(in_features, out_features)` — the *opposite* of `nn.Linear`. So you write `x @ W + b` with **no transpose**. (2) `lm_head.weight` is `wte.weight` (weight tying), and it *is* stored as `(vocab, d_model)`, so the final projection needs `x @ wte.T`.

Count the parameters yourself (`sum(v.size for v in params.values())`) and confirm ~124M, then check where they live — the 12 MLPs and the embedding table dominate, exactly as lesson 3 claimed.

## Milestone 2 — The primitive ops

```python
def gelu(x):                                     # GPT-2 uses the tanh approximation
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)        # numerical stability (lesson 2)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

def layer_norm(x, g, b, eps=1e-5):               # normalize across d_model, per token
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return g * (x - mu) / np.sqrt(var + eps) + b

def linear(x, w, b):                             # Conv1D convention: w is (in, out)
    return x @ w + b
```

Test each on a tiny input before going further: `softmax` rows must sum to 1, `layer_norm` output must have ≈0 mean and ≈1 variance per row.

## Milestone 3 — Attention, block, and the full forward pass

Write `attention` so it works for **both** phases ([lesson 6](06-prefill-vs-decode.md)): the query length may be `T` (prefill) or `1` (decode), while keys/values have length `T_past + T`.

```python
N_HEADS, D_HEAD = 12, 64      # 12 * 64 = 768

def mha(x, p, cache=None):
    """x: (T, 768). cache: dict with 'k','v' of shape (heads, T_past, 64) or None."""
    T = x.shape[0]
    qkv = linear(x, p["attn.c_attn.weight"], p["attn.c_attn.bias"])   # (T, 2304)
    q, k, v = np.split(qkv, 3, axis=-1)                               # 3 × (T, 768)

    # (T, 768) -> (heads, T, 64)
    to_heads = lambda t: t.reshape(T, N_HEADS, D_HEAD).transpose(1, 0, 2)
    q, k, v = to_heads(q), to_heads(k), to_heads(v)

    if cache is not None:                          # ← the KV-cache (lesson 5)
        k = np.concatenate([cache["k"], k], axis=1)
        v = np.concatenate([cache["v"], v], axis=1)
        cache["k"], cache["v"] = k, v
    T_full = k.shape[1]

    scores = q @ k.transpose(0, 2, 1) / np.sqrt(D_HEAD)               # (heads, T, T_full)

    # causal mask: query at absolute position (T_full - T + i) may see keys <= that
    rows = np.arange(T)[:, None] + (T_full - T)
    cols = np.arange(T_full)[None, :]
    scores = np.where(cols <= rows, scores, -np.inf)                  # (T, T_full) broadcast

    out = softmax(scores) @ v                                         # (heads, T, 64)
    out = out.transpose(1, 0, 2).reshape(T, N_HEADS * D_HEAD)         # concat heads
    return linear(out, p["attn.c_proj.weight"], p["attn.c_proj.bias"])

def block(x, p, cache=None):
    x = x + mha(layer_norm(x, p["ln_1.weight"], p["ln_1.bias"]), p, cache)          # residual
    h = layer_norm(x, p["ln_2.weight"], p["ln_2.bias"])
    h = linear(gelu(linear(h, p["mlp.c_fc.weight"], p["mlp.c_fc.bias"])),
               p["mlp.c_proj.weight"], p["mlp.c_proj.bias"])
    return x + h                                                                     # residual

def gpt2(token_ids, params, caches=None, past_len=0):
    """token_ids: (T,) ints. Returns logits (T, 50257)."""
    pos = np.arange(past_len, past_len + len(token_ids))        # ← absolute positions!
    x = params["wte"][token_ids] + params["wpe"][pos]           # (T, 768)
    for i, blk in enumerate(params["blocks"]):
        x = block(x, blk, None if caches is None else caches[i])
    x = layer_norm(x, params["ln_f.weight"], params["ln_f.bias"])
    return x @ params["wte"].T                                  # tied unembedding → logits
```

Reshape `params` into that nested form (a list of 12 per-block dicts plus the shared tensors) in a small loader function. Then print the shape of every intermediate for a 5-token prompt and check it against your lesson-2 shape table. If a shape surprises you, stop and find out why.

## Milestone 4 — Generate, with and without the cache

```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")

def generate(prompt, params, n=20, use_cache=True):
    ids = enc.encode(prompt)
    if not use_cache:                                   # naive loop from lesson 4
        for _ in range(n):
            logits = gpt2(np.array(ids), params)
            ids.append(int(logits[-1].argmax()))        # greedy (lesson 7)
        return enc.decode(ids)

    caches = [{"k": np.zeros((N_HEADS, 0, D_HEAD), dtype=np.float32),
               "v": np.zeros((N_HEADS, 0, D_HEAD), dtype=np.float32)} for _ in range(12)]
    logits = gpt2(np.array(ids), params, caches, past_len=0)        # PREFILL: all prompt tokens
    nxt = int(logits[-1].argmax())
    out = ids + [nxt]
    for _ in range(n - 1):                                          # DECODE: one token per pass
        logits = gpt2(np.array([nxt]), params, caches, past_len=len(out) - 1)
        nxt = int(logits[-1].argmax())
        out.append(nxt)
    return enc.decode(out)

print(generate("The capital of France is", params, n=10))
```

You should get grammatical, on-topic English (GPT-2 small is not smart, but it *is* fluent) — and, crucially, **exactly** what HuggingFace's greedy decode produces for the same prompt, which milestone 5 checks. If you get word salad, your weights are loaded wrong (usually a transpose or a head-reshape). Then add temperature/top-k/top-p from [lesson 7](07-sampling.md).

## Milestone 5 — Verify against HuggingFace (non-negotiable)

```python
import torch
prompt_ids = enc.encode("The capital of France is")
mine = gpt2(np.array(prompt_ids), params)                                   # (T, 50257)
theirs = hf(torch.tensor([prompt_ids])).logits[0].detach().numpy()

print("max abs diff:", np.abs(mine - theirs).max())        # expect < 1e-3 (float32 drift)
assert np.allclose(mine, theirs, atol=1e-3)
assert mine[-1].argmax() == theirs[-1].argmax()            # same next token — the real test
```

Then check that 20 greedy tokens match HF's `generate(do_sample=False)` **exactly**, and that cached vs uncached generation produce **identical** token IDs (they must — the cache is a pure optimization, not an approximation). That last assertion is the one that catches position-index bugs.

---

## The bugs you will hit (in order of likelihood)

| Symptom | Cause |
|---|---|
| Fluent-but-wrong text, diverges from HF after a few tokens | Wrong positional index when using the cache: you must pass **absolute** position `past_len`, not 0 |
| Complete word salad | Transposed a `Conv1D` weight (they're `(in, out)` already), or reshaped heads wrong — `(T, 12, 64).transpose` order matters |
| Logits off by ~1e-2, tokens mostly match | Wrong GELU (exact `erf` vs GPT-2's `tanh` approximation), or LayerNorm `eps` |
| First token right, everything after wrong | Mask built from relative instead of absolute positions during decode |
| `nan` everywhere | Softmax over an all-`-inf` row (bad mask), or missing max-subtraction |
| Cached and uncached outputs differ | Appending K/V *before* computing the current token's Q/K/V, or double-appending |
| `RuntimeWarning: divide by zero encountered in matmul` on macOS | **Not your bug.** NumPy ≥2 on Apple Accelerate raises spurious FP flags in BLAS matmul — reproduce it with a plain `np.random.randn(11,12) @ np.random.randn(12,36)`. Check `np.isfinite(logits).all()` and move on |

Debug by comparing **layer by layer** against HF: `hf(..., output_hidden_states=True).hidden_states[i]` gives you the residual stream after each block. Find the first layer where you diverge; the bug is in that block.

---

## Measure it (this is what makes it a portfolio piece)

Once it's correct, produce numbers — this is the [`playbooks/benchmarking.md`](../../playbooks/benchmarking.md) habit:

1. **Cache vs no-cache:** generate 100 tokens both ways. Record total time and tokens/s. Expect a large gap that *widens* with length.
2. **The O(N²) curve:** time per generated token vs sequence position, both ways. Plot it. The uncached line bends upward (each step reprocesses everything); the cached line is nearly flat with a gentle slope (attention over a growing cache). **This plot is [lesson 5](05-kv-cache.md) made visible** — put it in your README.
3. **Cache growth:** print `caches[0]["k"].shape` each step and multiply out the total KV bytes across all 12 layers. Compare against the [lesson 8](08-inference-math-and-memory.md) formula: `2 × 12 layers × 12 heads × 64 × 4 bytes` = 73,728 bytes/token in float32. Confirm your measured bytes match the formula.
4. **Where the time goes:** time the MLP vs attention vs `lm_head` per step. In NumPy on CPU you'll see the big matmuls dominate — the same shape of answer you'd get from a GPU profiler in Phase 2.

---

## Stretch goals (any one makes this a "large" project)

- **Batch it.** Add a batch dimension everywhere: `(B, T, 768)`, cache `(B, heads, T, 64)`. Then measure tokens/s at batch 1, 4, 16 and watch throughput rise — [lesson 8](08-inference-math-and-memory.md)'s batching argument on your own laptop. Handling **ragged** prompt lengths (left-padding + mask) teaches you exactly the problem continuous batching solves in Phase 3.
- **Pre-allocate the cache** to `max_len` and write into a slice instead of `np.concatenate` each step. Measure the difference — you just discovered why real engines pre-allocate KV blocks (and, when you see the wasted tail space, why **PagedAttention** exists).
- **Quantize the weights to INT8** (per-channel scale, dequantize on use). Measure memory saved and error vs FP32 output. Phase 4 in miniature.
- **Port it to GPT-2 medium/large** by parameterizing `n_layer/n_head/d_model` from the HF config — proves your code is architecture-driven, not hardcoded.
- **Do it in raw PyTorch tensors on GPU** (no `nn.Module`) and compare tokens/s against NumPy-on-CPU.

---

## Deliverable

Commit to `projects/03-gpt2-from-scratch/`, with a `README.md` containing:

- The **shape table** for one forward pass at your chosen prompt length (prefill *and* a decode step).
- The **verification evidence**: max logit diff vs HuggingFace, and confirmation that 20 greedy tokens match exactly.
- The **cache-vs-no-cache** timing table and the per-token-time plot.
- Measured KV-cache bytes vs the formula's prediction.
- A short "what broke and how I found it" section. Interviewers care about that paragraph more than the code.

Then update the Status column for project 03 in [`projects/README.md`](../../projects/README.md).

---

## Key takeaways

- GPT-2's forward pass is ~150 lines of NumPy: embed → 12 × (LN → attention → +, LN → MLP → +) → LN → unembed.
- The **KV-cache is ~5 lines** (concatenate, store, reuse) and must be a *pure optimization*: cached and uncached outputs must be token-identical.
- Correct **absolute positions** and a correct **causal mask** are what make prefill and decode agree — the top two bug sources.
- Verifying against HuggingFace layer by layer is the technique that makes reimplementation projects tractable.
- Measuring cache-vs-no-cache turns lesson 5's O(N²) claim into a plot you produced yourself.

**Next:** [Exercises & exit artifact →](10-exercises-and-artifacts.md) — close out Phase 1 and prove it.
