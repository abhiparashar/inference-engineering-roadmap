# 7 — Sampling: Turning Logits Into Tokens

> **You'll be able to say:** "A forward pass gives me a raw score for every token in the vocabulary. Sampling is the rule that turns those ~50,000 numbers into the one token I emit — greedy, temperature, top-k, top-p. It's cheap compared to the forward pass, but it decides output quality, it's where determinism goes to die, and it's the hook for structured output and speculative decoding."

We've been writing `argmax` and moving on. Time to open that box.

---

## Where we are in the loop

From [lesson 3](03-transformer-block.md), the last thing the model does is project the final token's vector into **logits** — one raw, unbounded score per vocabulary entry:

```
last token's hidden state (d_model=768)
        │  @ lm_head  (768 × 50257)
        ▼
logits  (50257,)      e.g. [ 2.4, -1.1, 8.7, 0.3, … ]   ← raw scores, not probabilities
        │  sampling  ← THIS LESSON
        ▼
one token id, e.g. 15496 (" Paris")
```

Logits are not probabilities: they can be negative, they don't sum to 1. **Softmax** ([lesson 2](02-attention-mechanism.md)) converts them into a probability distribution over the whole vocabulary. Sampling is the policy for picking one token from that distribution.

---

## Greedy: always take the top one

```python
next_token = int(logits.argmax())     # highest-scoring token, every time
```

Simple, fast, and *deterministic in principle*. Its failure mode is real though: greedy text gets repetitive and flat ("The best way to learn is to learn the best way to learn…"), because always choosing the single most likely token locks the model into loops. Greedy is the right default for **benchmarks, tests, and correctness comparisons** ([lesson 9](09-build-gpt2-from-scratch.md) compares against HuggingFace with greedy), and it's common for code and structured extraction.

---

## Temperature: sharpen or flatten the distribution

Divide the logits by `T` before the softmax:

```python
probs = softmax(logits / T)
```

- `T < 1` (e.g. 0.7) → differences get **amplified** → distribution sharper → more predictable, safer text.
- `T = 1` → the model's own distribution, untouched.
- `T > 1` (e.g. 1.5) → differences get **squashed** → flatter → more surprising, more incoherent.
- `T → 0` → the max dominates completely → mathematically equivalent to **greedy**. (This is why APIs often accept `temperature=0` to mean "deterministic"; implementations special-case it to avoid dividing by zero.)

```
logits [4.0, 3.0, 1.0]
  T=0.5 → probs [0.88, 0.12, 0.00]     sharp   ← nearly always picks token 0
  T=1.0 → probs [0.70, 0.26, 0.04]     as-is
  T=2.0 → probs [0.55, 0.33, 0.12]     flat    ← token 2 now has a real chance
```

Temperature alone is a blunt instrument: even at low temperature, the *tail* of 50,000 tokens holds a lot of total probability mass, so a garbage token can still be drawn. Hence truncation.

---

## Top-k: only consider the k best

Keep the `k` highest-probability tokens, zero out everything else, renormalize, then sample.

```python
def top_k(logits, k=50):
    kth = np.partition(logits, -k)[-k]     # k-th largest value
    logits = np.where(logits < kth, -np.inf, logits)
    return logits                           # softmax + sample from these
```

This kills the long tail of nonsense. Its weakness is that `k` is fixed while the model's confidence isn't: after "The capital of France is" the distribution is a spike (k=50 drags in 49 junk options), while mid-essay it's genuinely broad (k=50 is too restrictive).

---

## Top-p (nucleus): keep the smallest set that covers p of the mass

The adaptive fix, and the modern default. Sort by probability, accumulate until the cumulative mass reaches `p` (e.g. 0.9), keep only those tokens.

```python
def top_p(probs, p=0.9):
    order = np.argsort(probs)[::-1]         # descending
    cum = np.cumsum(probs[order])
    cutoff = np.searchsorted(cum, p) + 1    # smallest prefix covering p
    keep = order[:cutoff]
    out = np.zeros_like(probs)
    out[keep] = probs[keep]
    return out / out.sum()                  # renormalize
```

```
confident step:  probs [0.92, 0.04, 0.01, …]  → nucleus = 1 token   (behaves like greedy)
uncertain step:  probs [0.11, 0.10, 0.09, …]  → nucleus = 30 tokens (stays creative)
```

The nucleus **resizes itself** with the model's confidence. That's why `temperature ≈ 0.7–1.0` + `top_p ≈ 0.9` is the near-universal default in production APIs. `top_k` and `top_p` are usually applied together (k first, then p).

Related knobs you'll see in real engines: **min-p** (keep tokens whose probability is at least a fraction of the top token's — cheaper and more adaptive than top-p), **repetition / presence / frequency penalties** (subtract from the logits of tokens already produced, to break loops), and **logit bias** (a caller-supplied per-token nudge, e.g. banning a token by setting it to −∞).

---

## The order of operations (get this right or your outputs are subtly wrong)

Every engine applies these in a fixed pipeline. The canonical order:

```
raw logits
  → logit bias / banned tokens        (hard constraints, set to -inf)
  → repetition & frequency penalties  (adjust logits of seen tokens)
  → divide by temperature
  → top-k truncation
  → top-p (nucleus) truncation
  → softmax → renormalize
  → draw one sample                   (multinomial / categorical)
```

The classic bug is applying temperature *after* truncation, which changes which tokens survive the cut. When you read vLLM's or HF's sampler in Phase 3/5, trace this exact order in the code.

---

## Stopping

Sampling also decides when to *stop*, which is part of the same step:

- **EOS token** — the model emits end-of-text; the request finishes naturally.
- **`max_tokens`** — a hard budget (also your memory guard: it caps KV-cache growth per request).
- **Stop strings** — e.g. `"\nUser:"`. Subtle, because a stop string can straddle token boundaries ([lesson 1](01-tokens-and-embeddings.md)), so servers match on the *detokenized* text and then trim, which is also why streaming APIs sometimes hold a token back before emitting.

---

## What this costs (the engineering angle)

Per decode step you sample over a `vocab_size` vector — 32k–256k floats. Compared to a full forward pass (billions of FLOPs and gigabytes of weight reads, [lesson 6](06-prefill-vs-decode.md)) that's small, but it is **not** free:

- The `lm_head` projection itself (`d_model × vocab_size`, e.g. 4096 × 128256 ≈ 0.5B params) is one of the largest single matrices in the model — a real chunk of the bytes you read every step.
- Top-p requires a **sort** over the vocabulary per sequence per step. At batch 256 that's 256 sorts every ~20 ms; engines use fused/partial-sort GPU kernels for this, and it can show up on a profile.
- Sampling runs on the GPU. Pulling logits to the CPU to sample in Python adds a synchronization point that stalls the pipeline — a classic naive-implementation performance bug.

### Determinism (a production trap)

"Greedy is deterministic" is only half true in practice:

- **Same batch composition, same seed → same output.** Fine.
- **Different batch composition → possibly different output.** Floating-point addition isn't associative, and batched GPU kernels change reduction order with batch shape. Logits shift by ~1e-6, and any near-tie at the argmax flips. Your request's output can therefore depend on *who else was in the batch* — which is why reproducibility bugs in serving are so maddening, and why "temperature=0" does not guarantee bit-identical replies from a busy server.
- Seeds must be **per request**, not global, or concurrent requests will interfere.

### Where sampling meets other optimizations

**Speculative decoding** (Phase 4) depends on this lesson: a draft model proposes tokens and the big model verifies them in one pass. To keep the output distribution *identical* to plain sampling, verification uses a **rejection-sampling** rule against the target distribution — you can't just compare argmaxes. Understanding sampling as "draw from a distribution" (not "pick the best token") is what makes that algorithm make sense.

**Structured / constrained output** (JSON schemas, grammars, tool calls) is also implemented here: at each step, mask the logits of every token that would violate the grammar to −∞ before sampling. All "guaranteed valid JSON" features are logit masking in the sampler.

---

## Key takeaways

- The model outputs **logits** (one raw score per vocab token); **softmax** makes them probabilities; **sampling** picks one.
- **Greedy** = argmax: repetitive but reproducible; the right choice for correctness tests and benchmarks.
- **Temperature** sharpens (`<1`) or flattens (`>1`) the distribution; `T→0` is greedy.
- **Top-k** keeps a fixed number of candidates; **top-p (nucleus)** keeps the smallest set covering probability mass `p` and adapts to model confidence. `T≈0.7–1.0` + `top_p≈0.9` is the production default.
- Order matters: bias/penalties → temperature → top-k → top-p → softmax → draw.
- Sampling is cheap next to the forward pass but not free: the `lm_head` is a huge matrix, top-p needs a per-sequence sort, and CPU-side sampling adds a GPU sync stall.
- **Determinism is not guaranteed** across different batch compositions, even at temperature 0 — floating-point reduction order changes with batch shape.
- Sampling is the hook for **speculative decoding** (rejection sampling) and **structured output** (logit masking).

**Next:** [Inference math & memory →](08-inference-math-and-memory.md) — the payoff lesson, where every claim in this phase becomes a number you can compute.
