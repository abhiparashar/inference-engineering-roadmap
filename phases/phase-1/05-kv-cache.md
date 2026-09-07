# 5 — The KV-Cache (The Central Trade-off of LLM Inference)

> **You'll be able to say:** "Without caching, generating each token re-does attention over the whole sequence — O(N²) total work. Caching the Keys and Values from past tokens makes each step O(N). The cost is memory that grows with sequence length × batch size. That memory-vs-compute trade is *the* central problem of LLM inference engineering."

This lesson is the reason the rest of the roadmap exists. Read it twice.

---

## The waste we're fixing

Recall the naive decode loop ([lesson 4](04-autoregressive-decoding.md)): every step re-runs the model over the *entire* sequence so far. Now recall attention ([lesson 2](02-attention-mechanism.md)): to attend, each token needs the **Key** and **Value** vectors of every earlier token.

Here's the key observation: **the Keys and Values of past tokens never change.** Token 3's K and V are computed from token 3's vector, which is fixed the moment token 3 exists. When we generate token 8, we recompute K and V for tokens 1-7 *even though we already computed them* at earlier steps. That's the waste.

```
step generating tok4:  compute K,V for tok1,2,3,4   → attend
step generating tok5:  compute K,V for tok1,2,3,4,5 → attend   ← recomputed 1-4 again!
step generating tok6:  compute K,V for tok1..6                 ← recomputed 1-5 again!
```

Redoing the past every step means step *i* costs O(i) work, and doing that for N steps costs **O(N²)** total. For a 2000-token generation, that's millions of redundant recomputations.

---

## The fix: cache K and V

Keep a running store — the **KV-cache** — of the Key and Value vectors for every token processed so far. Then at each new step you only:

1. Compute Q, K, V for the **one new token**.
2. **Append** its K and V to the cache.
3. Attend: the new token's Q against **all cached K**, then weight **all cached V**.

No recomputation of the past. Each step is now **O(N)** (one query against N cached keys), and total generation is O(N²) → but with a *tiny* constant, because you're not re-running the model over the whole sequence — you run it on **one token** per step. In practice this is a massive speedup (often 10-100x for long generations).

```
                 ┌─────────── KV-cache (per layer, per head) ───────────┐
step tok4:  new  │ K1 K2 K3 K4                                          │
                 │ V1 V2 V3 V4                                          │
step tok5:  new  │ K1 K2 K3 K4 K5   ← just appended K5,V5; 1-4 reused   │
                 │ V1 V2 V3 V4 V5                                       │
step tok6:  new  │ K1 K2 K3 K4 K5 K6                                    │
                 └──────────────────────────────────────────────────────┘
                    only the NEW token is fed through the model each step
```

### In code (conceptually)

```python
# cache holds past keys/values, one entry per layer
def decode_step(model, new_token, cache):
    x = embed(new_token)                     # only the NEW token, shape (1, d_model)
    for layer in model.layers:
        q, k, v = layer.qkv(x)               # Q,K,V for just this token
        cache[layer].k.append(k)             # append to cache  ← the whole trick
        cache[layer].v.append(v)
        # attend: this token's q against ALL cached k, then weight ALL cached v
        x = attention(q, cache[layer].k, cache[layer].v)
        x = layer.mlp(x)
    return logits(x)                          # next-token prediction
```

Contrast with lesson 4's naive `model(tokens)` over the whole growing list. Same output; the cache just stops you redoing settled work. Note **Q is not cached** — you only ever need the current token's query; past queries are done with. Only **K and V** are cached (hence the name).

---

## The catch: memory (and why it's the whole ballgame)

Nothing is free. The KV-cache trades *compute* for *memory*, and that memory is large and grows with every token. Its size is:

```
KV-cache bytes = batch × seq_len × num_layers × num_kv_heads × head_dim × 2 × bytes_per_number
                                                                         │
                                                       the "2" = one for K, one for V
```

Walk the factors: **more concurrent users** (batch), **longer conversations** (seq_len), **deeper models** (layers) all multiply the cache. Let's make it concrete for a 7B-class model (≈32 layers, 32 heads, head_dim 128, FP16 = 2 bytes):

```
per token, per sequence = 32 layers × 32 heads × 128 × 2 (K,V) × 2 bytes
                        ≈ 524,288 bytes ≈ 0.5 MB per token

a 2,000-token conversation:  2000 × 0.5 MB ≈ 1 GB   ← for ONE user
serving 40 such users:       40 × 1 GB     ≈ 40 GB  ← more than the weights!
```

That's the punchline: **at scale, the KV-cache can dwarf the model weights.** On an 80 GB GPU holding a 14 GB model, the *remaining ~66 GB is almost entirely KV-cache* — so **KV-cache memory, not model size, is what caps how many users you can serve concurrently.** This single constraint is why the following exist:

| Technique | What it does about KV-cache memory | Where |
|---|---|---|
| **PagedAttention** (vLLM) | Stops fragmentation/over-allocation by storing the cache in OS-style paged blocks | Phase 4 |
| **Prefix sharing / caching** | Two requests with the same system prompt share the same cached blocks | Phase 4/6 |
| **KV-cache quantization** | Store K,V in INT8/FP8 → half or quarter the bytes | Phase 4 |
| **GQA / MQA** (fewer KV heads) | Share K,V across query heads → `num_kv_heads` ≪ `num_heads` shrinks the cache | model design |
| **Disaggregated prefill/decode** | Move the KV-cache between GPU pools optimized for each phase | Phase 6 |

Every one of these is "manage the KV-cache better." That's why this lesson is the hinge of the roadmap: **understand the KV-cache memory trade and you understand *why* every serving system is built the way it is.**

> Note the sneaky one: **GQA/MQA** — modern models deliberately use *fewer* KV heads than query heads (e.g. 32 query heads but 8 KV heads) purely to shrink the KV-cache. When you see `num_kv_heads < num_heads` in a config, that's this optimization, baked into the architecture.

---

## The mental model to keep

```
                 COMPUTE  ◀───────────────────────▶  MEMORY
   no cache:  recompute the past every step        (cheap memory, O(N²) compute)
   KV-cache:  store the past, reuse it              (O(N) compute, BIG memory ∝ len×batch)
```

You are almost never short on compute during decode — you're short on **memory bandwidth** (reading it) and **memory capacity** (holding the cache). That's why "decode is memory-bound" ([lesson 6](06-prefill-vs-decode.md), quantified in [lesson 8](08-inference-math-and-memory.md)) and why the KV-cache sits at the center of everything.

---

## Key takeaways

- Past tokens' **K and V don't change**, so recomputing them each decode step is pure waste → naive decoding is O(N²).
- The **KV-cache** stores past K and V and appends the new token's each step → each step processes only **one token**, a huge speedup. (Q is not cached.)
- The cost is **memory** that grows as `batch × seq_len × layers × kv_heads × head_dim × 2 × dtype`.
- At scale the **KV-cache can exceed the model weights**, making it — not model size — the true limit on serving concurrency.
- Essentially every serving optimization (PagedAttention, prefix sharing, KV quantization, GQA/MQA, disaggregation) is a way to **manage KV-cache memory**. This is the hinge of the whole field.

**Next:** [Prefill vs decode →](06-prefill-vs-decode.md) — why processing the prompt and generating tokens are two different performance problems.
