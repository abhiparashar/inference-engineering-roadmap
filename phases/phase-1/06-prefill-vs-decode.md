# 6 — Prefill vs Decode (Two Phases, Two Bottlenecks)

> **You'll be able to say:** "Inference has two phases. **Prefill** processes the whole prompt in one parallel pass — lots of math on lots of tokens, so it's *compute-bound*. **Decode** emits one token per forward pass — the same weights read from memory to produce a single token, so it's *memory-bandwidth-bound*. They have different bottlenecks, so they need different optimizations, and mixing them in one server causes real problems."

This lesson names the split. [Lesson 8](08-inference-math-and-memory.md) proves it with numbers.

---

## The two phases

With a KV-cache in place ([lesson 5](05-kv-cache.md)), a request's life has exactly two shapes:

```
prompt = "Explain the KV-cache in one paragraph"   (say 8 tokens)

┌──────────────── PREFILL ────────────────┐┌────────── DECODE ──────────┐
 one forward pass over ALL 8 prompt tokens   one forward pass PER token
 fills the KV-cache with 8 entries           appends 1 KV entry each step
 produces the 1st output token               produces tokens 2, 3, 4, …
        ▲ this is your TTFT                        ▲ each gap is your TPOT
```

**Prefill** ("the prompt phase"): the prompt tokens are all known up front, so there's no sequential dependency between them — the model runs over all of them **at once**, exactly like the batched matrices in [lesson 2](02-attention-mechanism.md). One pass, `seq_len = 8`, and the KV-cache comes out populated.

**Decode** ("the generation phase"): each new token depends on the previous one ([lesson 4](04-autoregressive-decoding.md)), so tokens must come out one at a time. Every step is a forward pass with `seq_len = 1`.

That single shape difference — `seq_len = P` vs `seq_len = 1` — is the whole lesson.

---

## Why the shape change flips the bottleneck

Every layer's weights are matrices. What changes between the phases is *what you multiply them by*.

```
PREFILL:  (P tokens × d_model) @ (d_model × d_ff)   →  matrix × matrix   (GEMM)
DECODE:   (1 token  × d_model) @ (d_model × d_ff)   →  vector × matrix   (GEMV)
                 └── the weight matrix is IDENTICAL in both cases ──┘
```

Read the weight matrix once in each case. In prefill you amortize that read over **P tokens** of useful math. In decode you amortize it over **one token**. That's arithmetic intensity ([Phase 0 lesson 2](../phase-0/02-memory-hierarchy.md)): useful FLOPs per byte moved.

| | Prefill | Decode |
|---|---|---|
| Tokens per forward pass | all P prompt tokens | 1 |
| Core operation | matrix × matrix (GEMM) | vector × matrix (GEMV) |
| Arithmetic intensity | high (≈ P FLOPs per weight byte) | terrible (≈ 1 FLOP per weight byte) |
| Bottleneck | **compute** (the GPU's math units) | **memory bandwidth** (reading weights + KV-cache) |
| GPU utilization | can approach peak | often < 5% of peak FLOPs |
| Scales with | prompt length (and it's O(P²) in attention) | number of output tokens |
| User-visible metric | **TTFT** | **TPOT** |
| Cost model | FLOPs you must do | bytes you must move |

The decode row is the uncomfortable one: **to produce a single token you read every weight in the model from memory.** A 13 GB model at 2 TB/s takes ~6.5 ms per token *no matter how fast your math units are*. The GPU spends that time mostly waiting on memory. (Numbers in [lesson 8](08-inference-math-and-memory.md).)

---

## The consequence: batching fixes decode, not prefill

Here's the move that makes serving economics work. If decode's problem is "one token per weight-read," then **read the weights once and serve many sequences with them**:

```
batch = 1   :  read 13 GB → produce  1 token   →  1 token per 13 GB moved
batch = 64  :  read 13 GB → produce 64 tokens  → 64 tokens per 13 GB moved
                            └── same weight read, 64× the useful work ──┘
```

Batching turns decode's GEMV back into a GEMM (`(64, d_model) @ (d_model, d_ff)`), raising arithmetic intensity ~64×. Throughput goes up nearly linearly with batch size while per-token latency barely moves — until you become compute-bound or run out of KV-cache memory. **This is why every production serving system is, at its core, a batching machine** (Phase 3: continuous batching).

Prefill gets almost nothing from batching, because it's *already* compute-saturated: it's doing real matrix-matrix work with a big P. Batching prefills mostly just queues more work for the same busy math units.

> **The rule:** decode is a bandwidth problem you fix with **batching and fewer bytes** (quantization, GQA, KV-cache compression). Prefill is a compute problem you fix with **better kernels and fewer FLOPs** (FlashAttention, prefix caching so you don't prefill the same system prompt twice).

---

## Where each optimization lands

| Optimization | Helps prefill | Helps decode | Phase |
|---|---|---|---|
| Continuous batching | ~no | **yes, hugely** | 3 |
| Weight quantization (INT8/INT4) | mildly | **yes** (fewer bytes to read) | 4 |
| KV-cache quantization / GQA | no | **yes** (smaller cache reads + more concurrency) | 4 |
| PagedAttention | no | **yes** (fits more sequences → bigger batches) | 4 |
| Speculative decoding | no | **yes** (more tokens per pass) | 4 |
| FlashAttention | **yes** (O(P²) attention) | some | 2 |
| Prefix caching | **yes** (skip re-prefilling shared prompts) | no | 4/6 |
| CUDA graphs / kernel-launch reduction | no | **yes** (per-step overhead dominates tiny work) | 2/4 |

Notice the asymmetry: most of the famous optimizations are decode optimizations, because decode is where the time goes in a typical chat workload (a 50-token prompt and a 500-token answer means 1 prefill pass and 500 decode passes).

---

## They fight each other in one server

Real servers handle prefills and decodes *simultaneously* — someone's prompt arrives while other users are mid-generation. This creates the single most common latency bug in LLM serving:

```
decode steps ticking along nicely:   ▮ ▮ ▮ ▮ ▮ ▮            (TPOT ≈ 20 ms, smooth)
a 4,000-token prompt arrives:        ▮ ▮ ▮ ██████████ ▮ ▮   (one long prefill hogs the GPU)
                                             ▲
                              every streaming user's tokens stall here
```

A big prefill is a large, indivisible chunk of compute. Schedule it naively and every in-flight decode waits behind it — users see their token stream freeze. Fixes you'll meet later:

- **Chunked prefill** (Phase 3/4): split a long prompt into fixed-size chunks and interleave them with decode steps, so no single prefill monopolizes the GPU.
- **Prefill/decode disaggregation** (Phase 6): run prefill and decode on *separate* GPU pools, sized and tuned independently, and ship the KV-cache between them. Whole systems (DistServe, Splitwise) exist for this. It only makes sense because the two phases have genuinely different bottlenecks — which is exactly what this lesson established.

This is also why you never report a single "latency" number for an LLM service. TTFT (prefill-dominated) and TPOT (decode-dominated) move independently; an optimization can improve one and wreck the other.

---

## Sanity check with shapes

Same model, one request, prompt of 8 tokens, generating 3:

```
prefill    : x (1, 8, d_model) → logits (1, 8, vocab)   [keep only the last row]
             KV-cache: 0 → 8 entries per layer
decode t1  : x (1, 1, d_model) → logits (1, 1, vocab)   KV-cache: 8 → 9
decode t2  : x (1, 1, d_model) → logits (1, 1, vocab)   KV-cache: 9 → 10
decode t3  : x (1, 1, d_model) → logits (1, 1, vocab)   KV-cache: 10 → 11
```

Two things to internalize: prefill computes logits at every position but you **throw away all but the last** (they're only needed during training); and the KV-cache grows by exactly one entry per decode step, which is why memory grows linearly with generation length.

---

## Key takeaways

- Inference splits into **prefill** (process the whole prompt in one parallel pass) and **decode** (emit one token per pass).
- The shapes differ — `seq_len = P` vs `seq_len = 1` — which turns prefill's **GEMM** into decode's **GEMV** against the same weights.
- Therefore prefill is **compute-bound** (high arithmetic intensity) and decode is **memory-bandwidth-bound** (you read the entire model to make one token).
- **Batching** is the fix for decode: same weight read, many tokens produced. It barely helps prefill, which is already compute-saturated.
- Prefill drives **TTFT**; decode drives **TPOT**. Report and optimize them separately.
- The two phases **interfere** on a shared GPU → chunked prefill (Phase 3/4) and prefill/decode disaggregation (Phase 6).

**Next:** [Sampling: turning logits into tokens →](07-sampling.md) — the last piece of the decode loop we've been hand-waving as `argmax`.
