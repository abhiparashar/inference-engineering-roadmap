# Phase 1 — Transformer Internals & Inference Math (Deep Dive)

> **Goal:** know *exactly* what happens, tensor by tensor, when a language model generates one token. Everything downstream — KV-cache, batching, quantization, PagedAttention, speculative decoding — is meaningless without this. This is the most important conceptual phase in the whole roadmap.

This folder is the long-form version of [Phase 1 in the ROADMAP](../../ROADMAP.md#phase-1--transformer-internals--inference-math). It teaches the transformer from the ground up in plain words, then turns the corner into *inference math*: why LLM generation is slow, why the KV-cache exists, and why decode is memory-bound. You'll run everything on a laptop — no GPU needed.

---

## Prerequisites

Finish [Phase 0](../phase-0/README.md) first, or at least be fluent in: forward pass, matrix multiply shapes, memory hierarchy / bandwidth, and "fewer bits = fewer bytes to move." Phase 1 leans on all of them, especially the memory-bandwidth idea from [Phase 0 lesson 2](../phase-0/02-memory-hierarchy.md) and [lesson 5](../phase-0/05-floating-point-and-precision.md).

You need comfort with matrix multiply *shapes* (which dimension lines up with which). If `(a, b) @ (b, c) = (a, c)` isn't automatic, watch 3Blue1Brown's linear algebra series first (linked in [GETTING-STARTED](../../GETTING-STARTED.md)).

## The big idea of this phase

An LLM is a next-token predictor. Give it some text, it outputs a probability for every possible next token; you pick one, append it, and repeat. That's the *whole* interface. The magic is in one operation — **attention** — that lets each token look back at every earlier token. And the central *engineering* problem of this whole field falls out of one consequence: generating token N would naively require redoing the work for tokens 1…N-1 every single step. The fix (the **KV-cache**) trades memory for compute, and that trade defines everything after.

By the end you'll be able to draw the exact tensor shapes flowing through an attention layer, explain the KV-cache to anyone, and compute — with real numbers — why decode is memory-bandwidth-bound.

---

## The lessons (read in order)

| # | File | What you'll be able to say afterwards |
|---|---|---|
| 1 | [Tokens & embeddings](01-tokens-and-embeddings.md) | "Text becomes token IDs becomes vectors; the model only ever sees vectors." |
| 2 | [The attention mechanism](02-attention-mechanism.md) | "Q·K decides *how much each token attends to each other token*; softmax makes it weights; ·V gathers the info. I can write the shapes." |
| 3 | [The full transformer block](03-transformer-block.md) | "A block = attention + MLP + residuals + norms; a model is N of these stacked. I know what each piece is for." |
| 4 | [Autoregressive decoding](04-autoregressive-decoding.md) | "Generation is a loop: predict next token, append, repeat. It's inherently sequential — the root of LLM latency." |
| 5 | [The KV-cache](05-kv-cache.md) | "Without caching, each step is O(N) redone work = O(N²) total. Caching K and V makes each step cheap, at the cost of memory that grows with length × batch." |
| 6 | [Prefill vs decode](06-prefill-vs-decode.md) | "Prefill = process the prompt in parallel (compute-bound). Decode = emit tokens one by one (memory-bound). Different bottlenecks → different optimizations." |
| 7 | [Sampling: turning logits into tokens](07-sampling.md) | "Greedy, temperature, top-k, top-p — how the actual next token gets chosen, and how it affects latency." |
| 8 | [Inference math & memory (the payoff)](08-inference-math-and-memory.md) | "I can compute KV-cache size, arithmetic intensity, and the time-per-token floor — and *prove* decode is memory-bound." |
| 9 | [Build: GPT-2 forward pass from scratch](09-build-gpt2-from-scratch.md) | "I reimplemented GPT-2 inference (with a KV-cache) from tensors up and matched HuggingFace's output." |
| 10 | [Exercises & exit artifact](10-exercises-and-artifacts.md) | The concrete proof that Phase 1 is done. |

---

## How to work through this phase

1. **Draw the shapes.** For lessons 2-5, keep a scratch page and write the shape of every tensor as it flows. If you can't write the shape, you don't understand the step yet. The Phase-1 self-check is literally "draw the shapes from memory."
2. **Run the snippets.** Every lesson has small NumPy/PyTorch snippets. Type and run them on tiny inputs (seq_len=4, d_model=8) so you can print and inspect full tensors.
3. **Do the build (lesson 9).** Reimplementing GPT-2's forward pass is the single best way to internalize this phase, and it's a strong portfolio piece.
4. **Produce the exit artifact (lesson 10).** No artifact = phase not finished.

**Time budget:** 1-2 weeks part-time. Lessons 2 (attention) and 5 (KV-cache) are the load-bearing ones — go slow there. Lesson 8 is where it all pays off with real numbers.

## Phase self-check (from the ROADMAP)

You're done when you can, without notes:

1. Draw the exact tensor shapes through one attention layer for `batch=2, seq_len=10, heads=8, d_model=512`. ([lesson 2](02-attention-mechanism.md))
2. Explain why decode-phase inference is **memory-bandwidth-bound** — that for each generated token you read the *entire* model's weights from memory to produce a single token, so arithmetic intensity is terrible. ([lesson 8](08-inference-math-and-memory.md))
3. Explain the KV-cache and why it turns O(N²) work into O(N) per step — and what it costs (memory ∝ length × batch). ([lesson 5](05-kv-cache.md))

---

Next after this: **[Phase 2 — GPU Architecture & Low-Level Performance](../../ROADMAP.md#phase-2--gpu-architecture--low-level-performance)**, where "memory-bound" stops being an abstraction and becomes a number you read off a profiler.
