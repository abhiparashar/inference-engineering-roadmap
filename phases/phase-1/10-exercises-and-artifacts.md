# 10 — Exercises & Exit Artifact

> **Rule of this repo:** no artifact = phase not finished. This file is how you *prove* Phase 1 is done — to yourself, and to a future interviewer.

Phase 1 is the conceptual keystone of the whole roadmap. Everything after this (GPU kernels, batching schedulers, PagedAttention, speculative decoding) assumes you can already draw the shapes and do the math. Be honest with yourself here.

---

## Warm-up exercises (short, run them)

Each maps to one lesson. Type the code, run it, write down what you saw.

1. **Tokenization reality check** ([lesson 1](01-tokens-and-embeddings.md)): with `tiktoken` (`gpt2` or `cl100k_base`), tokenize `"strawberry"`, `" strawberry"`, `"Strawberry"`, `"1234567890"`, and a sentence of your own. Record the token counts and IDs. Explain in one line why the leading space changes the tokenization, and why "count the r's in strawberry" is hard for an LLM.
2. **Attention by hand** ([lesson 2](02-attention-mechanism.md)): implement single-head causal attention in NumPy for `seq_len=4, d_model=8`. Print `scores` before and after masking, and confirm each softmax row sums to 1 and that row *i* has exactly `i+1` nonzero weights.
3. **The shape table** ([lesson 2](02-attention-mechanism.md)): write out, **from memory**, every tensor shape through multi-head attention for `batch=2, seq_len=10, heads=8, d_model=512`. Then check it against the lesson. Redo it a day later.
4. **Parameter census** ([lesson 3](03-transformer-block.md)): load GPT-2 small's state dict and compute what fraction of the 124M parameters live in (a) the embedding table, (b) attention projections, (c) the MLPs. State which one quantization should target first and why.
5. **Watch the cache grow** ([lesson 5](05-kv-cache.md)): with HuggingFace, generate 10 tokens using `use_cache=True` one step at a time and print the K tensor shape each step. Then compute the total cache bytes and check them against the formula from [lesson 8](08-inference-math-and-memory.md).
6. **Cache vs no cache** ([lessons 4-5](05-kv-cache.md)): time generating 100 tokens with `use_cache=True` vs `use_cache=False`. Record both times and the ratio. Plot per-token time vs position for both; identify the O(N²) curve by eye.
7. **Prefill vs decode** ([lesson 6](06-prefill-vs-decode.md)): time a single forward pass over a 512-token prompt (prefill) vs a single 1-token decode step with a warm cache. Compute tokens-processed-per-second for each. Explain the (huge) gap in terms of arithmetic intensity.
8. **Sampling knobs** ([lesson 7](07-sampling.md)): take one logits vector and print the top-10 tokens with their probabilities at `T = 0.2, 0.7, 1.0, 1.5`, then the size of the top-p nucleus at `p = 0.5, 0.9, 0.99`. Do it once for a confident position ("The capital of France is") and once for an open one ("I think that"). Explain why top-p adapts and top-k doesn't.
9. **The calculator** ([lesson 8](08-inference-math-and-memory.md)): implement `inference_math()` and run it for three models you care about (e.g. Llama-3-8B, Mistral-7B, Llama-3-70B) on a GPU you can actually rent. For each, report: weight memory, KV MB/token, max concurrent users at 4k context, batch-1 decode floor, and compute- or memory-bound.
10. **Find the KV-cache in production code** ([lab](../../labs/README.md)): in `huggingface/transformers`, open a decoder model (e.g. `modeling_llama.py`) and locate where `past_key_value` is read and updated. Quote the 3-5 lines that *are* the KV-cache. Then do the same for `CausalSelfAttention` in `karpathy/nanoGPT`.

Keep these in `labs/phase1/`. The *numbers* feed your writeup.

---

## Conceptual self-check (answer without notes)

If any answer is fuzzy, re-read the linked lesson before starting Phase 2.

1. Walk the path from a string of text to a token being emitted, naming every stage. ([1](01-tokens-and-embeddings.md), [3](03-transformer-block.md), [7](07-sampling.md))
2. What do Q, K, and V each represent, and what does each of the four steps of `softmax(QKᵀ/√d)V` do? Why divide by `√d`? ([2](02-attention-mechanism.md))
3. Why is attention **O(seq²)**, and which two production techniques exist mainly because of it? ([2](02-attention-mechanism.md))
4. What does the causal mask do, and what would break without it during generation? ([2](02-attention-mechanism.md))
5. Where do most of a transformer's parameters live, and why does that matter for decode speed? ([3](03-transformer-block.md), [8](08-inference-math-and-memory.md))
6. Why can't you parallelize the generation of one sequence's tokens? What *can* you parallelize instead? ([4](04-autoregressive-decoding.md), [6](06-prefill-vs-decode.md))
7. Explain the KV-cache: what's stored, why K and V but not Q, what it saves, and what it costs. ([5](05-kv-cache.md))
8. Write the KV-cache size formula from memory and compute it for a 32-layer, 8-KV-head, head_dim-128, FP16 model at 8k context, batch 16. ([5](05-kv-cache.md), [8](08-inference-math-and-memory.md))
9. Prefill vs decode: which is compute-bound, which is memory-bound, *why*, and which user-facing metric each drives? ([6](06-prefill-vs-decode.md))
10. Why does batching help decode enormously but prefill barely at all? ([6](06-prefill-vs-decode.md), [8](08-inference-math-and-memory.md))
11. Temperature, top-k, top-p — one sentence each, plus the order they're applied in. Why isn't `temperature=0` bit-for-bit reproducible on a busy server? ([7](07-sampling.md))
12. **The keystone question:** *prove* that decode is memory-bandwidth-bound. Chain it end to end: FLOPs per token ≈ 2×params → bytes read ≈ all weights → arithmetic intensity ≈ 1 FLOP/byte (at batch 1) → hardware ridge point ≈ 150 FLOP/byte → therefore memory-bound by ~150× → therefore time/token ≈ model_bytes ÷ bandwidth → therefore halving the bytes (quantization) nearly halves the time, and batching is the only path to the ridge point. ([8](08-inference-math-and-memory.md))

Question 12 is the one that matters. It is also, almost verbatim, a standard inference-engineering interview question. Deliver it with real numbers and you have the core of this entire field.

---

## Exit artifact (this is what "finishing Phase 1" means)

Produce **one** of the following and commit it. Option A is the strongest portfolio piece in the first half of the roadmap.

### Option A — GPT-2 from scratch, verified and measured (recommended)

The [lesson 9](09-build-gpt2-from-scratch.md) build, committed to `projects/03-gpt2-from-scratch/`:

- Working NumPy implementation: embeddings → 12 blocks → logits → sampling, plus a hand-rolled KV-cache.
- **Verification**: max logit diff vs HuggingFace (`< 1e-3`) and 20 greedy tokens matching exactly; cached and uncached generation producing identical token IDs.
- **Measurements**: cache-vs-no-cache tokens/s table, the per-token-time-vs-position plot showing the O(N²) bend, and measured KV bytes matching the lesson-8 formula.
- A README with your shape table and a "what broke and how I found it" section.

### Option B — "Why decode is memory-bound" writeup + calculator

A single markdown file, `projects/phase1-inference-math/README.md`, containing:

- Your `inference_math()` calculator, committed and runnable, with a results table for at least three real models on at least two real GPUs (A100 and H100 spec sheets are public).
- **One diagram** drawn from memory: either the attention shape flow, or the prefill-vs-decode timeline with TTFT/TPOT marked.
- A **300-word** answer to self-check question 12, with your own numbers — the full chain from `2 × params` FLOPs to "quantization makes decode ~2× faster."
- Your measured prefill-vs-decode timing from exercise 7, and an explanation of any gap between your measurement and the theoretical floor (hint: MBU, kernel-launch overhead, Python).

Do both if you can. A is the depth proof; B is the one you'll actually reuse at work.

---

## How you know you're ready for Phase 2

- [ ] You can draw the attention shape table for `batch=2, seq_len=10, heads=8, d_model=512` from memory, twice, on different days.
- [ ] You can explain the KV-cache — mechanism, benefit, cost — in 60 seconds, to a non-expert.
- [ ] You can compute KV-cache memory and max concurrency for any model config, from the formula, without looking it up.
- [ ] You can deliver self-check question 12 (the memory-bound proof) with real numbers, unprompted.
- [ ] You ran the cache-vs-no-cache benchmark and *saw* the O(N²) curve yourself.
- [ ] Your exit artifact is committed.

Then go to **[Phase 2 — GPU Architecture & Low-Level Performance](../../ROADMAP.md#phase-2--gpu-architecture--low-level-performance)**. Phase 1 gave you the *what* and the *why*; Phase 2 gives you the *where*: HBM vs SRAM, tensor cores, occupancy, the roofline model, and FlashAttention. "Memory-bound" stops being an argument you make on paper and becomes a number you read off a profiler.

---

## Where these ideas come back (so you know it wasn't busywork)

| Phase 1 idea | Comes back as |
|---|---|
| Tokens are not words | Tokenizer overhead, cost-per-token billing, stop-string edge cases (Phases 3, 9) |
| Attention's O(N²) grid | FlashAttention, long-context serving, chunked prefill (Phases 2-4) |
| MLP holds most parameters | Quantization targets; why weight bytes dominate decode (Phase 4) |
| Autoregressive loop | TTFT/TPOT metrics, streaming APIs, speculative decoding (Phases 3-4) |
| KV-cache | PagedAttention, prefix sharing, KV quantization, cache offload (Phases 4, 6) |
| Prefill vs decode | Continuous batching, chunked prefill, disaggregated serving (Phases 3, 6) |
| Sampling | Structured output/grammars, speculative decoding's rejection rule, determinism SLAs (Phases 4, 7) |
| Arithmetic intensity & the ridge point | The roofline model, MFU vs MBU, kernel profiling (Phase 2) |
| KV-cache memory formula | Capacity planning, max-batch tuning, cost-per-million-tokens (Phases 5-8) |

Phase 0 was the grammar; Phase 1 is the vocabulary and the arithmetic. From Phase 2 on, you stop learning what the machine does and start making it faster.
