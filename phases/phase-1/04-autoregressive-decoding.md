# 4 — Autoregressive Decoding (Generating One Token at a Time)

> **You'll be able to say:** "Generation is a loop: run the model, get the next-token prediction, append it, feed the whole thing back, repeat. Each token depends on all the previous ones, so it's fundamentally *sequential* — and that sequential nature is the root cause of LLM latency."

---

## The core loop

From [lesson 3](03-transformer-block.md), one forward pass turns a sequence of tokens into **logits** — a score for every possible next token, at the last position. Generation just does this repeatedly:

```
1. Start with the prompt tokens.
2. Run a forward pass → logits for the next token.
3. Pick a token from those logits (sampling — lesson 7).
4. Append the chosen token to the sequence.
5. Go to step 2, now with one more token.
6. Stop when you hit a stop token (e.g. end-of-text) or a length limit.
```

This is **autoregressive** decoding: "auto" (self) + "regressive" (depends on its own previous outputs). Each new token is conditioned on everything generated so far — including tokens the model *itself* just produced.

```
prompt:  "The capital of France is"
step 1:  run model on [The, capital, of, France, is]        → predicts " Paris"
step 2:  run model on [The, capital, of, France, is, Paris] → predicts "."
step 3:  run model on [..., Paris, .]                        → predicts <end>  → stop
```

### In code (the naive version — no cache yet)

```python
def generate(model, tokens, max_new=50):
    for _ in range(max_new):
        logits = model(tokens)          # forward pass over the WHOLE sequence
        next_logits = logits[-1]        # only the last position predicts the next token
        next_token = int(next_logits.argmax())   # greedy pick (lesson 7)
        tokens = tokens + [next_token]  # append
        if next_token == END_TOKEN:
            break
    return tokens
```

Read step 2's comment carefully: this naive loop re-runs the model over the *entire* growing sequence every single step. That waste is exactly what the KV-cache ([lesson 5](05-kv-cache.md)) eliminates — but first understand *why* the loop must be sequential at all.

---

## Why it's inherently sequential (and why that's the whole problem)

You **cannot** compute token 5 until you know token 4, because token 4 becomes part of the input that produces token 5. There's a hard dependency chain:

```
tok1 ─▶ tok2 ─▶ tok3 ─▶ tok4 ─▶ tok5 ─▶ …
   each arrow is one full forward pass; none can start before the previous finishes
```

This is completely different from a classifier ([Phase 0 lesson 4](../phase-0/04-what-is-inference.md)), which does *one* forward pass and is done. Consequences that define the field:

- **Latency scales with output length.** A 500-token answer needs ~500 sequential forward passes. If each takes 10 ms, that's ~5 seconds — and you can't parallelize your way out, because step N needs step N-1's result. This is why "make each decode step faster" (quantization, better kernels, speculative decoding) is worth so much.
- **You can't use the GPU's full width on one sequence.** A GPU wants thousands of parallel operations; generating one token for one user is a tiny amount of work that leaves the GPU mostly idle. (This is the "batch size 1 wastes >90% of the GPU" fact from [Phase 0](../phase-0/README.md).) The fix is **batching** many users' step-N together — the whole of Phase 3.
- **Two distinct cost phases appear.** Processing the initial prompt (all its tokens at once) behaves very differently from generating tokens one-by-one. That split is **prefill vs decode** ([lesson 6](06-prefill-vs-decode.md)) and it's foundational.

> **This is the sentence to remember:** LLM generation is a *sequential loop of forward passes*, so latency is dominated by *number of tokens × cost per token*, and almost every LLM inference optimization attacks one of those two factors.

---

## The two knobs: number of tokens, and cost per token

Since total time ≈ (tokens generated) × (time per token), there are only two ways to make generation faster, and every technique in Phase 4 is one of them:

| Attack | How | Where in roadmap |
|---|---|---|
| **Fewer forward passes per token** | Speculative decoding: a cheap draft model proposes several tokens, the big model verifies them in one pass | Phase 4 |
| **Cheaper forward passes** | KV-cache (don't recompute the past), quantization (fewer bytes to read), better kernels, CUDA graphs (less launch overhead) | Lessons 5-6, Phases 2 & 4 |
| **(bonus) more useful work per pass** | Batching many users into one forward pass — raises throughput, not single-user latency | Phase 3 |

Keep this table in mind; it's the map of the entire optimization half of the roadmap, and it exists *because* decoding is autoregressive.

---

## User-facing latency: TTFT and TPOT

Because generation is a loop that streams tokens out, users experience two separate latencies (you'll measure these constantly from Phase 3 on):

- **TTFT (Time To First Token):** how long until the *first* token appears. Dominated by queueing + processing the prompt (prefill). This is perceived "responsiveness."
- **TPOT (Time Per Output Token):** the average gap between subsequent tokens once streaming starts. Dominated by the per-step decode cost. This is perceived "typing speed."

```
user hits enter
   │◀────── TTFT ──────▶│◀TPOT▶│◀TPOT▶│◀TPOT▶│ …
   │   queue + prefill  │ tok1 │ tok2 │ tok3 │
```

They have *different* bottlenecks (TTFT ≈ prefill = compute-bound; TPOT ≈ decode = memory-bound), which is why you always report them separately and optimize them separately — the punchline of [lesson 6](06-prefill-vs-decode.md).

---

## Key takeaways

- **Autoregressive decoding** = a loop: forward pass → pick next token → append → repeat, until a stop token or length limit.
- It's **inherently sequential** — token N needs token N-1 — so you can't parallelize a single sequence's generation. This is the root cause of LLM latency.
- Total time ≈ **(tokens generated) × (time per token)**; every optimization reduces one of those two factors (or batches to raise throughput).
- The naive loop wastefully reprocesses the whole sequence each step → motivates the **KV-cache** (next lesson).
- Users feel two latencies: **TTFT** (first token, ≈ prefill) and **TPOT** (per subsequent token, ≈ decode). Report and optimize them separately.

**Next:** [The KV-cache →](05-kv-cache.md) — the single most important optimization in LLM serving, and it falls directly out of this loop.
