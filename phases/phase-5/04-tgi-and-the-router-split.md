# 4 — TGI and the Router/Server Split

> **You'll be able to say:** "TGI splits the stack along a *language* boundary: a Rust router owns HTTP, validation, tokenization, the queue and the batching decision; Python shards own only the forward pass, reached over gRPC. That buys predictable per-request CPU cost — no GIL, no asyncio contention with the model loop — at the price of a protocol between the two halves that every new feature must cross. Its batching heuristic is explicit where vLLM's is implicit: `max_batch_prefill_tokens` bounds a prefill, `max_batch_total_tokens` is the KV budget, and `waiting_served_ratio` + `max_waiting_tokens` decide when to pause decoding to run a prefill for waiting requests."

TGI is worth a careful read even if you never deploy it, because it makes the *interrupt-decode-to-run-prefill* tradeoff visible as two tunable numbers. vLLM buries the same tradeoff inside chunked prefill and a token budget.

---

## The architecture

```
   client
     │ HTTP (OpenAI-compatible /v1/..., plus /generate, /generate_stream)
     ▼
 ┌──────────────────────────── ROUTER (Rust) ─────────────────────────────┐
 │  router/src/server.rs        axum HTTP server, SSE streaming, /metrics  │
 │  router/src/validation.rs    param + length validation, TOKENIZATION    │
 │                              (on a pool of `validation_workers` threads)│
 │  router/src/infer/mod.rs     per-request channel, response assembly     │
 │  backends/v3/src/queue.rs    the waiting queue + next_batch() policy    │
 │  backends/v3/src/backend.rs  the batching_task loop (prefill vs decode) │
 │  backends/v3/src/block_allocator.rs, radix.rs                           │
 │                              block allocation + radix prefix cache      │
 └───────────────────────────────┬────────────────────────────────────────┘
                                 │ gRPC (protobuf), one connection per shard
                                 │ backends/client/src/v3/{client,sharded_client}.rs
 ┌───────────────────────────────▼─── MODEL SERVER (Python) ──────────────┐
 │  server/text_generation_server/…    one process per TP shard            │
 │    models/flash_causal_lm.py        prefill()/decode() on a Batch        │
 │    layers/…                         attention, quantization kernels      │
 │    CUDA graphs, paged KV, FlashAttention                                 │
 └──────────────────────────────────────────────────────────────────────────┘
```

The whole point is the horizontal line. In vLLM the same line exists, but it's a *process* boundary between two Python processes ([lesson 2](02-vllm-architecture.md)); in TGI it's a *language* boundary. Both solve the identical problem: keep per-request and per-chunk CPU work off the thread that must issue the next forward pass.

### Why Rust for layer 1, specifically

Per request, the router does: parse JSON → validate (max tokens, best_of, stop sequences, grammar) → **tokenize** → enqueue → assemble tokens into SSE chunks → detokenize increments → serialize JSON per chunk. At 100 concurrent streams × 40 tok/s that's ~4,000 JSON serializations and detokenization steps per second, plus tokenization bursts on arrival.

- In Python that work is GIL-bound and directly competes with the scheduling loop; you fix it with more processes (vLLM's `--api-server-count`) and accept the IPC cost.
- In Rust it's threads with no GIL, predictable latency, and low memory per connection. Tokenization runs on a dedicated worker pool (`validation_workers`) using the Rust `tokenizers` crate.

The cost is real and worth stating plainly: **every feature must be expressed in the protocol between Rust and Python.** New sampling parameters, new modalities, new scheduler signals — each needs protobuf changes on both sides. That friction is a large part of why the fast-moving research features (novel spec-decode methods, exotic quantization, new model architectures within days) tend to land in the Python engines first.

---

## The batching decision, made explicit

Read `backends/v3/src/backend.rs::batching_task`. Structure, paraphrased:

```
loop:
   batch = queue.next_batch(...)            # form an initial batch
   prefill(batch)                           # one forward over all new requests
   loop:
       decode(batch)                        # steady state: one token per sequence
       # can we let waiting requests in?
       token_budget = max_batch_total_tokens − tokens_currently_held
       min_size = if waiting_tokens >= max_waiting_tokens { None }        # force it in
                  else { Some(batch_size × waiting_served_ratio) }        # only if enough wait
       if let Some(new) = queue.next_batch(min_size, max_size,
                                           max_batch_prefill_tokens, token_budget):
            prefill(new)                    # PAUSE decoding, run a prefill
            batch = concatenate(batch, new) # then continue decoding both
```

That inner `if` is the entire prefill/decode conflict from [Phase 3 lesson 4](../phase-3/04-continuous-batching.md), exposed as policy:

- **`waiting_served_ratio`** (default `0.3`): only interrupt decoding when the number of waiting requests is at least this fraction of the running batch. Low value = admit eagerly (better TTFT, choppier TPOT); high value = protect the running batch (smoother TPOT, worse TTFT).
- **`max_waiting_tokens`** (default `20`): a starvation guard. If waiting requests have sat through this many decode tokens, admit them regardless of the ratio. It's a fairness deadline measured in tokens rather than seconds.
- When the model supports **prefill chunking**, TGI logs that both of these are *ignored* — because chunking makes the "pause everything to prefill" decision unnecessary, exactly as it does in vLLM. Seeing that warning is a good way to learn which half of the design you're actually running.

### The token budgets

| Flag | Meaning | Default |
|---|---|---|
| `--max-batch-prefill-tokens` | Cap on tokens in a single prefill forward — the compute-bound one | `max_input_tokens + 50` |
| `--max-batch-total-tokens` | Total tokens (prompt + generated) the KV cache can hold across the batch — i.e. **the KV pool, in tokens** | inferred by probing memory at startup |
| `--max-concurrent-requests` | Router-level admission cap; beyond it clients get an error instead of an unbounded queue | 128 |
| `--max-input-tokens` / `--max-total-tokens` | Per-request limits, validated in the router before anything reaches the GPU | model-derived |
| `--max-batch-size` | Hard request-count cap (for hardware that needs padded batches) | unset |
| `--cuda-graphs` | Which batch sizes to capture graphs for | a default list |
| `--quantize` | `awq`, `gptq`, `eetq`, `bitsandbytes`, `fp8`, … | off |
| `--num-shard` / `--sharded` | Tensor parallelism across GPUs | 1 |
| `--speculate N` | Speculative tokens (n-gram / Medusa depending on model) | off |

Two mappings worth memorizing for interviews:

```
  TGI --max-batch-total-tokens   ≈ vLLM's KV pool size (in tokens), stated directly
                                   instead of derived from --gpu-memory-utilization
  TGI --max-batch-prefill-tokens ≈ vLLM's --max-num-batched-tokens, but for prefill only
                                   (vLLM has ONE budget shared by prefill and decode)
```

TGI states the pool in tokens and infers the memory; vLLM states the memory fraction and infers the tokens. Same number, opposite direction — and TGI's direction is friendlier to capacity planning, because "how many tokens can I hold" is the question you actually computed in [Phase 4 lesson 4](../phase-4/04-kv-cache-optimization.md).

---

## Startup: the warmup probe

TGI runs a **warmup** at startup that actually executes prefill and decode at the configured limits to (a) capture CUDA graphs and (b) *empirically* determine `max_batch_total_tokens` if you didn't set it: it grows the batch until it hits an OOM boundary, then backs off. This is philosophically different from vLLM's memory-profiling estimate — measured rather than modeled — and it's why TGI startup can take a while and why a too-large `--max-batch-prefill-tokens` fails loudly at startup instead of at 3 a.m. under a rare long prompt.

Operationally: **a config that survives warmup is a config that has been proven to fit.** That's a genuinely good property, and it costs startup time — the same tradeoff as TensorRT-LLM's build step ([lesson 6](06-tensorrt-llm-and-compiled-engines.md)), at a much smaller scale.

---

## Prefix caching in TGI

`backends/v3/src/radix.rs` implements a radix-tree prefix cache over blocks, with `block_allocator.rs` handling allocation; it's the same idea you built in [Phase 4 lesson 6](../phase-4/06-prefix-caching-and-radix-attention.md) and that SGLang made famous ([lesson 5](05-sglang-and-radixattention.md)). Read it if you want to see the data structure written in Rust with explicit refcounts and an LRU — it's shorter and easier to follow than the Python equivalents, and it's a good target for the "reimplement from reading" exercise.

---

## Choosing TGI, honestly

**Reasons it's the right call:**
- You're inside the HuggingFace ecosystem (Inference Endpoints, HF hub tokens, HF chat templates) and want the least ops friction for a single model.
- You want a router that shrugs off many concurrent streams without process-count tuning.
- You want warmup-proven memory limits rather than estimated ones.
- Docker image + a couple of flags is genuinely the fastest path from zero to a served model.

**Reasons it isn't:**
- Feature and model velocity is generally behind vLLM/SGLang, and the two-language protocol is why.
- The community, the OSS contribution surface, and the job-market gravity are around vLLM.
- If you plan to *modify* the scheduler, you now need Rust plus protobuf plus Python.

**The interview-grade summary:** TGI and vLLM implement the same mechanisms; they differ in where they put the boundary between "request handling" and "model execution", and therefore in what's cheap to change. Rust router = cheap concurrency, expensive features. Python engine = cheap features, concurrency you must engineer around (processes, IPC, async scheduling).

---

## Read these five files (90 minutes)

1. `router/src/validation.rs` — everything rejected before the GPU sees it. This is the list of validations *your* server should have had in [Phase 3 lesson 6](../phase-3/06-scheduling-policies-and-admission-control.md).
2. `backends/v3/src/queue.rs` — `next_batch(min_size, max_size, prefill_token_budget, token_budget)`. Note that the budget check happens per candidate request, and note what happens to a request that can never fit.
3. `backends/v3/src/backend.rs` — the `batching_task` loop above; find where `waiting_served_ratio` and `max_waiting_tokens` are used and confirm the chunking warning.
4. `backends/v3/src/block_allocator.rs` + `radix.rs` — paging and prefix cache, in ~600 readable lines.
5. `server/text_generation_server/models/flash_causal_lm.py` — the Python side: what a `Batch` is, how `filter()` removes finished sequences, and how `concatenate()` merges a new prefill into a running batch. **`filter`/`concatenate` are continuous batching**, spelled out as explicit data-structure surgery. Your Phase-3 engine did the same thing with dictionaries.

Write down, for your notes: the exact condition under which TGI pauses decoding to serve a waiting request, and the vLLM mechanism that makes that decision unnecessary.

---

**Next:** [SGLang and RadixAttention →](05-sglang-and-radixattention.md) — what happens when the prefix cache stops being an optimization and becomes the scheduler's primary input.
