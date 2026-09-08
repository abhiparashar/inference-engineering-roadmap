# 10 — Exercises & Exit Artifact

> **Rule of this repo:** no artifact = phase not finished. Phase 3 proved you can operate a service. Phase 4 proves you can make it **cheap** — and that you can tell the difference between a technique that helps your workload and one that helps somebody's slide deck.

Most of this runs on a laptop. Only the quantization benchmark (Part B of [lesson 9](09-build-paged-kv-and-quant-bench.md)) really needs CUDA; use a Colab T4 or a few rented GPU-hours.

---

## Warm-up exercises

**Predict with arithmetic first, then measure, then explain the gap.** In this phase every technique has a closed-form prediction, so a missing prediction is a missing exercise.

1. **Your cost model** ([1](01-what-to-optimize.md)): fill in your GPU and target model; report `Bmax`, tok/s and `$/1M` at B = 1, 32, `Bmax`, for FP16 and for INT4+INT8-KV. Identify which term dominates `bytes_moved` in each row.
2. **The crossover** ([1](01-what-to-optimize.md), [4](04-kv-cache-optimization.md)): solve `kv_per_token · ctx · B = weight_bytes` for your model. State the `ctx·B` at which the cache overtakes the weights, and what that implies about which optimization to do first.
3. **Granularity sweep** ([2](02-quantization-fundamentals.md)): reproduce the weight-error table (per-tensor / per-channel / group-128 / group-32, symmetric and asymmetric, INT8 and INT4). Plot error vs **bits/weight including metadata** and mark the Pareto frontier.
4. **Perplexity ladder** ([2](02-quantization-fundamentals.md)): quantize all linear weights at INT8/INT4/INT3 × {per-tensor, group-128, group-32} and report Δppl. Find the configuration where naive rounding breaks.
5. **Find the outliers** ([2](02-quantization-fundamentals.md), [3](03-quantization-methods.md)): plot `absmax / p99.9` for weights and `max/median` for activations, per layer. Compare a 124M model with a ≥1B model — the pathology should grow.
6. **AWQ's scaling trick** ([3](03-quantization-methods.md)): implement `s = a^α`, sweep α ∈ [0, 1], report output error. Confirm there is an interior optimum and that α = 1 is worse than doing nothing.
7. **Mixed-precision baseline** ([3](03-quantization-methods.md)): keep the top-k% activation-salient channels in FP16, sweep k ∈ {0, 0.1, 1, 5}%, and plot error vs effective bits/weight against the AWQ curve.
8. **KV concurrency table** ([4](04-kv-cache-optimization.md)): for your GPU, compute concurrent sequences for MHA vs GQA × FP16/INT8 KV × ctx ∈ {2k, 8k, 32k}. Mark the cells that meet your target concurrency.
9. **KV quantization error** ([4](04-kv-cache-optimization.md)): reproduce the attention-output-error table; add per-channel-K + per-token-V as one scheme and confirm it beats both uniform choices.
10. **Fragmentation simulator** ([5](05-paged-attention.md)): three allocators, your workload's length distribution. Report utilization and mean concurrency, then sweep block size and find the flat region.
11. **The `max_tokens` tax** ([5](05-paged-attention.md)): in the reserve-max allocator, sweep `max_len` and show utilization ≈ `mean_len / max_len`. Convert a client's careless `max_tokens=4096` default into GB of wasted HBM.
12. **Prefix-cache economics** ([6](06-prefix-caching-and-radix-attention.md)): measure hit rate for three workloads and sweep cache capacity to find the knee. Convert the knee into GB and into the concurrent sequences it costs.
13. **Routing** ([6](06-prefix-caching-and-radix-attention.md)): compare round-robin, least-outstanding, and prefix-affinity across 4 simulated replicas. Report hit rate **and** p99 TTFT — this is a real tradeoff, not a free win.
14. **Speculative speedup surface** ([7](07-speculative-decoding.md)): plot `speedup(α, k, c)`; find optimal k for three (α, c) pairs and the break-even `c` for each α. State the k you'd configure.
15. **Measure α and c for real** ([7](07-speculative-decoding.md)): pick a draft/target pair, measure acceptance rate on code vs prose vs math, and measure the *time* ratio of one draft pass to one target pass. Compare the predicted speedup to the measured one.
16. **Losslessness check** ([7](07-speculative-decoding.md)): greedy generation with and without speculation must produce identical token sequences. If it doesn't, your acceptance rule is wrong — find it.
17. **Overhead breakdown** ([8](08-compilation-and-kernels.md)): measure step time at batch 1 and at large batch; compute the fixed per-step cost. On CUDA, add `torch.compile(mode="reduce-overhead")` and a hand-captured CUDA graph.
18. **Recompilation trap** ([8](08-compilation-and-kernels.md)): serve with varying batch sizes under `TORCH_LOGS=recompiles`, show the TTFT spikes, then bucket shapes and show them disappear.
19. **Read the block manager** ([5](05-paged-attention.md), [9](09-build-paged-kv-and-quant-bench.md)): with your own `BlockPool` open, list five things vLLM's does that yours doesn't, citing files and functions.
20. **One flag, measured** ([4](04-kv-cache-optimization.md), [6](06-prefix-caching-and-radix-attention.md)): run vLLM with and without `--enable-prefix-caching`, and with `--kv-cache-dtype fp8`, on the same harness and workload. Report TTFT, throughput and memory for all three.

Keep the code and numbers in `labs/phase4/`.

---

## Conceptual self-check (no notes)

1. Write the decode cost model from memory and use it to explain why batch 1 is ~100× more expensive per token than a well-batched, quantized configuration. ([1](01-what-to-optimize.md))
2. Why does INT4 speed up decode ~4× but prefill ~1×? ([1](01-what-to-optimize.md), [2](02-quantization-fundamentals.md))
3. What is a quantization *group*, and why is per-tensor INT4 catastrophic while group-128 INT4 is shippable? Give the numbers. ([2](02-quantization-fundamentals.md))
4. Explain outlier features and name three different strategies for handling them. ([2](02-quantization-fundamentals.md), [3](03-quantization-methods.md))
5. What does GPTQ do that round-to-nearest doesn't? What does AWQ do, and why does it need a search over α? ([3](03-quantization-methods.md))
6. Which quantization methods speed up **prefill**, and why only those? ([3](03-quantization-methods.md))
7. Compute KV bytes per token for a model of your choice, and turn it into concurrent sequences at 8k context on an 80 GB GPU. ([4](04-kv-cache-optimization.md))
8. Why does K want per-channel quantization scales while V is fine per-token? ([4](04-kv-cache-optimization.md))
9. Explain PagedAttention as OS virtual memory in under 60 seconds, including block table, page fault, refcount and copy-on-write. ([5](05-paged-attention.md))
10. Why is the default block size 16 and not 1 or 256? ([5](05-paged-attention.md))
11. Why is prefix caching *exact*, and what four conditions must hold for a hit to be legal? ([6](06-prefix-caching-and-radix-attention.md))
12. What does a radix tree buy over a flat block-hash map? ([6](06-prefix-caching-and-radix-attention.md))
13. Derive the speculative-decoding speedup formula and explain each term. When is speculation a *slowdown*? ([7](07-speculative-decoding.md))
14. Why does speculative decoding stop helping as batch size grows? ([7](07-speculative-decoding.md), [1](01-what-to-optimize.md))
15. What do CUDA graphs require, and why does that force batch-size bucketing in a serving engine? ([8](08-compilation-and-kernels.md))
16. **The keystone question:** *"Our 8B chat service costs too much. Cut inference cost 5× without hurting quality."* Answer as a procedure with arithmetic: measure the batch and the byte split, compute the crossover, then apply — in order — batching/scheduling, prefix caching (if prompts share prefixes), FP8/INT8 KV, weight quantization, paging, and only then kernels; with the predicted factor and the risk for each step, and the measurement that would confirm or refute it. ([all lessons](README.md))

Question 16 is the phase. A strong answer is ordered, quantified, and names what it would *not* do (e.g. "speculative decoding, because we're at batch 40").

---

## Exit artifact

Produce **at least Option A**; A + B is the strongest pair in the roadmap.

### Option A — Project 06: simplified PagedAttention (required)

`projects/06-paged-attention/README.md` with:

- **`paged_kv.py`**: block pool, refcounts, block tables, content-addressed prefix cache, copy-on-write — and its **passing self-test output** for all four invariants.
- **Integration** into your Phase-3 continuous-batching engine, with `pool_utilization`, `prefix_cache_hit_rate` and `cow_copies_total` exported in `/metrics`.
- **The fragmentation study**: utilization and mean concurrency for reserve-max vs oracle-contiguous vs paged, on your workload, plus the block-size sweep.
- **The prefix-sharing study**: hit rate vs cache capacity, and end-to-end TTFT with and without caching at a fixed offered load.
- **A before/after benchmark** with the Phase 3 harness: mean batch size, throughput, p50/p99 TTFT, goodput — paged vs your Phase-3 baseline, same workload, same client.
- **An honest costs section**: the gather overhead (or your paged attention step), CoW copies, and what your implementation does *not* handle (swapping, watermark, sliding-window blocks, kernel integration).
- **A vLLM mapping table** and five specific things theirs does that yours doesn't.

### Option B — Project 03: the quantization table (strongly recommended)

`projects/03-quantization-benchmark/README.md` with the six-column table from [lesson 9](09-build-paged-kv-and-quant-bench.md) Part B, two plots, and three paragraphs: predicted vs measured, where the win disappears at large batch, and what you'd ship given an SLO and a quality floor.

### Option C — Deep read (optional, cheap, high signal)

600 words: *"Why vLLM is fast, in four mechanisms."* Continuous batching, PagedAttention, prefix caching, and the kernel/graph layer — with the specific source files you read and one measured number of your own per mechanism.

---

## How you know you're ready for Phase 5

- [ ] You can write the decode cost model from memory and use it to rank optimizations for a given workload.
- [ ] You can state the KV-vs-weights crossover for your model and context.
- [ ] You can explain, with numbers, why granularity matters more than bit width below INT8.
- [ ] You can name what each quantization method does about outliers, and which ones help prefill.
- [ ] You can explain PagedAttention as virtual memory and write `slot(pos)` on a whiteboard.
- [ ] You have a **working, tested** block pool with refcounts, prefix sharing and copy-on-write.
- [ ] You can derive speculative decoding's speedup and say when it's a slowdown.
- [ ] You measured at least one technique end-to-end with your own harness, predicted the result first, and explained the gap.
- [ ] Your exit artifact is committed.

Then go to **[Phase 5 — Production Serving Frameworks](../../ROADMAP.md#phase-5--production-serving-frameworks-read-the-masters-code)**. You now know what vLLM, TGI, TensorRT-LLM and SGLang are *doing* — Phase 5 is deploying, configuring, benchmarking and reading them, and every flag you meet (`--block-size`, `--enable-prefix-caching`, `--kv-cache-dtype`, `--quantization`, `--num-speculative-tokens`, `--enforce-eager`, `--max-num-batched-tokens`) is a lesson you've already done by hand.

---

## Where these ideas come back

| Phase 4 idea | Comes back as |
|---|---|
| Decode cost model / crossover | Capacity planning and cost-per-token dashboards (Phases 6-8) |
| Quantization methods | Framework flags, checkpoint selection, quality regression gates (Phases 5, 8) |
| KV-cache economics | Long-context products, P/D disaggregation, KV transfer over the network (Phase 6) |
| PagedAttention / block manager | Reading and modifying vLLM; the KV-transfer layer in disaggregated serving (Phases 5-6) |
| Prefix caching / radix trees | Cache-aware routing, sticky load balancing, agent-loop optimization (Phases 6-7) |
| Speculative decoding | Latency tiers, on-device inference, per-request SLO classes (Phases 5-7) |
| Kernels, graphs, compilation | Engine warmup, `--enforce-eager` triage, TensorRT-LLM builds (Phase 5) |
| Predict-then-measure discipline | Every performance claim you make for the rest of your career |

Phase 3 made the GPU busy. Phase 4 made each token cheap. Phase 5 is where you stop writing engines and start operating — and extending — the ones the industry runs.
