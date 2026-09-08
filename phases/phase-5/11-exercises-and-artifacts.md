# 11 — Exercises & Exit Artifact

> **Rule of this repo:** no artifact = phase not finished. Phase 4 proved you can make tokens cheap. Phase 5 proves you can **operate and extend the systems the industry actually runs** — and that when you say "framework A is faster", you can produce the leveled configuration, the metric, and the mechanism.

Most of this is laptop work: source reading, configs, Triton on CPU, Ray Serve on CPU. Only the shootout ([lesson 10](10-build-shootout-and-ensemble.md) Part A) needs a GPU, and 4-6 rented GPU-hours covers it.

---

## Warm-up exercises

Keep the code, configs and numbers in `labs/phase5/`.

**Source reading**

1. **Ten-hop trace** ([2](02-vllm-architecture.md)): trace one request through vLLM from route handler to SSE chunk. File + function + line at a pinned commit. One page.
2. **The scheduler's contract** ([2](02-vllm-architecture.md)): list every field of `SchedulerOutput` and what the executor does with it. Which fields exist only because of chunked prefill? Which only because of spec decode?
3. **Preemption path** ([2](02-vllm-architecture.md), [3](03-vllm-in-production.md)): find the code that selects a preemption victim. Which end of the running list, and why that end? What happens to its KV blocks and its `num_computed_tokens`?
4. **Block manager diff** ([2](02-vllm-architecture.md), [Phase 4 lesson 9](../phase-4/09-build-paged-kv-and-quant-bench.md)): with your own `paged_kv.py` open, list five things vLLM's KV manager does that yours doesn't, citing file and function for each.
5. **Three prefix caches** ([2](02-vllm-architecture.md), [5](05-sglang-and-radixattention.md), [6](06-tensorrt-llm-and-compiled-engines.md)): compare vLLM's chained block hashes, SGLang's radix tree, and TRT-LLM's `blockRadixTree`. One table: key, match operation, eviction unit, protection mechanism for in-use blocks.
6. **Two schedulers** ([4](04-tgi-and-the-router-split.md)): write the exact condition under which TGI pauses decoding to prefill waiting requests (`waiting_served_ratio`, `max_waiting_tokens`), and the vLLM mechanism that makes that decision unnecessary.
7. **Capacity policies** ([6](06-tensorrt-llm-and-compiled-engines.md)): explain `MAX_UTILIZATION` vs `GUARANTEED_NO_EVICT` and compute, for your model and GPU, the concurrency each would allow at 8k `max_seq_len`. Which would you ship, and against which SLO?
8. **Validation surface** ([4](04-tgi-and-the-router-split.md)): list everything TGI's `router/src/validation.rs` rejects. Which of those checks does your Phase-3 server lack? Add the two most important.

**Operating**

9. **Flag→mechanism map** ([3](03-vllm-in-production.md)): take 25 flags from `vllm serve --help` and name the mechanism each controls and the Phase 3/4 lesson it comes from. Mark the ones you cannot explain and go read them.
10. **Predict the startup log** ([3](03-vllm-in-production.md)): before launching, compute KV bytes/token, expected pool GB, pool tokens and max concurrency. Launch; compare to the log; explain any gap over 15%.
11. **The `--max-model-len` tax** ([3](03-vllm-in-production.md)): serve the same model at `--max-model-len` ∈ {2k, 8k, 32k} and record reported max concurrency and measured throughput at fixed load. Convert the difference into $/1M tokens with the Phase 4 cost model.
12. **fp8 KV** ([3](03-vllm-in-production.md), [Phase 4 lesson 4](../phase-4/04-kv-cache-optimization.md)): `--kv-cache-dtype fp8` on/off. Report pool size, concurrency, throughput, and a quality check on a small eval. Was the predicted 2× concurrency realized?
13. **Budget sweep** ([3](03-vllm-in-production.md)): `--max-num-batched-tokens` ∈ {1024, 2048, 8192, 32768} on a mixed workload (some 4k prompts, many short ones). Plot TTFT p99 and TPOT p99 against it. Find the knee and explain both directions.
14. **Break it deliberately** ([3](03-vllm-in-production.md)): shrink the pool with `--num-gpu-blocks-override` until `vllm:num_preemptions` climbs. Record throughput and TPOT during thrash; explain why the fix is capacity, not tuning.
15. **Prefix cache economics** ([3](03-vllm-in-production.md), [5](05-sglang-and-radixattention.md)): hit rate and TTFT with prefix caching on/off, for a 1-2k shared system prompt. Then *sabotage* it by prepending a timestamp to every prompt and re-measure. Quantify the cost of that one line of application code.
16. **Cache-aware scheduling** ([5](05-sglang-and-radixattention.md)): SGLang `--schedule-policy lpm` vs `fcfs` at fixed load on a mixed two-system-prompt workload. Report hit rate, throughput, and the fairness cost (TTFT p99 spread).

**Serving layer**

17. **Dynamic batching curve** ([7](07-triton-inference-server.md)): Triton, one model, sweep `max_queue_delay_microseconds` ∈ {0, 500, 2000, 10000} at fixed rate. Plot avg batch size (`nv_inference_count / nv_inference_exec_count`) and p99 latency. This is [Phase 3 lesson 3](../phase-3/03-dynamic-batching.md)'s theory measured in production software.
18. **Instance groups** ([7](07-triton-inference-server.md)): same model, `count` ∈ {1, 2, 4}. Show where extra instances help (CPU-bound stage) and where they only add memory and contention (GPU-saturated stage).
19. **Ensemble vs client** ([7](07-triton-inference-server.md), [10](10-build-shootout-and-ensemble.md)): three-stage pipeline both ways, measured under concurrency. Split the difference into round trips, serialization, and lost cross-request batching.
20. **Autoscaling signal** ([8](08-ray-serve-and-composition.md)): Ray Serve deployment with `target_ongoing_requests` derived from your own load ladder. Measure cold-start latency at `min_replicas: 0`, then argue for the value you'd actually set in production.

**Synthesis**

21. **The shootout** ([10](10-build-shootout-and-ensemble.md)): the full leveled matrix, two workloads, two engines, prefix caching on/off.
22. **Issue archaeology** ([9](09-reading-engine-source.md)): two real issues, prediction before reading, root cause after. One written up in full.

---

## Conceptual self-check (no notes)

1. Name the five layers of a serving stack and place vLLM, TGI, SGLang, TensorRT-LLM, Triton Server and Ray Serve on them. ([1](01-the-serving-stack-landscape.md))
2. Why is Triton Inference Server not a competitor to vLLM? What is the relationship? ([1](01-the-serving-stack-landscape.md), [7](07-triton-inference-server.md))
3. Trace a request through vLLM in ten hops, naming files. ([2](02-vllm-architecture.md))
4. Why does vLLM V1 run the engine in a separate process from the API server? ([2](02-vllm-architecture.md))
5. What does "there is no prefill phase and no decode phase in the scheduler" mean, and which three features does that unification buy? ([2](02-vllm-architecture.md))
6. Given a GPU, a model and a context length, compute the KV pool and the max concurrency, and name three flags that change it. ([3](03-vllm-in-production.md))
7. `num_requests_waiting` is high and `kv_cache_usage_perc` is 35%. What's wrong, and what do you change? ([3](03-vllm-in-production.md))
8. What is preemption thrash, how do you see it in metrics, and why is tuning the wrong response? ([3](03-vllm-in-production.md))
9. Why did HuggingFace write TGI's router in Rust, in terms of per-request and per-chunk work? What did that cost them? ([4](04-tgi-and-the-router-split.md))
10. Explain `waiting_served_ratio` and `max_waiting_tokens`, and why prefill chunking makes both irrelevant. ([4](04-tgi-and-the-router-split.md))
11. What does RadixAttention add over a block-hash prefix cache — as a data structure, and as a *scheduling* input? ([5](05-sglang-and-radixattention.md))
12. How is grammar-constrained decoding implemented, and what is jump-forward decoding? Is it lossless? ([5](05-sglang-and-radixattention.md))
13. What did TensorRT-LLM's build-time specialization buy and cost, and why did NVIDIA move to a PyTorch-default runtime? ([6](06-tensorrt-llm-and-compiled-engines.md))
14. Compare `MAX_UTILIZATION` and `GUARANTEED_NO_EVICT` to the allocator study in [Phase 4 lesson 5](../phase-4/05-paged-attention.md). ([6](06-tensorrt-llm-and-compiled-engines.md))
15. Write a `config.pbtxt` dynamic-batching block from memory and explain each field's effect on p99. ([7](07-triton-inference-server.md))
16. Why does an ensemble beat a client making three calls, at concurrency? ([7](07-triton-inference-server.md))
17. Why is GPU utilization a bad autoscaling signal for inference, and what should you use? ([8](08-ray-serve-and-composition.md))
18. When do you pick Ray Serve over a Triton ensemble, and what does it cost you? ([8](08-ray-serve-and-composition.md))
19. You're handed an unfamiliar engine. Name the three greps and the four data structures you'd read first. ([9](09-reading-engine-source.md))
20. **The keystone question:** *"Our vendor says their engine is 2.3× faster than ours. Evaluate that claim."* Answer as a procedure: the 12-item leveling checklist, the workload definition, the open-loop client, the engine metrics that explain a real delta, and the specific defaults (prefix caching, chunked prefill, CUDA graphs, token budget, max context) that most often manufacture a fake one. Then state what result would make you *believe* the claim. ([10](10-build-shootout-and-ensemble.md), [1](01-the-serving-stack-landscape.md))

Question 20 is the phase. A strong answer never argues about frameworks; it argues about configuration, workload and measurement — and it names the number that would change its mind.

---

## Exit artifact

Produce **Option A**. A + B is the strongest pair; C is cheap and disproportionately valuable for interviews.

### Option A — Project 09: the framework shootout (required)

`projects/09-framework-shootout/README.md` with:

- **The leveling log**: all 12 axes, the value used per engine, and everything you couldn't level.
- **Two workloads** (shared-prefix chat; long-prompt/short-output) × ≥2 engines × prefix caching on/off, as a load ladder — not a single point.
- **The metrics table**: TTFT p50/p99, TPOT p50, ITL p99, output tokens/sec, shed rate, at each load.
- **Mechanism explanations** for every gap over ~10%, backed by engine-side metrics (mean batch size, preemptions, cache hit rate, queue time) — not adjectives.
- **Operational comparison**: startup time, memory config path, metrics quality, overload behavior, config ergonomics.
- **A prediction section**: what the Phase-4 cost model said the ceiling was, and how close each engine got.
- **Reproducibility**: exact versions/commits, exact commands, the client script, and `results.jsonl`.

### Option B — Project 10: the Triton ensemble (large)

`projects/10-triton-ensemble/README.md` with the four configs, the planned-vs-measured latency budget per stage, the bottleneck study with a fix and before/after numbers, the queue-delay sweep plot, the ensemble-vs-client comparison, and an honest limits section. ([Lesson 10](10-build-shootout-and-ensemble.md) Part B.)

### Option C — The root-cause write-up (cheap, high signal)

400-600 words on one real GitHub issue about batching or memory: measured symptom, mechanism-level root cause with file:line, evidence, fix sketch, and the test that would have caught it. ([Lesson 9](09-reading-engine-source.md).) Publish it. If it turns into a merged doc/test/metric PR, say so at the top of your CV's projects section.

---

## How you know you're ready for Phase 6

- [ ] You can name the five layers and place any framework on them.
- [ ] You can trace a request through vLLM naming files, and find the equivalent loop in an engine you've never read, in 15 minutes.
- [ ] You can size a deployment from arithmetic and check it against the startup log.
- [ ] You can diagnose queueing vs capacity vs thrash from `/metrics` alone.
- [ ] You can write a `config.pbtxt` with dynamic batching and explain every field.
- [ ] You can state when Ray Serve, a Triton ensemble, or plain replicas is the right composition, and why.
- [ ] You have run a **leveled** two-engine comparison and explained every delta.
- [ ] You have explained one real issue's root cause from the source.
- [ ] Your exit artifact is committed.

Then go to **[Phase 6 — Distributed Inference at Scale](../phase-6/README.md)**. Everything so far has been one engine on one GPU, operated well. Phase 6 is what happens when the model doesn't fit on one GPU (tensor and pipeline parallelism), when one replica can't hold the traffic (routing, and prefix-aware sticky routing that breaks naive load balancers), and when prefill and decode want opposite hardware (disaggregation). Every flag you met here — `--tensor-parallel-size`, KV connectors, cache-event streams, `cacheTransceiver` — is a Phase 6 topic you've already seen the hook for.

---

## Where these ideas come back

| Phase 5 idea | Comes back as |
|---|---|
| The five-layer model | Every architecture review and design doc you write |
| Scheduler/token budget tuning | Capacity planning, cost per token, SLO defense (Phases 6-7) |
| KV manager internals | KV transfer, offloading, disaggregated prefill/decode (Phase 6) |
| Prefix cache as a scheduling input | Cache-aware routing and sticky load balancing (Phase 6) |
| Engine `/metrics` | Dashboards, alerts, SLOs, autoscaling signals (Phases 7-8) |
| Leveling a benchmark | Every vendor claim, every regression gate in CI (Phases 7-8) |
| Triton ensembles & per-stage budgets | Multi-stage pipelines: RAG, moderation, speech (Phases 9-10) |
| Ray Serve composition & autoscaling | Multi-model fleets, cold-start strategy (Phases 6-8) |
| Reading unfamiliar engine source | OSS contributions, incident response, and the rest of your career |

Phase 4 made each token cheap. Phase 5 made you dangerous inside other people's engines. Phase 6 makes the whole thing bigger than one GPU.
