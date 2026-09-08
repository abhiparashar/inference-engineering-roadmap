# 3 — Operating and Tuning vLLM

> **You'll be able to say:** "Every flag maps to a mechanism from Phases 3-4. `--gpu-memory-utilization` sets the KV pool size, and the startup log tells me the resulting block count and max concurrency, which I check against `KV bytes/token × context`. `--max-num-batched-tokens` is the per-step token budget that couples prefill to decode: raise it for throughput, lower it for TPOT stability. And I diagnose from `/metrics`: `num_requests_waiting` high with `kv_cache_usage_perc` low is a *scheduling/CPU* problem; both high is a *capacity* problem; `num_preemptions` climbing is *thrash*, which means the KV pool is too small for the offered concurrency."

Reading source (lesson 2) makes you fast at this lesson, because tuning vLLM is just choosing values for the variables you already read.

> **Version note.** Defaults move between releases. The values quoted here are from the v0.2x line (`vllm/config/scheduler.py`, `vllm/config/cache.py`); always confirm with `vllm serve --help` and the startup log on *your* version rather than trusting any blog, including this one.

---

## Start it, and read the startup log properly

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 128 \
  --port 8000
```

The startup log is a capacity report, and most people scroll past it. Read these lines:

```
  Memory profiling ... model weights take X GiB; non_torch ... ; PyTorch activation peak ... ;
  the rest of the memory reserved for KV Cache is Y GiB
  GPU KV cache size: N tokens
  Maximum concurrency for 8192 tokens per request: C x
  Capturing CUDA graphs (...)                       ← this is your startup latency
```

Three numbers matter, and you should have predicted all three before launching ([Phase 4 lesson 1](../phase-4/01-what-to-optimize.md)):

```
  KV bytes/token = 2 · layers · kv_heads · head_dim · dtype_bytes         (per Phase 1 lesson 8)
  KV pool bytes  = gpu_memory_utilization · total_HBM − weights − activations − non-torch overhead
  max concurrency ≈ KV pool tokens / max_model_len
```

If the log's "maximum concurrency" is 4× lower than you expected, you have your answer for why throughput is bad, and it is *always* one of: weights bigger than you thought (dtype, LoRA, vision tower), `--max-model-len` set to the model's full 128k context, `--gpu-memory-utilization` too low, or KV dtype FP16 where FP8 would do.

**`--max-model-len` is a capacity knob, not a politeness setting.** Serving a 128k-context model with `--max-model-len 128000` when your traffic is 4k prompts costs you most of your concurrency (the *max concurrency* figure is pool ÷ max_model_len). Set it to the largest context you actually promise.

---

## Flags → mechanisms

The authoritative list is `vllm serve --help` and `vllm/engine/arg_utils.py`. These are the ones that change behavior you can measure.

### Memory and the KV pool

| Flag | Mechanism (from Phase 3/4) | How to choose |
|---|---|---|
| `--gpu-memory-utilization` (default ~0.9) | Fraction of HBM vLLM may claim; everything left after weights+activations becomes the block pool | Raise toward 0.95 on a dedicated GPU to buy concurrency; lower if anything else shares the GPU. Too high = OOM *during* a long-prompt step, not at startup |
| `--max-model-len` | Per-request context ceiling; divides the pool into "max concurrency" | Set to your real p99 context, not the model card's maximum |
| `--kv-cache-dtype fp8` (or `fp8_e4m3`/`fp8_e5m2`) | KV quantization ([Phase 4 lesson 4](../phase-4/04-kv-cache-optimization.md)) | ~2× the concurrency, small quality cost. Measure quality; measure throughput; this is often the single best flag on long-context workloads |
| `--block-size` (8/16/32/…) | Paging granularity ([Phase 4 lesson 5](../phase-4/05-paged-attention.md)) | Leave it. The block-size sweep in Phase 4 showed a wide flat region; the backend often forces a value anyway |
| `--swap-space` | CPU swap for preempted blocks | Usually leave at default; recompute normally beats swap |
| `--num-gpu-blocks-override` | Force the pool size | Debugging/repro only — e.g. shrinking the pool deliberately to reproduce preemption |

### The scheduler

| Flag | Mechanism | How to choose |
|---|---|---|
| `--max-num-batched-tokens` (default 2048 in recent versions; auto-tuned in some) | The per-step **token budget** you read in `schedule()` | Bigger = better prefill throughput and higher GPU utilization; smaller = smoother TPOT because a giant prefill can't monopolize a step. 8192 is a reasonable starting point for chat; 2048-4096 if TPOT stability is the SLO |
| `--max-num-seqs` (default 128) | Max concurrent running sequences | Cap it below what memory allows if you want lower per-user latency; raise it for batch/offline throughput. Note the *real* cap is usually KV memory, not this |
| `--enable-chunked-prefill` / `--no-enable-chunked-prefill` (on by default in V1) | Split a long prefill across steps ([Phase 3 lesson 4](../phase-3/04-continuous-batching.md)) | Keep on for interactive traffic — it's the mechanism that stops a 30k-token prompt from freezing every decoder for hundreds of ms |
| `--scheduling-policy fcfs\|priority` | Queue discipline ([Phase 3 lesson 6](../phase-3/06-scheduling-policies-and-admission-control.md)) | `priority` when you have tiers; then pass `priority` per request. Understand starvation before enabling it |
| `--long-prefill-token-threshold` | Marks a prefill "long" so it can be throttled/chunked separately | Use when a minority of huge prompts is wrecking p99 TPOT |
| `--max-num-partial-prefills` (where present) | Concurrency limit on partial prefills | Same purpose: bound how much of a step long prompts can eat |

### Caching and reuse

| Flag | Mechanism | How to choose |
|---|---|---|
| `--enable-prefix-caching` / `--no-enable-prefix-caching` (on by default in V1) | Automatic prefix caching ([Phase 4 lesson 6](../phase-4/06-prefix-caching-and-radix-attention.md)) | Leave on. Turn it *off* only to measure its value, or if every prompt is unique and you want the (tiny) hashing cost back |
| `--prefix-caching-hash-algo` | Hash used for block identity | Default is fine; relevant for multi-tenant isolation discussions |

### Model execution

| Flag | Mechanism | How to choose |
|---|---|---|
| `--quantization` / checkpoint format | Weight quantization ([Phase 4 lesson 3](../phase-4/03-quantization-methods.md)) | Usually auto-detected from the checkpoint. Verify the *kernel* chosen in the log — a format without a fast kernel on your GPU generation is a downgrade |
| `--dtype` | Compute dtype | `auto` (bf16 on modern cards). Don't fight it |
| `--enforce-eager` | Disables CUDA graphs ([Phase 2 lesson 6](../phase-2/06-overhead-bound-and-cuda-graphs.md)) | Debugging and memory-tight cases only. Costs real throughput at small batch; buys faster startup and less memory |
| `--compilation-config` / `-O` | `torch.compile` level, piecewise graphs, capture sizes | Advanced; the defaults are the product of a lot of tuning |
| `--tensor-parallel-size`, `--pipeline-parallel-size`, `--data-parallel-size` | Parallelism ([Phase 6](../../ROADMAP.md#phase-6--distributed-inference-at-scale)) | TP only when the model (plus a useful KV pool) doesn't fit; TP is not free — it adds an all-reduce per layer |
| `--speculative-config '{"method":"ngram",...}'` | Speculative decoding ([Phase 4 lesson 7](../phase-4/07-speculative-decoding.md)) | Only at low batch, and only after measuring acceptance rate. At high batch it usually *loses* |
| `--enable-lora`, `--max-loras`, `--max-lora-rank` | Multi-adapter serving | Serving 20 fine-tunes from one base model on one GPU is a large cost win; adapters add per-step overhead |
| `--limit-mm-per-prompt`, `--mm-processor-cache-gb` | Multimodal input limits and caches | Image tokens are tokens: they consume the same budget and KV |

### API / router process

| Flag | Mechanism | How to choose |
|---|---|---|
| `--api-server-count N` | Multiple API server processes in front of one engine core | Raise when the API process is CPU-bound (many streams, big JSON) — the layer-1 pressure from [lesson 1](01-the-serving-stack-landscape.md) |
| `--max-num-queued-reqs` / `--max-num-queued-tokens` | Admission control: reject instead of queueing forever | This is Phase 3 lesson 6's load shedding, as a flag. Set it. An unbounded queue turns overload into a 5-minute TTFT instead of an honest 429 |
| `--served-model-name`, `--chat-template`, `--tool-call-parser` | Protocol surface | Get the chat template right or your quality benchmarks are measuring the wrong prompt |

---

## Sizing, with arithmetic (do this before launching)

Worked example: **Qwen2.5-7B (28 layers, 4 KV heads, head_dim 128) on a 24 GB card, bf16 weights, 8k context.**

```
  weights            ≈ 7.6e9 params × 2 B            = 15.2 GB
  usable (util 0.90) = 24 × 0.90                     = 21.6 GB
  activations+overhead (measured in the log)         ≈  1.5 GB
  KV pool                                            ≈  4.9 GB

  KV bytes/token = 2 · 28 · 4 · 128 · 2 B            = 57,344 B  ≈ 57.3 KB
  pool tokens    = 4.9e9 / 57,344                    ≈ 85,000 tokens
  max concurrency at 8k ctx                          ≈ 10 sequences
  max concurrency at 2k ctx                          ≈ 41 sequences
```

Now the decisions fall out of the arithmetic instead of out of a forum post:

- 10 concurrent 8k sequences is **thin**. Options, in order of value: `--kv-cache-dtype fp8` (→ ~20), cut `--max-model-len` to what you actually serve, quantize weights to INT4/AWQ (frees ~7.6 GB → pool ~12.5 GB → ~26 seqs at 8k), or raise `--gpu-memory-utilization` to 0.95 (+1.2 GB, ~+2 seqs).
- `--max-num-seqs 128` is meaningless here; memory caps you at ~10-26. Setting it to 256 doesn't add concurrency, it just hides where the limit is.
- Compare the arithmetic to the startup log's "GPU KV cache size" and "Maximum concurrency". If they disagree by more than ~15%, find out why *before* benchmarking — usually LoRA, a vision tower, a bigger-than-expected activation peak, or a KV dtype you didn't set.

---

## `/metrics`: what to watch, and what each pattern means

vLLM exposes Prometheus metrics at `/metrics`. The core set (names are stable across recent versions):

```
  vllm:num_requests_running            gauge    in the current batch
  vllm:num_requests_waiting            gauge    queued, not yet scheduled
  vllm:kv_cache_usage_perc             gauge    fraction of the block pool in use
  vllm:num_preemptions                 counter  scheduler had to evict a running request
  vllm:prefix_cache_queries / _hits    counter  APC effectiveness → hit rate
  vllm:iteration_tokens_total          hist     tokens per engine step (your batch size, in tokens)
  vllm:time_to_first_token_seconds     hist     TTFT
  vllm:inter_token_latency_seconds     hist     ITL / TPOT
  vllm:e2e_request_latency_seconds     hist     end to end
  vllm:request_queue_time_seconds      hist     time spent waiting (the queueing part of TTFT)
  vllm:request_prefill_time_seconds    hist     prefill duration
  vllm:request_decode_time_seconds     hist     decode duration
  vllm:prompt_tokens / generation_tokens counter throughput numerators
```

Diagnosis table — this is the part to memorize:

| Symptom | `waiting` | `kv_cache_usage` | `preemptions` | Diagnosis | Action |
|---|---|---|---|---|---|
| Bad TTFT | high | **low** (<50%) | 0 | Not memory-bound: token budget too small, or API/CPU-side bottleneck | Raise `--max-num-batched-tokens`; check API-process CPU, raise `--api-server-count` |
| Bad TTFT | high | high (>90%) | 0 | Genuinely at capacity | More KV (fp8 KV, quantize, smaller `max-model-len`) or more replicas |
| Bad TPOT, sawtooth | low | high | **rising** | Preemption thrash: admitted more than memory supports | Lower `--max-num-seqs`, raise pool, enable fp8 KV |
| TPOT spikes, periodic | low | any | 0 | Long prefills colliding with decode | Lower `--max-num-batched-tokens`, ensure chunked prefill on, use `--long-prefill-token-threshold` |
| Low throughput, GPU idle | 0 | low | 0 | Not enough offered load, or client is closed-loop | Fix the benchmark ([Phase 3 lesson 7](../phase-3/07-measuring-honestly.md)) |
| Good throughput, awful p99 | high | high | any | Queueing, exactly as `1/(1−ρ)` predicts ([Phase 3 lesson 5](../phase-3/05-queueing-theory.md)) | Admission control + shedding + capacity, not tuning |

Two derived quantities worth putting on a dashboard:

- **Prefix-cache hit rate** = `prefix_cache_hits / prefix_cache_queries`. On chat/agent traffic this should be substantial; near zero means your prompts differ in the first tokens (a timestamp or a request ID at the top of the system prompt is the classic self-inflicted wound).
- **Mean tokens per step** from `iteration_tokens_total`. This is your real batch size. If it's ~1-4 while requests are waiting, you have a scheduling or budget problem, not a GPU problem.

---

## The five failure modes

1. **OOM at startup.** Weights + activation peak + pool > HBM. Lower `--gpu-memory-utilization` or `--max-model-len`, or quantize. Deterministic and easy.
2. **OOM mid-run** (rarer in V1, but real with multimodal or huge prefills). The profiling estimate for activation peak was too low for an unusual request shape. Lower `--gpu-memory-utilization`, cap `--max-num-batched-tokens`, cap image counts.
3. **Preemption thrash.** `num_preemptions` climbing steadily, throughput sagging, TPOT jittery. Each preemption throws away computed KV and recomputes it later — negative work. Fix capacity, don't tune around it.
4. **Startup/first-request latency.** CUDA-graph capture and `torch.compile` cost 30-120 s at startup ([Phase 2 lesson 6](../phase-2/06-overhead-bound-and-cuda-graphs.md)). That's a readiness-probe and autoscaling problem ([Phase 8](../../ROADMAP.md#phase-8--mlops-glue-containers-orchestration-cicd-iac)), not a bug. `--enforce-eager` trades steady-state throughput for faster starts; prefer pre-warmed replicas.
5. **Silent quality regressions.** A quantized checkpoint, an FP8 KV cache, or a wrong chat template can each cost quality invisibly while every latency metric improves. Gate model/config changes on an eval, not on vibes ([Phase 7](../../ROADMAP.md#phase-7--observability-reliability-and-cost-sre-for-inference)).

---

## A tuning procedure that terminates

```
 1. Fix the workload. Real prompt/output length distribution, real arrival process (Poisson,
    open loop). Phase 3 lesson 7. Without this, nothing below means anything.
 2. Compute expected concurrency and tokens/sec (Phase 4 lesson 1). Write it down.
 3. Launch with defaults + your --max-model-len. Compare startup log to step 2.
 4. Run the harness at a load ladder (e.g. 1, 2, 4, 8, 16 req/s). Record TTFT/TPOT p50/p99,
    throughput, and the metric set above at each point. Find the knee.
 5. Change ONE flag. Re-run. Keep it only if the knee moved in the direction of your SLO.
    Order to try: fp8 KV → max-num-batched-tokens → max-num-seqs → quantized weights →
    speculative decoding (low batch only).
 6. Stop when the binding constraint is memory capacity at 90%+ pool usage with near-zero
    preemptions and the batch is full. Then buy GPUs or go to Phase 6.
```

Anything not in this loop — block size, swap space, exotic compile flags — is noise for almost every deployment. Spending a day on `--block-size` while `--max-model-len` is 128k is the archetypal beginner mistake.

---

## Do this now (1-2 hours, GPU optional for the first two)

1. `vllm serve --help > flags.txt`; for each flag you recognize, write the Phase 3/4 lesson it comes from. Count the fraction you can explain.
2. On CPU or a small GPU, serve a tiny model (e.g. `Qwen/Qwen2.5-0.5B-Instruct`), then `curl localhost:8000/metrics` and match every `vllm:` metric to a mechanism.
3. With a GPU: run your Phase-3 harness at two loads, once with `--no-enable-prefix-caching` and once with it on, using a workload with a shared 1-2k system prompt. Report TTFT p50/p99 and hit rate. Predict the TTFT delta first from the shared fraction.
4. Deliberately break it: set `--num-gpu-blocks-override` to something tiny and watch `num_preemptions` and TPOT. Seeing thrash once makes it recognizable forever.

---

**Next:** [TGI and the router/server split →](04-tgi-and-the-router-split.md) — the same five layers, drawn along a language boundary instead of a process boundary, and what that changes.
