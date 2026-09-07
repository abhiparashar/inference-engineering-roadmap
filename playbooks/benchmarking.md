# Playbook: Benchmarking Inference Systems

Apply this checklist to *every* project in the roadmap, not just once. Bad benchmarking methodology is the #1 way people fool themselves into thinking an optimization worked when it didn't.

## 1. Define load, not just requests
- Never report "ran 1000 requests, took X seconds, so Y req/s." That measures your *client's* concurrency, not the server's real capacity under realistic arrival patterns.
- Benchmark at a **fixed target QPS** (arrival rate) using a Poisson or constant-interval request generator, and report what happens to latency *at that load* — this is what `locust`, `vegeta`, and `k6` are built for, and it's what [ Anyscale's LLMPerf](https://github.com/ray-project/llmperf) and vLLM's own `benchmark_serving.py` do.
- Sweep QPS from low to the saturation point and plot **latency vs QPS** — this curve is the real story, not a single point.

## 2. Report the right metrics for LLM serving
- **TTFT (Time To First Token)** — dominated by prefill + queueing time. This is what users perceive as "responsiveness."
- **TPOT (Time Per Output Token)**, a.k.a. inter-token latency — dominated by decode step cost. This is what users perceive as "streaming speed."
- **End-to-end latency** = TTFT + TPOT × num_output_tokens. Reporting only this hides *which* phase regressed.
- **Throughput**: tokens/sec (input+output, or output-only — state which) across the whole system, not per-request.
- Always report **p50, p90, p99** (and ideally p999 for anything user-facing) — never just mean/average. Tail latency is what pages you at 3am in production.

## 3. Warm up before measuring
- First-request latency includes CUDA context init, kernel JIT/autotune (`torch.compile`, cuDNN autotune), and cold caches (OS page cache, CUDA graph capture). Discard a warmup window (e.g. first 30-60s or first N requests) from your measurement window.

## 4. Control your variables
- Pin: model, precision, batch config, hardware (GPU model + count), input/output length distribution, and concurrency — vary exactly **one** thing per comparison.
- Input/output length matters enormously: a benchmark with short outputs hides decode-bound bottlenecks; a benchmark with long shared prefixes hides prefix-caching wins. State your length distribution explicitly (e.g. "ShareGPT dataset, mean 200 input / 250 output tokens" — a common realistic default used by vLLM's own benchmarks).

## 5. Isolate the resource under test
- Run the load generator on a **different machine/process** than the server when possible — a busy client can itself become the bottleneck and silently cap your "server" numbers.
- Watch GPU utilization (`nvidia-smi dmon` or DCGM) *while* benchmarking — if GPU util is <90% at claimed saturation, your bottleneck is somewhere else (client, network, Python GIL, scheduler overhead), not the model.

## 6. Statistical hygiene
- Run each configuration multiple times (at least 3) and report variance, not a single run — cloud GPUs have noisy-neighbor variance.
- Use enough total requests that percentile estimates are stable — p99 from 50 requests is noise; prefer 500+ for meaningful tail numbers.

## 7. Reference implementations to imitate
- `vllm-project/vllm` → `benchmarks/benchmark_serving.py` — read this before writing your own harness; it already encodes most of the above.
- `ray-project/llmperf` — multi-provider LLM load testing tool, good for comparing hosted APIs.

## 8. Reporting template
Every benchmark writeup in this repo should include a table like:

| Config | QPS | p50 TTFT | p99 TTFT | p50 TPOT | p99 TPOT | Throughput (tok/s) | GPU Util | Notes |
|---|---|---|---|---|---|---|---|---|
| baseline (FP16, batch=1) | 1 | | | | | | | |
| dynamic batching (T=20ms) | 5 | | | | | | | |
| ... | | | | | | | | |

Plus one plot: latency (p50 and p99) on Y axis vs QPS on X axis, one line per configuration.
