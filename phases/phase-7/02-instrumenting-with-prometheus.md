# 2 — Instrumenting with Prometheus: Histograms, Labels, and Streaming-Aware Timing

> **You'll be able to say:** "I know why a Prometheus *summary* cannot be aggregated across replicas and a *histogram* can, how `histogram_quantile` actually interpolates (and therefore how bucket choice bounds my p99 error), how to pick buckets for a metric spanning 50 ms to 300 s, and why one `user_id` label can multiply my series count by a million. I can instrument a streaming handler so TTFT is recorded at first token even when the client disconnects, and I can compute per-token SLIs from counters without dividing two unrelated rates."

[Lesson 1](01-what-to-measure.md) chose the SLIs. This lesson emits them correctly. Almost every "our p99 dashboard is wrong" incident traces to one of four mistakes made here: wrong metric type, wrong buckets, wrong labels, or timing recorded in the wrong place in an async streaming handler.

---

## The four metric types, and the only two you should use

```
  COUNTER    monotonically increasing total            tokens_total, requests_total
             → always query with rate()/increase(); never graph raw
  GAUGE      instantaneous value, up and down          num_requests_running, kv_usage
             → scrape-time snapshot: a spike between scrapes is INVISIBLE
  HISTOGRAM  pre-defined buckets + _sum + _count        ttft_seconds_bucket{le="0.5"}
             → aggregatable across replicas; quantiles computed at QUERY time
  SUMMARY    client-side quantiles + _sum + _count      ttft_seconds{quantile="0.99"}
             → NOT aggregatable. avg of two replicas' p99 is not the fleet p99.
               Use only for a single-process, never-load-balanced value. In practice: never.
```

**Rule: latency → histogram, work → counter, occupancy → gauge, quantiles → never precomputed.**

The summary trap deserves one more sentence because it is subtle and common: if replica A's p99 TTFT is 400 ms and replica B's is 4 s, there is no arithmetic on those two numbers that yields the fleet p99 — you need the underlying distributions. Histograms give you exactly that, because bucket counts *add*.

### Gauges lie between scrapes

A gauge scraped every 15 s cannot show a 2-second queue spike. If a value matters at sub-scrape resolution, back it with a counter/histogram too: `queue_wait_seconds` (histogram, recorded per request) is truth; `num_requests_waiting` (gauge) is a convenient approximation for alerts and autoscaling.

---

## Bucket design: where p99 accuracy is actually decided

`histogram_quantile()` **linearly interpolates within the bucket that contains the target rank**. Therefore:

```
  error bound on a reported quantile ≈ width of the bucket it lands in
  ⇒ you need FINE buckets where your SLO threshold is, and coarse ones elsewhere
  ⇒ a p99 that lands in bucket [1.0, 10.0) is reported as "somewhere in 1–10 s"
     and rendered as a confident-looking number. That is the classic silent failure.
```

Three rules that follow:

1. **Put a bucket boundary exactly at your SLO threshold.** If the SLO is "p99 TTFT < 500 ms," you need `le="0.5"`. Then the SLO query becomes exact and interpolation-free: `sum(rate(ttft_bucket{le="0.5"}[5m])) / sum(rate(ttft_count[5m]))` is the *true* fraction of good requests, not an estimate. This is the single most important instrumentation decision in the phase.
2. **Use ~8-14 buckets per metric, log-spaced** around the region of interest. Every bucket is a time series per label combination; 30 buckets × 20 label combos = 600 series for one metric.
3. **Bound the top.** `+Inf` catches everything above your last boundary, and anything in `+Inf` is un-quantifiable. If real E2E reaches 300 s, your last finite boundary must exceed it.

### Concrete bucket sets for inference

```python
# TTFT: SLO at 0.5 s, needs resolution 50 ms → 30 s
TTFT_BUCKETS = (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0, 30.0)
#                                      ^^^ SLO boundary

# TPOT / ITL: humans read ~5-10 tok/s; interesting range 5 ms → 1 s
TPOT_BUCKETS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.2, 0.5, 1.0)
#                                    ^^^ 30 ms/token ≈ 33 tok/s target

# E2E: dominated by output length, spans 3 orders of magnitude
E2E_BUCKETS  = (0.1, 0.5, 1, 2, 5, 10, 20, 40, 80, 160, 320)

# token counts: powers of two, because context limits are powers of two
TOKENS_BUCKETS = (1, 8, 32, 128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 131072)
```

Two notes on the modern escape hatch: **native (exponential) histograms** in Prometheus 2.40+ and OTel remove bucket design almost entirely — you configure a relative resolution instead of boundaries, and storage cost is far lower. If your stack supports them end-to-end (Prometheus with the feature flag, or Mimir/VictoriaMetrics/Thanos), use them and skip the bucket arithmetic. Until then, the boundary-at-the-SLO rule is non-negotiable. Also enable **exemplars**: a histogram observation can carry a trace ID, which is what turns "p99 is bad" into "here are three slow traces" in one click ([lesson 4](04-tracing-and-logging.md)).

---

## Label cardinality: the cost model

```
  series count = (number of metric names, incl. one per histogram bucket)
               × (product of distinct values of every label)

  EXAMPLE, ONE REPLICA:
    ttft_seconds with 11 buckets + _sum + _count           = 13 series
    × model {3} × route {2}                                = 78 series        fine
    × tenant_id {5,000}                                    = 390,000 series   NO
    × request_id {∞}                                       = unbounded        outage
```

Prometheus cost is roughly linear in *active series*; a few million series is a sizeable server, and cardinality explosions are a classic self-inflicted production incident (the monitoring system falls over precisely when you need it).

| Label | Verdict | Why |
|---|---|---|
| `model`, `model_version`, `quantization` | **yes** | small, and every diagnosis needs it |
| `route` / `endpoint` (normalized) | **yes** | `/v1/chat/completions`, not the raw path with IDs |
| `status_class`, `finish_reason`, `error_class` | **yes** | bounded, drives the error taxonomy |
| `replica` / `pod` | usually added by the scraper | needed for "one bad replica"; multiplies everything by replica count |
| `tenant_class` (free/pro/internal) | **yes** — bucketed | gives per-tier SLOs without per-tenant cardinality |
| `tenant_id`, `api_key`, `user_id` | **no** | thousands+; use logs/traces, or a separate billing pipeline ([lesson 9](09-cost-per-million-tokens.md)) |
| `prompt_length`, `output_length` | **no** as a label | that is what a histogram *is*; labelling by length is a cardinality bomb |
| `request_id`, `trace_id`, prompt text | **never** | unbounded; this is tracing/logging territory (and exemplars for the join) |

**The per-tenant question is the one real design decision.** You need per-tenant numbers for billing and for "which customer broke us," and you cannot put tenant IDs in Prometheus. The standard resolution: metrics carry `tenant_class`; per-tenant usage is emitted as **structured log events or a usage record to a data warehouse**, aggregated there. Two systems, on purpose, because their cardinality budgets differ by five orders of magnitude.

---

## Streaming-aware timing: getting TTFT right in code

The naive decorator-style timing measures the wrong thing for a streaming endpoint: the handler returns as soon as the stream starts (or only after it ends, depending on framework), and a client disconnect skips your `record()` call entirely — silently deleting your worst requests from the histogram. Survivorship bias in your own SLIs.

```python
from prometheus_client import Counter, Gauge, Histogram
import time

TTFT = Histogram("inf_ttft_seconds", "admit->first token",
                 ["model", "route"], buckets=TTFT_BUCKETS)
ITL  = Histogram("inf_inter_token_seconds", "gap between output tokens",
                 ["model"], buckets=TPOT_BUCKETS)
E2E  = Histogram("inf_e2e_seconds", "admit->last token",
                 ["model", "outcome"], buckets=E2E_BUCKETS)
QWAIT = Histogram("inf_queue_wait_seconds", "admit->first schedule",
                  ["model"], buckets=TTFT_BUCKETS)
TOK  = Counter("inf_tokens_total", "tokens processed",
               ["model", "direction", "tenant_class"])
FINISH = Counter("inf_finish_total", "terminal states", ["model", "reason"])
INFLIGHT = Gauge("inf_requests_in_flight", "streaming responses open", ["model"])

async def generate_stream(req):
    model, route = req.model, "chat"
    t_admit = time.perf_counter()
    first_token_at = None
    last_token_at = None
    n_out = 0
    outcome = "error"                      # pessimistic default
    INFLIGHT.labels(model).inc()
    try:
        TOK.labels(model, "prompt", req.tenant_class).inc(req.prompt_tokens)
        async for tok in engine.generate(req):          # engine yields tokens
            now = time.perf_counter()
            if first_token_at is None:
                first_token_at = now
                TTFT.labels(model, route).observe(now - t_admit)
            else:
                ITL.labels(model).observe(now - last_token_at)   # per-gap, not mean
            last_token_at = now
            n_out += 1
            yield tok
        outcome = "complete"
        FINISH.labels(model, req.finish_reason).inc()
    except asyncio.CancelledError:          # client hung up mid-stream
        outcome = "client_abort"
        FINISH.labels(model, "abort").inc()
        raise
    except EngineOOM:
        outcome = "capacity"
        FINISH.labels(model, "error").inc()
        raise
    finally:
        INFLIGHT.labels(model).dec()
        TOK.labels(model, "output", req.tenant_class).inc(n_out)   # partial counts too
        if first_token_at is not None:
            E2E.labels(model, outcome).observe(time.perf_counter() - t_admit)
```

The five things that block is doing deliberately:

1. **TTFT is observed at the first yielded token**, inside the loop — not after the generator completes.
2. **ITL is observed per gap**, so the histogram is a distribution over *steps*. That is what catches the stalls lesson 1 showed get amplified by output length. TPOT is then derivable as `rate(_sum)/rate(_count)`, and p99 ITL as a real quantile.
3. **`finally` runs on disconnect**, so aborted requests still contribute latency and token counts. Without this your dashboards look best exactly when users are giving up.
4. **`outcome` is a label on E2E only** (bounded: complete/client_abort/capacity/timeout), so you can exclude aborts from SLO math without losing them.
5. **Partial output tokens are billed/counted.** You spent the GPU time; cost accounting must see it ([lesson 9](09-cost-per-million-tokens.md)).

### Multiprocess servers

`prometheus_client` in a Gunicorn/Uvicorn multi-worker setup needs `PROMETHEUS_MULTIPROC_DIR` and `MultiProcessCollector`, or each worker reports its own subset and gauges become meaningless. In practice, prefer **one metrics endpoint per process with the process as a target label** (what vLLM does), or a single-process async server, over multiprocess aggregation files. Ray Serve and TGI take the "aggregate in the router/controller" route; know which model your stack uses before you debug a "missing metrics" mystery.

---

## Queries you will actually write

```promql
# TTFT p99 (interpolated — accurate only if buckets are dense near the answer)
histogram_quantile(0.99, sum by (le, model) (rate(inf_ttft_seconds_bucket[5m])))

# EXACT SLO compliance: fraction of requests under the 0.5 s boundary
  sum(rate(inf_ttft_seconds_bucket{le="0.5", model="llama-8b"}[5m]))
/ sum(rate(inf_ttft_seconds_count{model="llama-8b"}[5m]))

# TPOT (mean inter-token time) — ratio of sums, never avg() of a ratio
  sum(rate(inf_inter_token_seconds_sum[5m]))
/ sum(rate(inf_inter_token_seconds_count[5m]))

# output tokens/sec per replica — the cost denominator
sum by (pod) (rate(inf_tokens_total{direction="output"}[5m]))

# error-class breakdown, excluding intentional shedding
sum by (class) (rate(inf_errors_total{class!="shed"}[5m]))

# KV occupancy hot replicas (saturation, gauge)
max by (pod) (vllm:gpu_cache_usage_perc)

# queue depth sustained over a window — the honest scale-out signal
avg_over_time(vllm:num_requests_waiting[2m])

# tokens per request (distribution shift detector: did prompts get bigger?)
  sum(rate(inf_tokens_total{direction="prompt"}[30m]))
/ sum(rate(inf_e2e_seconds_count[30m]))
```

**The aggregation order rule**: always `sum(rate(...))` then `histogram_quantile`, never `avg(histogram_quantile(...))`. And never `avg()` a per-replica ratio; sum numerators and denominators separately, or a replica serving 3 requests distorts the fleet number as much as one serving 3,000.

### Recording rules

p99 over `[5m]` across many buckets and pods is an expensive query to run on every dashboard refresh and every alert evaluation. Precompute the handful you use constantly:

```yaml
groups:
  - name: inference-slis
    interval: 30s
    rules:
      - record: inf:ttft_seconds:p99_5m
        expr: histogram_quantile(0.99, sum by (le, model) (rate(inf_ttft_seconds_bucket[5m])))
      - record: inf:ttft:good_ratio_5m
        expr: sum by (model) (rate(inf_ttft_seconds_bucket{le="0.5"}[5m]))
            / sum by (model) (rate(inf_ttft_seconds_count[5m]))
      - record: inf:output_tokens:rate5m
        expr: sum by (model) (rate(inf_tokens_total{direction="output"}[5m]))
```

Alerts in [lesson 5](05-slos-and-error-budgets.md) reference `inf:ttft:good_ratio_5m` rather than recomputing it, which is both cheaper and — more importantly — guarantees the alert and the dashboard show the same number.

---

## Scrape mechanics that bite

| Thing | Default | Consequence |
|---|---|---|
| scrape interval | 15-60 s | you cannot see anything shorter; a 10 s stall is invisible in gauges |
| `rate()` window | must be ≥ 4× interval | `rate(x[1m])` with a 60 s scrape is noisy/empty; use `[5m]` |
| staleness | 5 min | a dead pod's series disappear; `absent()` alerts need care |
| counter reset | detected by `rate()` | a pod restart is handled, but `increase()` across restarts of *the same* series can under-count |
| metrics endpoint cost | grows with series | a 50k-series `/metrics` on a loaded engine can take hundreds of ms of CPU; keep cardinality low so scraping never competes with inference |
| histogram on the hot path | ~µs per observation | fine per request and per *token gap*; **not** fine per token per label combination if labels are many |

One inference-specific gotcha: **do not put the metrics endpoint behind the same event loop that is blocked by tokenization or a long sync call.** If `/metrics` times out during overload, your monitoring goes blind at the worst possible moment, and alerts based on `absent()` fire alongside real ones.

---

## Do this now (60 minutes)

1. **Instrument your Phase-3 server** with the streaming-aware block above: TTFT, per-gap ITL, E2E with an `outcome` label, queue wait, token counters, finish-reason counter. Verify `curl localhost:8000/metrics | grep inf_` shows buckets, and that killing a client mid-stream still increments `inf_finish_total{reason="abort"}`.
2. **Prove the bucket-boundary claim.** Load-test until p99 TTFT lands in a wide bucket, then compare `histogram_quantile(0.99, ...)` against the true p99 computed from your client-side log of every request. Add a boundary at your SLO threshold, re-run, and show the exact-ratio query now matches the client-side truth within a fraction of a percent.
3. **Cost your labels.** Run `count({__name__=~"inf_.*"})` to get your series count, then compute what it becomes at 50 replicas, and again if you added `tenant_id` with 5,000 tenants. Write the two numbers down next to your Prometheus server's memory limit; that comparison is the whole argument for keeping tenant identity in logs.

---

**Next:** [GPU and host telemetry →](03-gpu-and-host-telemetry.md) — DCGM fields worth scraping, why `GPU_UTIL` is a decoy, and the hardware error signals (XID, ECC, throttling) that precede the incidents you'd otherwise call "random slowness."
