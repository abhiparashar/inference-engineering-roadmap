# 11 — Build: An Observability Stack and a Canary/Shadow Harness

> **You'll be able to say:** "I instrumented a streaming inference server with Prometheus histograms and OpenTelemetry spans, stood up Prometheus + Grafana + Jaeger with Docker Compose, wrote recording rules and burn-rate alerts against a written SLO, and proved the alert fires and clears under injected latency. Then I built a shadow/canary proxy that mirrors real traffic to a second model version, discards its output, and emits an automated report comparing latency percentiles, throughput, output-length distribution, finish-reason mix and paired semantic similarity — with a promote/reject verdict computed against thresholds I declared in advance."

This is the Phase 7 build. Two projects, sharing one stack:

- **Part A (small, project 13)**: instrument + Prometheus + Grafana + Jaeger + alerts, proven with injected failures.
- **Part B (large, project 14)**: a shadow/canary harness with an automated quality-and-latency diff report.

Everything here runs on a **laptop with CPU-only models**. Use a tiny model (`gpt2`, `Qwen2.5-0.5B`, or the stub generator below) so the observability engineering — the actual subject — isn't gated on GPUs. Swap in a real vLLM backend later by changing one URL.

---

## Part A — The observability stack

### A1. The service under observation

Reuse your Phase-3 server ([Phase 3 lesson 8](../phase-3/08-build-naive-vs-batched-server.md)). If you want a standalone target, this is a complete instrumented stub with realistic latency behaviour, including a fault-injection endpoint you need for testing alerts.

```python
# server.py — instrumented streaming inference server (stub or real backend)
import asyncio, os, random, time, uuid, json, contextlib
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, Response
from prometheus_client import (Counter, Gauge, Histogram, CONTENT_TYPE_LATEST,
                               generate_latest)
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

MODEL_VERSION = os.environ.get("MODEL_VERSION", "stub-fp16-v1")
MODEL = os.environ.get("MODEL_NAME", "stub-0.5b")
# fault injection knobs, settable at runtime via /admin/fault
FAULT = {"extra_prefill_s": 0.0, "stall_every": 0, "error_rate": 0.0,
         "short_outputs": False}

trace.set_tracer_provider(TracerProvider(resource=Resource.create(
    {"service.name": "inference-stub", "service.version": MODEL_VERSION})))
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(
    OTLPSpanExporter(endpoint=os.environ.get("OTLP", "http://localhost:4317"),
                     insecure=True)))
tracer = trace.get_tracer(__name__)

TTFT_B = (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0, 30.0)
ITL_B  = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.2, 0.5, 1.0)
E2E_B  = (0.1, 0.5, 1, 2, 5, 10, 20, 40, 80, 160, 320)
TOK_B  = (1, 8, 32, 128, 512, 1024, 2048, 4096, 8192, 16384)

L = ["model", "model_version"]
TTFT  = Histogram("inf_ttft_seconds", "admit->first token", L, buckets=TTFT_B)
ITL   = Histogram("inf_inter_token_seconds", "gap between tokens", L, buckets=ITL_B)
E2E   = Histogram("inf_e2e_seconds", "admit->last token", L + ["outcome"], buckets=E2E_B)
QWAIT = Histogram("inf_queue_wait_seconds", "admit->schedule", L, buckets=TTFT_B)
PREFILL = Histogram("inf_prefill_seconds", "schedule->first token", L, buckets=TTFT_B)
OUTLEN = Histogram("inf_output_tokens", "output length", L, buckets=TOK_B)
INLEN  = Histogram("inf_prompt_tokens", "prompt length", L, buckets=TOK_B)
TOKENS = Counter("inf_tokens_total", "tokens", L + ["direction", "tenant_class"])
FINISH = Counter("inf_finish_total", "terminal state", L + ["reason"])
ERRORS = Counter("inf_errors_total", "errors by class", L + ["class"])
RUNNING = Gauge("inf_requests_running", "in-flight streams", L)
WAITING = Gauge("inf_requests_waiting", "queued", L)
KV      = Gauge("inf_kv_cache_usage_ratio", "simulated KV occupancy", L)
GPUS    = Gauge("inference_replica_gpus", "gpus allocated to this replica", L)
GPUS.labels(MODEL, MODEL_VERSION).set(float(os.environ.get("GPUS", "1")))

app = FastAPI()
SEM = asyncio.Semaphore(int(os.environ.get("MAX_CONCURRENCY", "8")))
MAX_QUEUE = int(os.environ.get("MAX_QUEUE", "64"))
_queued = 0

def log_event(**kw):
    print(json.dumps({"ts": time.time(), "event": "request.completed", **kw}), flush=True)

async def fake_generate(prompt_tokens: int, max_tokens: int):
    """Stub with realistic shape: prefill ~ O(prompt), decode ~ constant per token."""
    await asyncio.sleep(0.0002 * prompt_tokens + FAULT["extra_prefill_s"])
    n = max(1, max_tokens // (4 if FAULT["short_outputs"] else 1))
    for i in range(n):
        await asyncio.sleep(0.02 + random.expovariate(1 / 0.004))
        if FAULT["stall_every"] and i and i % FAULT["stall_every"] == 0:
            await asyncio.sleep(1.5)                        # a preemption-like stall
        yield f"tok{i} "

@app.post("/v1/generate")
async def generate(req: Request):
    global _queued
    body = await req.json()
    prompt = body.get("prompt", "")
    prompt_tokens = max(1, len(prompt) // 4)
    max_tokens = int(body.get("max_tokens", 64))
    tenant_class = body.get("tenant_class", "free")
    rid = str(uuid.uuid4())
    lab = (MODEL, MODEL_VERSION)

    if _queued >= MAX_QUEUE:                                # bounded queue (lesson 7)
        ERRORS.labels(*lab, "shed").inc()
        return JSONResponse({"error": "overloaded", "request_id": rid},
                            status_code=429, headers={"Retry-After": "1"})

    span = tracer.start_span("gateway.request")
    span.set_attribute("gen_ai.request.model", MODEL)
    span.set_attribute("gen_ai.response.model", MODEL_VERSION)
    span.set_attribute("gen_ai.usage.input_tokens", prompt_tokens)
    span.set_attribute("inference.tenant_class", tenant_class)
    INLEN.labels(*lab).observe(prompt_tokens)
    TOKENS.labels(*lab, "prompt", tenant_class).inc(prompt_tokens)

    t_admit = time.perf_counter()
    _queued += 1; WAITING.labels(*lab).set(_queued)

    async def stream():
        global _queued
        nonlocal span
        first = last = None
        n_out = 0
        outcome = "error"
        reason = "error"
        async with SEM:
            _queued -= 1; WAITING.labels(*lab).set(_queued)
            t_sched = time.perf_counter()
            QWAIT.labels(*lab).observe(t_sched - t_admit)
            span.set_attribute("inference.queue_wait_ms", (t_sched - t_admit) * 1000)
            RUNNING.labels(*lab).inc()
            KV.labels(*lab).set(min(1.0, (SEM._value and 0 or 0) + 0.1 +
                                   0.9 * (1 - SEM._value / 8)))
            try:
                if random.random() < FAULT["error_rate"]:
                    raise RuntimeError("injected capacity failure")
                async for tok in fake_generate(prompt_tokens, max_tokens):
                    now = time.perf_counter()
                    if first is None:
                        first = now
                        TTFT.labels(*lab).observe(now - t_admit)
                        PREFILL.labels(*lab).observe(now - t_sched)
                        span.add_event("first_token")
                        span.set_attribute("inference.ttft_ms", (now - t_admit) * 1000)
                    else:
                        ITL.labels(*lab).observe(now - last)
                    last = now; n_out += 1
                    yield tok
                outcome, reason = "complete", "stop"
            except asyncio.CancelledError:
                outcome, reason = "client_abort", "abort"
                raise
            except Exception as e:                            # noqa: BLE001
                ERRORS.labels(*lab, "capacity").inc()
                span.record_exception(e)
                outcome, reason = "error", "error"
            finally:
                RUNNING.labels(*lab).dec()
                FINISH.labels(*lab, reason).inc()
                TOKENS.labels(*lab, "output", tenant_class).inc(n_out)
                OUTLEN.labels(*lab).observe(n_out)
                if first is not None:
                    E2E.labels(*lab, outcome).observe(time.perf_counter() - t_admit)
                span.set_attribute("gen_ai.usage.output_tokens", n_out)
                span.set_attribute("inference.outcome", outcome)
                span.end()
                log_event(request_id=rid, model=MODEL, model_version=MODEL_VERSION,
                          tenant_class=tenant_class, prompt_tokens=prompt_tokens,
                          output_tokens=n_out, outcome=outcome, finish_reason=reason,
                          queue_wait_ms=round((t_sched - t_admit) * 1000, 2),
                          ttft_ms=round((first - t_admit) * 1000, 2) if first else None,
                          e2e_ms=round((time.perf_counter() - t_admit) * 1000, 2))

    return StreamingResponse(stream(), media_type="text/plain")

@app.post("/admin/fault")
async def set_fault(req: Request):
    FAULT.update(await req.json())
    return FAULT

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
def health():
    return {"ok": True, "model_version": MODEL_VERSION}
```

Points worth noticing in that code, because they are the lesson content made executable: TTFT is observed **inside** the loop at the first token; ITL is observed **per gap**; the `finally` block runs on client disconnect so aborts are counted; token counters are incremented even for partial generations; the structured completion event carries tenant identity that the metrics deliberately do not ([lesson 2](02-instrumenting-with-prometheus.md), [lesson 4](04-tracing-and-logging.md)).

### A2. The stack

```yaml
# docker-compose.yml
services:
  prometheus:
    image: prom/prometheus:latest
    command:
      - --config.file=/etc/prometheus/prometheus.yml
      - --enable-feature=exemplar-storage
      - --web.enable-lifecycle
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./rules.yml:/etc/prometheus/rules.yml:ro
    ports: ["9090:9090"]
    extra_hosts: ["host.docker.internal:host-gateway"]

  grafana:
    image: grafana/grafana:latest
    environment:
      GF_AUTH_ANONYMOUS_ENABLED: "true"
      GF_AUTH_ANONYMOUS_ORG_ROLE: Admin
    ports: ["3000:3000"]
    volumes: ["./grafana-provisioning:/etc/grafana/provisioning:ro"]

  jaeger:
    image: jaegertracing/all-in-one:latest
    environment: { COLLECTOR_OTLP_ENABLED: "true" }
    ports: ["16686:16686", "4317:4317"]

  alertmanager:
    image: prom/alertmanager:latest
    ports: ["9093:9093"]

  # optional, only if you have an NVIDIA GPU + container toolkit
  # dcgm-exporter:
  #   image: nvidia/dcgm-exporter:latest
  #   cap_add: ["SYS_ADMIN"]
  #   deploy: { resources: { reservations: { devices: [{ capabilities: ["gpu"] }] } } }
  #   ports: ["9400:9400"]
```

```yaml
# prometheus.yml
global: { scrape_interval: 5s, evaluation_interval: 5s }   # 5s for a lab; 15-30s in prod
rule_files: [ /etc/prometheus/rules.yml ]
alerting:
  alertmanagers: [ { static_configs: [ { targets: ["alertmanager:9093"] } ] } ]
scrape_configs:
  - job_name: inference
    static_configs:
      - targets: ["host.docker.internal:8000", "host.docker.internal:8001"]
  # - job_name: dcgm
  #   static_configs: [ { targets: ["dcgm-exporter:9400"] } ]
  - job_name: vllm            # when you point this at a real engine
    static_configs: [ { targets: ["host.docker.internal:8100"] } ]
```

```yaml
# rules.yml — recording rules + burn-rate alerts for a written SLO
# SLO: 99% of valid requests have TTFT <= 0.5s over 28 days  → budget = 0.01
groups:
  - name: inference-slis
    interval: 10s
    rules:
      - record: inf:ttft:good_ratio_5m
        expr: sum(rate(inf_ttft_seconds_bucket{le="0.5"}[5m]))
            / sum(rate(inf_ttft_seconds_count[5m]))
      - record: inf:ttft:good_ratio_30m
        expr: sum(rate(inf_ttft_seconds_bucket{le="0.5"}[30m]))
            / sum(rate(inf_ttft_seconds_count[30m]))
      - record: inf:ttft:good_ratio_1h
        expr: sum(rate(inf_ttft_seconds_bucket{le="0.5"}[1h]))
            / sum(rate(inf_ttft_seconds_count[1h]))
      - record: inf:ttft:good_ratio_6h
        expr: sum(rate(inf_ttft_seconds_bucket{le="0.5"}[6h]))
            / sum(rate(inf_ttft_seconds_count[6h]))
      - record: inf:ttft_seconds:p99_5m
        expr: histogram_quantile(0.99,
                sum by (le, model, model_version) (rate(inf_ttft_seconds_bucket[5m])))
      - record: inf:tpot_seconds:mean_5m
        expr: sum(rate(inf_inter_token_seconds_sum[5m]))
            / sum(rate(inf_inter_token_seconds_count[5m]))
      - record: inf:output_tokens:rate5m
        expr: sum(rate(inf_tokens_total{direction="output"}[5m]))
      - record: inf:cost_per_1m_output_tokens
        expr: (sum(inference_replica_gpus) * 3.00)
            / (sum(rate(inf_tokens_total{direction="output"}[10m])) * 3600) * 1e6

  - name: inference-slo-alerts
    rules:
      - alert: TTFTSLOBurnRateFast
        expr: (1 - inf:ttft:good_ratio_1h) > (14.4 * 0.01)
          and (1 - inf:ttft:good_ratio_5m) > (14.4 * 0.01)
        for: 1m
        labels: { severity: page, slo: ttft }
        annotations:
          summary: "TTFT SLO burning >14.4x (2% of 28d budget per hour)"
          runbook: "runbooks/ttft-slo-burn.md"
      - alert: TTFTSLOBurnRateSlow
        expr: (1 - inf:ttft:good_ratio_6h) > (6 * 0.01)
          and (1 - inf:ttft:good_ratio_30m) > (6 * 0.01)
        for: 5m
        labels: { severity: page, slo: ttft }
      - alert: KVSaturated
        expr: max(inf_kv_cache_usage_ratio) > 0.9
        for: 5m
        labels: { severity: ticket }
      - alert: QueuePersistentlyNonEmpty
        expr: avg_over_time(inf_requests_waiting[5m]) > 4
        for: 5m
        labels: { severity: ticket }
      - alert: QualityShift
        expr: (sum(rate(inf_finish_total{reason="error"}[10m]))
             / sum(rate(inf_finish_total[10m]))) > 0.01
        for: 5m
        labels: { severity: ticket }
      - alert: MetricsPipelineBlind
        expr: absent(up{job="inference"}) or min(up{job="inference"}) == 0
        for: 2m
        labels: { severity: page }
```

### A3. Prove it works (this is the deliverable, not the YAML)

```bash
# 1. baseline load
python loadgen.py --url http://localhost:8000/v1/generate --qps 8 --duration 300

# 2. inject a prefill slowdown → TTFT SLO should burn
curl -XPOST localhost:8000/admin/fault -d '{"extra_prefill_s": 0.9}'
#    watch: inf:ttft:good_ratio_5m drops, TTFTSLOBurnRateFast fires within ~2 min

# 3. clear the fault → alert must clear quickly (that's the short-window's job)
curl -XPOST localhost:8000/admin/fault -d '{"extra_prefill_s": 0.0}'

# 4. inject decode stalls → ITL p99 spikes while TTFT stays fine
curl -XPOST localhost:8000/admin/fault -d '{"stall_every": 25}'

# 5. overload → queue depth, 429s, bounded-queue behaviour
python loadgen.py --qps 60 --duration 120

# 6. blackhole observability → serving unaffected, MetricsPipelineBlind fires
docker compose stop prometheus
```

Record, for each injection: which alert fired, how long detection took, and which dashboard panel identified the cause. **That table is the Part A artifact** — it proves the stack has diagnostic power, which a screenshot of a dashboard does not.

---

## Part B — The shadow/canary harness

### B1. Design

```
                 ┌──────────────► PRIMARY  :8000  (model_version=stub-fp16-v1)
   client ──► proxy :9000  ─┤        │  response streamed back to client
                 │        └────────► response recorded (metadata only)
                 └── sample p% ────► CANARY  :8001  (model_version=stub-int4-v2)
                                       │  response DISCARDED
                                       └─► metadata + text recorded for comparison

  INVARIANTS
   1. canary failure or slowness NEVER affects the client (separate task, no await
      on the client path, hard timeout, exceptions swallowed and counted)
   2. canary load is capped (own semaphore) and shed first under pressure
   3. identical request payload → paired comparison (same prompt, same params)
   4. tool calls / side effects are stubbed in canary mode
   5. stored text lives in a separate, access-controlled path (lesson 4 privacy policy)
```

### B2. The proxy

```python
# shadow_proxy.py — mirror traffic, return only primary, record both arms
import asyncio, json, time, os, uuid
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

PRIMARY = os.environ.get("PRIMARY", "http://localhost:8000/v1/generate")
CANARY  = os.environ.get("CANARY",  "http://localhost:8001/v1/generate")
SHADOW_FRACTION = float(os.environ.get("SHADOW_FRACTION", "1.0"))
CANARY_TIMEOUT  = float(os.environ.get("CANARY_TIMEOUT", "60"))
CANARY_SEM = asyncio.Semaphore(int(os.environ.get("CANARY_CONCURRENCY", "4")))
OUT = open(os.environ.get("RESULTS", "results.jsonl"), "a")

app = FastAPI()
client = httpx.AsyncClient(timeout=httpx.Timeout(300.0))

def record(rec):
    OUT.write(json.dumps(rec) + "\n"); OUT.flush()

async def call_arm(url, payload, arm, pair_id, keep_text):
    """Stream from one arm, measuring TTFT/ITL/E2E. Returns a record."""
    t0 = time.perf_counter()
    first = last = None
    n = 0
    itls = []
    chunks = []
    status = 0
    try:
        async with client.stream("POST", url, json=payload) as r:
            status = r.status_code
            async for chunk in r.aiter_text():
                if not chunk:
                    continue
                now = time.perf_counter()
                if first is None:
                    first = now
                else:
                    itls.append(now - last)
                last = now
                n += len(chunk.split())
                if keep_text:
                    chunks.append(chunk)
                if arm == "primary":
                    yield chunk                       # only primary reaches the client
        outcome = "complete"
    except Exception as e:                            # noqa: BLE001
        outcome = f"error:{type(e).__name__}"
    finally:
        rec = {
            "pair_id": pair_id, "arm": arm, "status": status, "outcome": outcome,
            "prompt_tokens": max(1, len(payload.get("prompt", "")) // 4),
            "output_tokens": n,
            "ttft_ms": None if first is None else round((first - t0) * 1000, 2),
            "e2e_ms": round((time.perf_counter() - t0) * 1000, 2),
            "tpot_ms": None if not itls else round(1000 * sum(itls) / len(itls), 3),
            "itl_p99_ms": None if not itls else round(
                1000 * sorted(itls)[int(0.99 * (len(itls) - 1))], 3),
            "text": "".join(chunks) if keep_text else None,
        }
        record(rec)

async def run_canary(payload, pair_id):
    """Fire-and-forget with isolation: the client path never awaits this."""
    try:
        async with asyncio.timeout(CANARY_TIMEOUT):
            async with CANARY_SEM:                       # cap canary load
                async for _ in call_arm(CANARY, payload, "canary", pair_id, True):
                    pass
    except (asyncio.TimeoutError, Exception):            # noqa: BLE001
        record({"pair_id": pair_id, "arm": "canary", "outcome": "shadow_failed"})

@app.post("/v1/generate")
async def proxy(req: Request):
    payload = await req.json()
    pair_id = str(uuid.uuid4())
    if (hash(pair_id) % 1000) / 1000.0 < SHADOW_FRACTION:
        asyncio.create_task(run_canary(dict(payload), pair_id))   # isolated
    return StreamingResponse(
        call_arm(PRIMARY, payload, "primary", pair_id, True),
        media_type="text/plain")
```

The isolation properties are the engineering content: the canary runs in its **own task**, behind its **own semaphore**, under its **own timeout**, and every exception is swallowed into a counter. Test that by pointing `CANARY` at a dead port and confirming client latency is unchanged.

### B3. The analysis and the verdict

```python
# analyze.py — paired comparison + promote/reject verdict against declared thresholds
import json, math, statistics as st, sys
from collections import Counter, defaultdict

THRESHOLDS = {                      # DECLARE BEFORE LOOKING AT THE DATA
    "min_pairs": 1000,
    "ttft_p95_ratio_max": 1.15,
    "tpot_p95_ratio_max": 1.15,
    "len_ratio_range": (0.85, 1.15),
    "similarity_mean_min": 0.92,
    "error_rate_delta_max": 0.005,
    "empty_rate_max_ratio": 2.0,
}

def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    return None if not xs else xs[min(len(xs) - 1, int(q * (len(xs) - 1)))]

def load(path):
    arms = defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        if "arm" in r:
            arms[r["pair_id"]][r["arm"]] = r
    return {k: v for k, v in arms.items() if "primary" in v and "canary" in v}

def similarity(a, b):
    """Token-set cosine: dependency-free stand-in for an embedding model.
    Replace with sentence-transformers for real semantic comparison."""
    ta, tb = Counter((a or "").split()), Counter((b or "").split())
    if not ta or not tb:
        return 0.0
    dot = sum(ta[t] * tb[t] for t in ta.keys() & tb.keys())
    return dot / (math.sqrt(sum(v * v for v in ta.values()))
                  * math.sqrt(sum(v * v for v in tb.values())) or 1)

def two_prop_z(k1, n1, k2, n2):
    if min(n1, n2) == 0:
        return 0.0
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)) or 1e-12
    return (p2 - p1) / se

pairs = load(sys.argv[1] if len(sys.argv) > 1 else "results.jsonl")
P = [v["primary"] for v in pairs.values()]
C = [v["canary"] for v in pairs.values()]
n = len(pairs)

report = {
    "pairs": n,
    "ttft_p50": (pct([r["ttft_ms"] for r in P], .5), pct([r["ttft_ms"] for r in C], .5)),
    "ttft_p95": (pct([r["ttft_ms"] for r in P], .95), pct([r["ttft_ms"] for r in C], .95)),
    "ttft_p99": (pct([r["ttft_ms"] for r in P], .99), pct([r["ttft_ms"] for r in C], .99)),
    "tpot_p95": (pct([r["tpot_ms"] for r in P], .95), pct([r["tpot_ms"] for r in C], .95)),
    "itl_p99":  (pct([r["itl_p99_ms"] for r in P], .99), pct([r["itl_p99_ms"] for r in C], .99)),
    "mean_out_tokens": (st.mean(r["output_tokens"] for r in P),
                        st.mean(r["output_tokens"] for r in C)),
    "empty_rate": (sum(r["output_tokens"] == 0 for r in P) / n,
                   sum(r["output_tokens"] == 0 for r in C) / n),
    "error_rate": (sum(r["outcome"] != "complete" for r in P) / n,
                   sum(r["outcome"] != "complete" for r in C) / n),
    "similarity_mean": st.mean(similarity(a["text"], b["text"]) for a, b in zip(P, C)),
    "similarity_p05": pct([similarity(a["text"], b["text"]) for a, b in zip(P, C)], .05),
    "empty_rate_z": two_prop_z(sum(r["output_tokens"] == 0 for r in P), n,
                               sum(r["output_tokens"] == 0 for r in C), n),
}

fails = []
if n < THRESHOLDS["min_pairs"]:
    fails.append(f"insufficient sample: {n} < {THRESHOLDS['min_pairs']}")
r = report["ttft_p95"]
if r[0] and r[1] and r[1] / r[0] > THRESHOLDS["ttft_p95_ratio_max"]:
    fails.append(f"TTFT p95 ratio {r[1]/r[0]:.2f}")
r = report["tpot_p95"]
if r[0] and r[1] and r[1] / r[0] > THRESHOLDS["tpot_p95_ratio_max"]:
    fails.append(f"TPOT p95 ratio {r[1]/r[0]:.2f}")
lr = report["mean_out_tokens"][1] / max(report["mean_out_tokens"][0], 1e-9)
if not (THRESHOLDS["len_ratio_range"][0] <= lr <= THRESHOLDS["len_ratio_range"][1]):
    fails.append(f"output-length ratio {lr:.2f}")
if report["similarity_mean"] < THRESHOLDS["similarity_mean_min"]:
    fails.append(f"similarity {report['similarity_mean']:.3f}")
if report["error_rate"][1] - report["error_rate"][0] > THRESHOLDS["error_rate_delta_max"]:
    fails.append("error-rate regression")

print(json.dumps(report, indent=2, default=str))
print("\nVERDICT:", "REJECT — " + "; ".join(fails) if fails else "PROMOTE")
sys.exit(1 if fails else 0)
```

### B4. Run the comparison

```bash
# primary and canary, two versions of the same service
MODEL_VERSION=stub-fp16-v1 PORT=8000 uvicorn server:app --port 8000 &
MODEL_VERSION=stub-int4-v2 PORT=8001 uvicorn server:app --port 8001 &

# make the canary genuinely different, the way a bad quantization would be:
curl -XPOST localhost:8001/admin/fault -d '{"short_outputs": true}'

SHADOW_FRACTION=1.0 uvicorn shadow_proxy:app --port 9000 &
python loadgen.py --url http://localhost:9000/v1/generate --qps 10 --duration 180
python analyze.py results.jsonl          # exit code 1 = REJECT
```

The `short_outputs` fault is the important rehearsal: it is **fast and cheap and green on every latency metric**, and the harness must still reject it — on the output-length ratio and the similarity score. That is the entire argument for quality gates in one experiment ([lesson 8](08-canary-and-shadow-traffic.md)).

Then run the honest control: no fault on the canary, same load. The verdict must be PROMOTE, and the ratios should sit near 1.0. A harness that rejects everything is as useless as one that promotes everything; you need both experiments to trust it.

---

## Extensions (pick the ones that match your goals)

1. **Point it at real engines.** Run two vLLM instances (FP16 and AWQ/GPTQ of the same model) and rerun Part B. Now the similarity score is measuring a real quantization, and you have Phase 4's quality question answered with Phase 7's tooling.
2. **Real embeddings for similarity.** Swap the token-set cosine for `sentence-transformers` (e.g. `all-MiniLM-L6-v2`); compare the verdicts. Note the cost per comparison and decide what sampling rate you'd run in production.
3. **Wire the gate into CI.** `analyze.py`'s exit code is a build gate; add it to a workflow that starts both versions, replays a fixed prompt set, and fails the PR on regression ([Phase 8 lesson 8](../phase-8/08-ci-cd-and-benchmark-gates.md)).
4. **Add DCGM** if you have a GPU, and add the L3 replica dashboard rows from [lesson 6](06-dashboards-and-diagnosis.md), including clocks and throttle reasons; induce a throttle with `nvidia-smi -pl` and watch tokens/sec follow.
5. **Tail-based sampling.** Insert an OTel Collector with the [lesson 4](04-tracing-and-logging.md) policy set and verify slow traces survive while fast ones are dropped.
6. **Cost panel.** Add the `inf:cost_per_1m_output_tokens` recording rule to a dashboard with `offset 7d` and confirm it moves when you change concurrency ([lesson 9](09-cost-per-million-tokens.md)).

---

## Checklist before you call this done

- [ ] TTFT observed at first token; ITL observed per gap; both verified against a client-side measurement
- [ ] Client disconnect increments abort counters and records partial token counts
- [ ] A bucket boundary exists at your SLO threshold, and the exact-ratio query matches client-side truth
- [ ] Recording rules + four-row burn-rate alerts deployed; fast-burn alert **tested** firing and clearing
- [ ] Traces show the queue/prefill/decode split for a slow request; exemplar links work from the histogram panel
- [ ] L1 and L2 dashboards exist with deploy annotations and a 7-day overlay
- [ ] Structured completion events include tenant identity and cost estimate; no prompt text in metrics or default logs
- [ ] Shadow proxy proven isolated: killing the canary leaves client latency unchanged
- [ ] Analysis produces REJECT for the `short_outputs` canary and PROMOTE for the identical-config control
- [ ] Sample-size requirement computed, not guessed, and enforced by the gate

---

**Next:** [Exercises and exit artifact →](12-exercises-and-artifacts.md) — the exercise set, the self-check questions, and the one-page SLO/cost/incident document that is your Phase 7 exit artifact.
