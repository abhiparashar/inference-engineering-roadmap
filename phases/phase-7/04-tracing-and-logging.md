# 4 — Tracing and Logging a Request End-to-End

> **You'll be able to say:** "Metrics tell me *how many* and *how bad*; traces tell me *where the time went for this one request*. I can span a request across gateway → router → engine queue → prefill → decode → detokenize, attach the `gen_ai.*` attributes that make an inference trace readable, and use tail-based sampling so the 0.1% of slow requests are the ones I keep. I can join a p99 alert to three real traces via exemplars. And I can run a logging pipeline for a prompt-bearing service without creating a privacy incident: structured events, no prompt text by default, hashed identity, explicit retention."

[Lessons 1-3](01-what-to-measure.md) gave you aggregates. Aggregates cannot answer "why was *that* request 9 seconds?" — the question you get asked in every incident and every customer escalation. That gap is what tracing closes, and inference has a specific shape of trace that generic APM setups get wrong.

---

## The three pillars, sized honestly

```
  METRICS   bounded cardinality, cheap, always-on, aggregate-only
            "TTFT p99 is 2.1 s across 40 replicas"                 → lessons 1-3, 5, 6
  TRACES    per-request, sampled, structured, joinable
            "THIS request waited 1.8 s in queue behind a 32k prefill"
  LOGS      per-event, high volume, searchable, expensive at scale
            "engine preempted seq 4471 at step 122, reason: KV exhausted"

  COST RATIO, very roughly, for a service doing 100 req/s:
    metrics   ~15 metric families           → megabytes/day
    traces    100% sampled, 30 spans each   → tens of GB/day     ← sample it
    logs      1 line/request + engine logs  → GB/day             ← structure + sample it
    prompts+completions stored              → hundreds of GB/day + a legal review
```

The engineering content of this lesson is mostly "how to get the diagnostic value of 100% tracing at 1% of the cost, while keeping the requests that matter."

---

## What an inference trace looks like

A generic HTTP tracer gives you one span for the request and tells you nothing. The span tree you want:

```
 ── span: POST /v1/chat/completions ─────────────────────────────────── 9.42 s
     ├─ auth + rate limit                                      1.8 ms
     ├─ tokenize prompt                     (prompt_tokens=7,912)      41 ms
     ├─ route: pick replica                                     0.6 ms
     │    └─ attrs: policy=prefix_hash, replica=vllm-7d9-abc, cache_hint=hit
     ├─ engine: queue wait                                    1,830 ms   ◀── the culprit
     ├─ engine: prefill                     (blocks=62, cached=48)      210 ms
     │    └─ attrs: prefix_cache_hit_tokens=6,144, chunked=true, chunks=4
     ├─ engine: decode                      (steps=311)               7,290 ms
     │    ├─ event: first_token @ t=2.08 s                  ◀── TTFT lands here
     │    ├─ event: preempted @ step 122, recompute            420 ms   ◀── the ITL stall
     │    └─ attrs: avg_batch_size=38, itl_p99_ms=71
     └─ detokenize + stream                                       38 ms
```

Read the tree and the diagnosis is immediate: TTFT was dominated by **queue wait**, not by compute, and the single worst inter-token gap was a **preemption**. Those are two different fixes (admission control/capacity vs KV budget). No metric dashboard distinguishes them for an individual request; this tree does it in one glance.

### Span boundaries worth having

| Span | Start → end | Why it earns its keep |
|---|---|---|
| `gateway.request` | first byte in → last byte out | the root; carries tenant, model, route |
| `auth` / `ratelimit` | — | occasionally the surprise (a slow auth backend) |
| `tokenize` | — | scales with prompt length; single-threaded in many stacks |
| `route` | — | records the routing decision and its inputs ([Phase 6 lesson 7](../phase-6/07-prefix-aware-routing.md)) |
| `engine.queue` | admit → first schedule | separates *our* queueing from compute — the most valuable span in the tree |
| `engine.prefill` | first schedule → first token | with prefix-cache hit tokens as an attribute |
| `engine.decode` | first token → last token | with events for first token and each preemption/stall |
| `stream` | first byte to client → last | catches slow consumers and network stalls |

Cross-process propagation is the only fiddly part: the gateway must pass W3C `traceparent` to the router, and the router into the engine. vLLM supports OpenTelemetry trace propagation (`--otlp-traces-endpoint`) so engine-side spans join your trace instead of forming an orphan tree; TGI emits OTel spans natively. If your engine can't propagate, the fallback is a **request-ID join**: pass `X-Request-Id`, log it on both sides, and correlate in the log store. Less pretty, nearly as useful.

---

## Attributes: use the `gen_ai.*` conventions

OpenTelemetry has semantic conventions for GenAI. Using them rather than inventing your own names means Grafana/Jaeger/vendor UIs already know how to display and aggregate your traces.

```
  gen_ai.operation.name        "chat"
  gen_ai.system / provider     "vllm" | "openai" | ...
  gen_ai.request.model         "llama-3.1-8b-instruct"
  gen_ai.response.model        the actual served version (may differ during canary!)
  gen_ai.request.max_tokens    512
  gen_ai.request.temperature   0.7
  gen_ai.usage.input_tokens    7912
  gen_ai.usage.output_tokens   311
  gen_ai.response.finish_reasons ["stop"]

  plus the serving-specific attributes nobody standardized but everybody needs:
  inference.replica            "vllm-7d9-abc"
  inference.queue_wait_ms      1830
  inference.ttft_ms            2080
  inference.itl_p99_ms         71
  inference.batch_size_avg     38
  inference.preemptions        1
  inference.prefix_cache_hit_tokens  6144
  inference.model_version      "awq-int4-v7"     ← canary/rollback attribution (lesson 8)
  inference.tenant_class       "pro"
  inference.cost_usd_estimate  0.0041            ← lesson 9
```

Note `gen_ai.prompt` / `gen_ai.completion` exist in the conventions as **opt-in** content capture. Default them **off**. See the privacy section below; this is the single highest-risk switch in your observability config.

---

## Sampling: keep the requests that matter

100% tracing of a high-QPS inference service is expensive, and 99.9% of those traces are boring. Naive head-based sampling at 1% throws away the interesting ones with equal probability.

```
  HEAD-BASED (decide at the root, before you know the outcome)
    + trivial, cheap, no buffering
    − you keep a uniform random 1%: your p99 traces are 1%-sampled too
    ✓ use for: baseline traffic shape at 0.1-1%

  TAIL-BASED (buffer spans in a collector, decide when the trace completes)
    + keep 100% of: errors, TTFT > SLO, E2E > threshold, preempted requests, canary traffic
    + keep 0.1% of everything else as a control group
    − needs an OTel Collector with memory for the buffer window (and a decision wait
      longer than your longest request — inference traces can last minutes!)
    ✓ use for: production. This is the correct answer for inference.

  PER-TENANT / DEBUG OVERRIDE
    + a header (or tenant flag) forcing sampled=true for one customer under investigation
    ✓ use for: escalations. Cheap, and it turns a week of back-and-forth into one trace.
```

The inference-specific tail-sampling gotcha: **`decision_wait` must exceed your longest generation**, or long streaming requests get their decision made before the interesting spans arrive — systematically discarding exactly the slow traces you configured tail sampling to catch. If your `max_tokens` allows 5-minute generations, either raise `decision_wait` or cap generation length.

```yaml
# otel-collector: keep slow + errored + canary, drop the rest
processors:
  tail_sampling:
    decision_wait: 120s          # > p99.9 request duration, or you lose the tail
    num_traces: 200000
    policies:
      - name: errors
        type: status_code
        status_code: { status_codes: [ERROR] }
      - name: slow-ttft
        type: numeric_attribute
        numeric_attribute: { key: inference.ttft_ms, min_value: 500 }
      - name: preempted
        type: numeric_attribute
        numeric_attribute: { key: inference.preemptions, min_value: 1 }
      - name: canary
        type: string_attribute
        string_attribute: { key: inference.model_version, values: [".*canary.*"], enabled_regex_matching: true }
      - name: baseline
        type: probabilistic
        probabilistic: { sampling_percentage: 0.2 }
```

### Exemplars: the metrics→traces join

A histogram observation can carry a trace ID as an exemplar. In Grafana, clicking the p99 bar on the TTFT panel then jumps to an actual slow trace. This is the difference between an on-call engineer spending 40 minutes reconstructing a bad request and spending 40 seconds reading one. Requires: exemplar support on the client (`prometheus_client` supports it), `--enable-feature=exemplar-storage` on Prometheus, and the trace to have been sampled — so your tail-sampling policy and your exemplar strategy must agree, otherwise you get dangling links.

---

## Structured logging for inference

One **event per request**, emitted at completion, as JSON, with fixed keys. This is your queryable record when traces were sampled away and metrics are too coarse.

```json
{
  "ts": "2026-09-12T14:03:11.412Z",
  "level": "info",
  "event": "request.completed",
  "request_id": "01J8Z...",
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "tenant_hash": "sha256:9f2c...",
  "tenant_class": "pro",
  "route": "/v1/chat/completions",
  "model": "llama-3.1-8b-instruct",
  "model_version": "awq-int4-v7",
  "replica": "vllm-7d9-abc",
  "prompt_tokens": 7912,
  "output_tokens": 311,
  "cached_prompt_tokens": 6144,
  "queue_wait_ms": 1830,
  "ttft_ms": 2080,
  "tpot_ms": 23.4,
  "itl_p99_ms": 71,
  "e2e_ms": 9420,
  "batch_size_avg": 38,
  "preemptions": 1,
  "finish_reason": "stop",
  "outcome": "complete",
  "http_status": 200,
  "sampled_trace": true,
  "cost_usd_estimate": 0.0041
}
```

Why this exact shape:

- **Every field is a number or a bounded string** — so it aggregates in any log store, and you can compute any percentile after the fact, per tenant, without metric cardinality ([lesson 2](02-instrumenting-with-prometheus.md) explained why tenant identity lives here and not in Prometheus).
- **`request_id` + `trace_id` together** let you cross from logs to traces and back.
- **`cached_prompt_tokens`** makes prefix-cache effectiveness a queryable, per-tenant property, which is how you find the customer whose traffic pattern is destroying your hit rate ([Phase 6 lesson 7](../phase-6/07-prefix-aware-routing.md)).
- **`model_version`** is what makes a canary comparison a log query instead of a project ([lesson 8](08-canary-and-shadow-traffic.md)).
- **`cost_usd_estimate`** turns the log stream into a billing/attribution dataset ([lesson 9](09-cost-per-million-tokens.md)).
- **No prompt or completion text.**

### Engine logs are a separate stream

vLLM/TGI emit their own periodic lines (`Avg prompt throughput: ..., Avg generation throughput: ..., Running: 38 reqs, Waiting: 4 reqs, GPU KV cache usage: 84.1%`). Keep them — they are the cheapest possible engine-side record and they survive when your metrics pipeline breaks. Two rules: **ship them structured if the engine supports JSON logging**, and **never run an engine at debug log level in production** — per-step logging is a throughput regression and a storage bill at once.

---

## Privacy, or how observability creates incidents

Prompts and completions are user content. In an inference service, the observability pipeline is the most likely place for user content to leak into a system nobody audited.

| Risk | Mechanism | Control |
|---|---|---|
| Prompt text in logs | `logger.info(f"request: {prompt}")` during debugging, never removed | lint/CI rule banning prompt interpolation in log calls; log *lengths and hashes*, never text |
| Prompt text in traces | `gen_ai.prompt` content capture enabled by an SDK default or a "just for debugging" flag | default off; if on, enable per-tenant with consent, short retention, restricted access |
| PII in error messages | exception messages embedding request payloads | scrub at the collector; test with a synthetic PII request |
| API keys / tokens in headers | span attributes auto-capturing `Authorization` | header allowlist, not denylist |
| Long retention | default 30-90 days of everything | retention per stream: metrics long, traces short, content shortest (or zero) |
| Cross-tenant exposure | dashboards showing raw samples | hash tenant IDs (`tenant_hash`), aggregate by `tenant_class` |

The practical policy that works: **metrics and structured events are always on and contain no content; content capture is a separate, off-by-default, short-retention, access-controlled pipeline** used for quality evaluation with an explicit legal basis. When you need real prompts to debug quality ([lesson 8](08-canary-and-shadow-traffic.md)), you sample into that pipeline deliberately — not as a side effect of an APM default. Related class of failure worth knowing: OWASP's LLM Top 10 calls out *sensitive information disclosure* and *insecure output handling*; the logging path is squarely inside both ([Phase 9 lesson 9](../phase-9/09-security-and-multi-tenancy.md)).

---

## Where time actually goes: the attribution table

Once tracing is on, you can answer the question that matters for prioritization — for a slow request, which component owned the time?

| Dominant span | Meaning | Fix lives in |
|---|---|---|
| `engine.queue` | you are over capacity or mis-scheduling | admission control ([Phase 3 lesson 6](../phase-3/06-scheduling-policies-and-admission-control.md)), autoscaling ([Phase 6 lesson 8](../phase-6/08-autoscaling-gpu-fleets.md)) |
| `engine.prefill` | long prompts, poor prefix reuse, no chunked prefill | prefix caching ([Phase 4 lesson 6](../phase-4/06-prefix-caching-and-radix-attention.md)), chunked prefill, TP |
| `engine.decode` with normal ITL | the request is just long | nothing to fix; it's `max_tokens` |
| `engine.decode` with ITL spikes | preemption, oversized batches, a long prefill sharing steps | KV budget, `max_num_batched_tokens`, scheduling policy |
| `tokenize` | CPU-bound host work, possibly cgroup-throttled | host CPU limits ([lesson 3](03-gpu-and-host-telemetry.md)), faster tokenizer, offload |
| `stream` | slow client or network | backpressure handling, timeouts |
| `route` | routing backend (e.g. cache-state lookup) slow | router design |

Keep this table on the incident runbook page ([lesson 10](10-incident-response-and-chaos.md)). "Read the trace, find the dominant span, jump to the row" is a complete triage procedure for latency incidents.

---

## Do this now (60 minutes)

1. **Instrument your Phase-3 server with OpenTelemetry** spans for `gateway.request`, `tokenize`, `engine.queue`, `engine.prefill`, `engine.decode`, `stream`, with the `gen_ai.*` and `inference.*` attributes above. Run it against a local Jaeger (one container). Produce one screenshot-worthy trace where the dominant span is queue wait — force it by load-testing above your saturation knee.
2. **Add the structured completion event** exactly as shaped above and pipe it to a file. Then answer, with a `jq` one-liner over 500 requests: p95 TTFT for `tenant_class="pro"` and the mean `cached_prompt_tokens / prompt_tokens` ratio. That is the per-tenant SLI and cache-effectiveness report you could not get from Prometheus.
3. **Configure tail-based sampling** in an OTel Collector with the policies above, verify a slow request survives and a fast one doesn't, then deliberately set `decision_wait` *below* your longest request duration and confirm the slow traces vanish. Reverting that is the lesson: your sampling window is part of your SLO tooling.

---

**Next:** [SLOs, error budgets, and burn-rate alerting →](05-slos-and-error-budgets.md) — turning these SLIs into a promise, a budget, and the small set of alerts that are worth waking someone up for.
