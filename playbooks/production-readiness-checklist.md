# Playbook: Production Readiness Checklist for an Inference Service

Run through this before calling any project in this repo "done," and definitely before anything resembling a real deployment. Modeled on how MAANG-style readiness reviews are actually structured (correctness → performance → reliability → operability → cost).

## Correctness
- [ ] Output parity verified against a reference implementation (e.g. quantized model output compared to FP16 baseline on a held-out eval set — perplexity or task accuracy delta documented, not assumed acceptable).
- [ ] Edge cases handled: empty input, max-context-length input, unicode/non-English input, malformed JSON, concurrent requests to the same session.
- [ ] Deterministic behavior documented (temperature=0 reproducibility, or explicitly stated as non-deterministic and why — e.g. batching-order-dependent floating point non-associativity).

## Performance
- [ ] Benchmarked per [`benchmarking.md`](benchmarking.md): p50/p90/p99 latency at target QPS, not just throughput.
- [ ] Profiled per [`profiling.md`](profiling.md): no accidental host-device syncs, GPU utilization matches expectation at saturation.
- [ ] Batch size / dynamic batching window tuned and justified with data, not defaults.
- [ ] Memory headroom validated: what's the max concurrent sequences × max context length before OOM, and is there a graceful rejection path before that point?

## Reliability
- [ ] Timeouts set on every network call (client→server, and server→any downstream dependency).
- [ ] Graceful degradation path defined: what happens on GPU OOM mid-batch? On a downstream dependency failure? (Should reject/retry gracefully, never crash the whole process and drop all in-flight requests.)
- [ ] Health check endpoint reflects real readiness (model loaded, GPU reachable) — not just "process is up."
- [ ] Retries are idempotent-safe or explicitly disabled for non-idempotent operations.
- [ ] Load tested to failure (find the actual breaking point, not just the happy path) — see chaos/failure testing notes in [`ROADMAP.md`](../ROADMAP.md) Phase 7.

## Observability
- [ ] Metrics exposed: request latency histogram, queue depth, batch size distribution, error rate, GPU memory/utilization. (Prometheus format, per [`ROADMAP.md`](../ROADMAP.md) Phase 7.)
- [ ] Structured logging with request IDs for traceability across the request lifecycle (received → queued → batched → executed → returned).
- [ ] Dashboards exist for the golden signals (latency, traffic, errors, saturation) — not just raw metric endpoints nobody looks at.
- [ ] Alerting thresholds tied to an actual SLO, not arbitrary numbers.

## Cost
- [ ] Cost-per-1M-tokens (or cost-per-request) calculated and documented for the current config.
- [ ] GPU utilization checked — paying for idle GPU capacity is the single most common inference cost leak.
- [ ] Autoscaling (or explicit manual scaling policy) defined for traffic variance, with cold-start time accounted for.

## Rollout safety
- [ ] Canary/shadow deployment plan for any model or config change (quantization, batch size, framework version) before 100% rollout.
- [ ] Rollback plan defined and tested (not just theoretically possible).
- [ ] Versioning: model weights, tokenizer, and serving config are all pinned/versioned together (a silent tokenizer mismatch is a classic, hard-to-debug production incident).

## Documentation
- [ ] README explains: what the service does, how to run it locally, how to run the benchmark suite, known limitations.
- [ ] API contract documented (request/response schema, streaming protocol if applicable, error codes).
