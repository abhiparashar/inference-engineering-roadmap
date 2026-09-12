# 12 — Exercises & Exit Artifact

> **Rule of this repo:** no artifact = phase not finished. Phase 8 proved you can ship one workload shape. Phase 9 proves you are an *inference* engineer rather than an LLM-serving engineer: you can classify an unfamiliar workload, choose silicon on cost per unit of work, tune retrieval to a recall target, package for a device you cannot log into, and hold a multi-stage pipeline inside an SLO while it is being attacked.

Almost everything here runs on a laptop. A GPU makes three exercises real (hardware decode, the CPU-vs-GPU crossover, MIG); a phone makes two real (on-device latency, thermal throttling); neither changes the reasoning.

---

## Warm-up exercises

**Workload shapes**

1. **Four numbers, three workloads.** For a ranking service, a video-moderation pipeline, and a voice assistant, fill in FLOPs/request, latency budget (and which clock it's measured against), per-request state, and arrival rate. Then mark which of continuous batching, PagedAttention, prefix caching, and tensor parallelism apply to each. ([1](01-the-other-inference-workloads.md))
2. **Budget ratio.** For your own Phase 3/7 server, compute per-request compute time ÷ end-to-end latency. Then do it for a stage-instrumented pipeline. State where the next optimization belongs in each. ([1](01-the-other-inference-workloads.md))
3. **Fan-out inventory.** Pick one user action in any product you know and count the model invocations it triggers, their archetypes, and their separate SLOs. ([1](01-the-other-inference-workloads.md))

**Ranking**

4. **Tail amplification, measured.** Issue N parallel lookups against a local store with 1% × 20 ms injected delay, for N = 1/10/100/300; plot request p50/p99 vs N. Then batch into multi-gets of 50 and re-plot. ([2](02-recommendation-and-ranking.md))
5. **Funnel budget.** Allocate a 20 ms budget across retrieval/filter/score/re-rank plus slack, and write the degraded output for each stage's timeout. ([2](02-recommendation-and-ranking.md))
6. **Overhead-bound, then fixed.** Run a ~5 MFLOP MLP at batch 1000 in eager mode; then with `torch.compile`; then with CUDA graphs if you have a GPU. Report the fraction of the original time that was launch overhead. ([2](02-recommendation-and-ranking.md), [Phase 2 L6](../phase-2/06-overhead-bound-and-cuda-graphs.md))
7. **Embedding quantization.** Take a 1M × 64 embedding table, quantize to int8, and report size, lookup latency, and cosine error. State which memory tier the smaller table now fits in. ([2](02-recommendation-and-ranking.md))

**Vision**

8. **Five-stage attribution.** Produce fetch/decode/resize/H2D/infer percentages for an image model at batch 1 and at your knee batch. Name the dominant stage and its fix. ([3](03-vision-serving.md))
9. **Batch knee.** Sweep batch 1→128; report latency, throughput, latency/image. Then compute the largest batch that fits a 100 ms SLO at 200 QPS. ([3](03-vision-serving.md))
10. **Break preprocessing.** Same 100 images through PIL-with-antialias, PIL-without, and OpenCV (BGR!). Report top-1 agreement and mean logit cosine against the first. Then write the golden-image CI gate. ([3](03-vision-serving.md))
11. **Channels-last.** Convert model and input to `channels_last` and re-measure at the knee. Record the free delta. ([3](03-vision-serving.md))
12. **Decode arithmetic.** Compute cores needed to decode 1000 full-HD JPEGs/sec at your measured MB/s/core, and compare to the cores available per GPU on a real instance type. ([3](03-vision-serving.md))

**Speech**

13. **RTF ladder.** Measure real-time factor for three model sizes on the same clip; convert each to streams-per-device using `0.6 / RTF`. ([4](04-speech-and-streaming.md))
14. **Write the streaming contract.** Chunk, lookahead, left context, partial policy, endpoint rule, max utterance, audio format — as an API document. ([4](04-speech-and-streaming.md))
15. **VAD win.** Report the speech-frame fraction of 10 minutes of conversational audio; that percentage is free cost reduction. ([4](04-speech-and-streaming.md))
16. **Voice-loop budget.** Sum endpoint + ASR final + LLM TTFT + TTS first audio + network with your own numbers; compare to 800 ms and name the two stages you'd attack. ([4](04-speech-and-streaming.md))

**Hardware and formats**

17. **Three runtimes, one model.** Latency, throughput, memory, accuracy for GPU FP16, CPU INT8/Q4, and one more. ([5](05-hardware-diversity.md), [8](08-model-formats-and-runtimes.md))
18. **Cost per unit of useful work** for each, at 30% and 80% utilization, with real instance prices. ([5](05-hardware-diversity.md))
19. **Crossover QPS.** Plot hourly cost vs QPS for CPU and GPU deployments of the same model; report the intersection. ([5](05-hardware-diversity.md))
20. **Prefill asymmetry.** For an LLM on CPU, report TTFT for a 1000-token prompt separately from steady tokens/sec. State whether prefill alone disqualifies CPU for your SLO. ([5](05-hardware-diversity.md))
21. **Export and verify.** Export to ONNX with explicit `dynamic_axes`; compare 200 fixed inputs against the reference (max |Δ|, mean cosine, argmax/greedy-token agreement); write down the tolerances you'd gate on. ([8](08-model-formats-and-runtimes.md))
22. **Break shape dynamism.** Re-export without `dynamic_axes`, run at a different batch size, record the exact failure. ([8](08-model-formats-and-runtimes.md))
23. **Read a partition report.** Count nodes per execution provider; then insert an exotic op and watch the graph split. Write the CI assertion that would catch it. ([8](08-model-formats-and-runtimes.md))

**Retrieval**

24. **Recall/latency frontier.** Flat vs HNSW (`efSearch` sweep) vs IVF (`nprobe` sweep) vs IVF-PQ on 100k+ real vectors: recall@10 against exact, p50/p99 each. Plot the frontier and mark your shipped config. ([6](06-vector-search-and-ann.md))
25. **Index RAM arithmetic** at N = 1M/10M/100M for fp32/fp16/int8/PQ, plus payload, 2 replicas, and rebuild headroom; convert to $/month. ([6](06-vector-search-and-ann.md))
26. **The experiment nobody runs.** Hold the LLM fixed; measure end-task quality at recall 0.99/0.95/0.90/0.80. Set your retrieval target from the result. ([6](06-vector-search-and-ann.md))
27. **Filtering crossover.** Post-filter vs pre-filter-and-exact at 50%, 5%, 0.5% selectivity; record recall, latency, and the crossover. ([6](06-vector-search-and-ann.md))

**Edge**

28. **Three precisions on device.** fp32/fp16/static-int8 of a small vision model: size, steady latency, top-1 agreement. ([7](07-edge-and-on-device.md))
29. **Cold start decomposition** (file read / session+compile / first-vs-steady inference), then add a compiled-artifact cache and re-measure. ([7](07-edge-and-on-device.md))
30. **Sustained vs burst.** Loop inference for 5 minutes, plot latency over time, report the plateau. ([7](07-edge-and-on-device.md))
31. **Update plan.** One page: hosting, signing, version reporting, staged cohorts, kill switch, bundled fallback. ([7](07-edge-and-on-device.md))

**Security and pipelines**

32. **Abuse table.** Five cheapest ways a client burns 100× the intended cost, and the specific limit that stops each. Then implement the four-dimension token bucket and prove clean 429s. ([9](09-security-and-multi-tenancy.md))
33. **Attack your own guardrail.** Ten adversarial inputs against your own detector; record bypasses; keep them as a regression suite. ([9](09-security-and-multi-tenancy.md))
34. **Trust-boundary diagram.** Label every input trusted/untrusted, every sink, and where authorization happens; mark the one place indirect injection could cross users. ([9](09-security-and-multi-tenancy.md))
35. **Log audit.** Grep your own logs/traces for prompt or completion content; decide content-free or gated pipeline for each hit. ([9](09-security-and-multi-tenancy.md), [Phase 7 L4](../phase-7/04-tracing-and-logging.md))
36. **Isolation test.** If you have MIG-capable hardware, run two workloads under time-slicing and under MIG; OOM one and record the effect on the other. Without the hardware, write the table of what each mechanism isolates. ([9](09-security-and-multi-tenancy.md))
37. **Nine-row budget table** for a 3 s RAG SLO: stage, budget, timeout, fallback. ([10](10-rag-and-agentic-serving.md))
38. **Kill a dependency.** Stop the vector DB mid-load-test; record client-visible behavior before and after implementing the degraded rung. ([10](10-rag-and-agentic-serving.md))
39. **Prompt-ordering effect.** Compare TTFT and prefix-cache hit rate with a static-first prompt versus a request-id-prefixed prompt. ([10](10-rag-and-agentic-serving.md), [Phase 4 L6](../phase-4/06-prefix-caching-and-radix-attention.md))
40. **Agent cost model.** For a 5-step loop with growing context, compute total prefill tokens with and without prefix caching, and with a compaction step. Report the multiplier. ([10](10-rag-and-agentic-serving.md))

---

## Conceptual self-check (no notes)

1. Name the four numbers that classify a serving workload, and the two quantities derived from them. ([1](01-the-other-inference-workloads.md))
2. Why do continuous batching and PagedAttention not apply to a ranking or vision service? ([1](01-the-other-inference-workloads.md))
3. A 300-way parallel feature fetch, each with p99 = 5 ms — what is your request latency, and what are three fixes? ([2](02-recommendation-and-ranking.md))
4. Why does a 1 TB ranking model need less compute than a 7B LLM, and what does that change about hardware choice? ([2](02-recommendation-and-ranking.md))
5. What is training/serving feature skew, and why do metrics not catch it? ([2](02-recommendation-and-ranking.md))
6. Name the five stages of a vision request and the usual dominant one. Give the formula for cores needed to decode. ([3](03-vision-serving.md))
7. Name four ways "resize the image" can silently differ between training and serving, and the gate that catches all four. ([3](03-vision-serving.md))
8. Define real-time factor and convert it to streams per device. Why does a stream occupy a slot even during silence? ([4](04-speech-and-streaming.md))
9. Why can't you make a full-attention ASR model stream by feeding it chunks? Name three real options. ([4](04-speech-and-streaming.md))
10. Write the cost-per-unit-of-useful-work formula and name three ways hardware comparisons are commonly faked. ([5](05-hardware-diversity.md))
11. What do TPU and Inferentia demand that CUDA doesn't, and what is the standard mitigation? ([5](05-hardware-diversity.md))
12. When is CPU serving the *right* choice? Give four situations and the physical limit that bounds it. ([5](05-hardware-diversity.md))
13. What is recall@k, what does it depend on, and why is "sub-millisecond ANN search" meaningless alone? ([6](06-vector-search-and-ann.md))
14. Compute HNSW memory for 10M × 768 fp16 vectors at M = 32, plus rebuild headroom. What breaks without the headroom? ([6](06-vector-search-and-ann.md))
15. Why do deletes degrade an HNSW index, and what is the operational consequence? ([6](06-vector-search-and-ann.md))
16. Name the three on-device budgets. Why is a burst benchmark misleading? ([7](07-edge-and-on-device.md))
17. Why can partial NPU offload be slower than pure CPU, and how do you detect it? ([7](07-edge-and-on-device.md))
18. Why is on-device rollback not a thing, and what replaces it? ([7](07-edge-and-on-device.md))
19. Name the four ways a model conversion fails, and the gate for each. ([8](08-model-formats-and-runtimes.md))
20. What does a format *not* contain, and which incident class does that cause? ([8](08-model-formats-and-runtimes.md))
21. Which artifacts are portable and which are pinned, and what does that imply about your registry and rollback? ([8](08-model-formats-and-runtimes.md), [Phase 8 L7](../phase-8/07-model-registry-and-artifacts.md))
22. Why does prompt injection have no complete fix? Name five blast-radius controls. ([9](09-security-and-multi-tenancy.md))
23. Why is "100 requests/minute" a bad inference rate limit? Write the right formulation, including how you limit output tokens you can't predict. ([9](09-security-and-multi-tenancy.md))
24. What does MIG isolate that time-slicing does not? When is each correct? ([9](09-security-and-multi-tenancy.md))
25. Why is indirect injection more severe than direct, and what makes it exploitable? ([9](09-security-and-multi-tenancy.md))
26. Why is a pipeline's p99 not the sum of stage p50s, and what is the correct design bound? ([10](10-rag-and-agentic-serving.md))
27. Give the RAG degradation ladder from full pipeline to shed, with the rung most teams forget. ([10](10-rag-and-agentic-serving.md))
28. Which four artifacts must be versioned together in a RAG service, and what breaks when they aren't? ([10](10-rag-and-agentic-serving.md))
29. Why does an agent loop's cost grow superlinearly in step count, and what are the four controls? ([10](10-rag-and-agentic-serving.md))
30. Why is a semantic cache a correctness *and* a privacy risk? ([9](09-security-and-multi-tenancy.md), [10](10-rag-and-agentic-serving.md))

If any answer takes more than ~60 seconds, re-read the linked lesson. Questions 3, 10, 13, 22, 23, 24 and 26 are asked, nearly verbatim, in senior inference and platform interviews — and 22-24 are asked in every AI-product security review.

---

## Exit artifact

Produce **Option A**. B is the breadth artifact if your target role isn't LLM-specific; C is cheap and pays for itself in design reviews.

### Option A — Project 17: the RAG service (required)

`projects/17-rag-pipeline-guardrails/README.md` containing:

- **`BUDGET.md`**: nine stages with budget, timeout and fallback.
- **Per-stage latency table**: p50/p99 over ≥ 200 queries, with the stage owning your p99 named.
- **Retrieval evidence**: index manifest, recall@k against exact search, and end-task quality at 2-3 recall levels.
- **Chaos table**: all seven rows from [lesson 11](11-build-rag-and-cpu-serving.md) step 6, with observed client-visible behavior.
- **Limiter evidence**: clean per-dimension 429s, and proof that a flooding tenant does not move the other tenant's p99.
- **Guardrail bypass table**: ≥ 10 attacks, which bypassed, what you changed or accepted — plus the regression suite in the repo.
- **Cost breakdown per request** by stage, showing retrieval versus LLM share.

### Option B — Project 16: the hardware/runtime decision (strongly recommended)

`projects/16-cpu-vs-gpu-serving/README.md` with the three-runtime table (p50/p99/throughput/memory/accuracy/σ), the verification record for every artifact, prefill reported separately from decode, cost per 1M units at two utilization levels, the crossover-QPS plot, and the one-paragraph decision plus its falsifier.

### Option C — The breadth document set

1. **Workload-shape playbook**: the four-number classification, the five archetypes, and the transfer table — written in your own words, with a worked example for a workload you've never served.
2. **Hardware decision memo**: the cost-per-unit-of-useful-work procedure, the portability-tax table, and the defaults table, with your measured numbers substituted in.
3. **Security review template** for a model-backed feature: trust boundaries, tool authorization, output sinks, limiter dimensions, log policy, supply-chain checks. One page you could hand to another team.

Update [`projects/README.md`](../../projects/README.md) status for 16 and 17 when done, and run the [production-readiness checklist](../../playbooks/production-readiness-checklist.md) against project 17.

---

## How you know you're ready for the capstones

- [ ] You can classify an unfamiliar inference workload in four numbers and say what transfers from Phases 2-8.
- [ ] You can explain, with arithmetic, why a 300-way fan-out breaks a 10 ms budget.
- [ ] You have a per-stage latency attribution for at least one non-LLM pipeline, and you fixed its dominant stage.
- [ ] You have measured the same model on at least two kinds of silicon and converted both to cost per unit of useful work.
- [ ] You have tuned an ANN index to a recall target and know its memory cost at 10× the data.
- [ ] You have converted a model and verified it numerically *and* on a task metric, on the target runtime.
- [ ] You can state what MIG isolates, what time-slicing doesn't, and which your workload needs.
- [ ] You have attacked your own guardrail and can quote its bypass rate.
- [ ] Every stage of your pipeline has a budget, a timeout, and a fallback — and you have tested each fallback by breaking the dependency.
- [ ] Your rate limits are denominated in tokens and concurrency, and one tenant cannot move another's p99.

---

## Where these ideas come back

| Phase 9 idea | Comes back as |
|---|---|
| Workload-shape classification | choosing which capstone architecture fits the problem you pick ([Phase 10 lesson 1](../phase-10/01-choosing-a-capstone.md)) |
| Per-stage budgets, timeouts, fallbacks | the [production-grade RAG service](../phase-10/07-capstone-production-rag.md) and the [multi-modal pipeline](../phase-10/04-capstone-multimodal-pipeline.md) |
| Vision/speech pipeline stages | the STT → LLM → TTS or image → detection → LLM capstone, as Triton ensembles or Ray Serve graphs ([Phase 5 lessons 7-8](../phase-5/07-triton-inference-server.md)) |
| Cost per unit of useful work, crossover analysis | the [cost-optimization case study](../phase-10/05-capstone-cost-optimization.md), where it is the entire deliverable |
| Verified conversion as a build step | any capstone that compiles an engine or ships a quantized artifact, and its reproducibility claim |
| ANN recall/latency frontier | retrieval quality gating in the RAG capstone's CI |
| Limiter + guardrails + bypass suite | the security section every capstone write-up needs to be credible |
| Tenancy and isolation mechanics | the deployment section of any capstone that shares hardware |

---

**Next:** the capstones — **[Phase 10](../phase-10/README.md)**. Nine phases of mechanics are behind you: the model math (1), the hardware (2), the serving loop (3), the optimizations (4), the frameworks (5), the fleet (6), the operations (7), the delivery pipeline (8), and the breadth of the field (9). What remains is to build something nobody assigned you, measure it honestly, and write it up so that a stranger can tell exactly what you did and what it cost — which is what [Phase 10 lesson 2](../phase-10/02-engineering-standards.md) makes checkable.
