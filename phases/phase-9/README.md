# Phase 9 — Beyond LLMs: Recsys, Vision, Speech, Hardware Diversity, Edge, Security, RAG (Deep Dive)

> **Goal:** stop being an LLM-serving engineer and become an *inference* engineer. By the end of this phase you can size a ranking service at a million QPS inside a 10 ms budget, find the bottleneck in a vision pipeline that isn't the model, write a streaming ASR latency contract, price TPU/Inferentia/CPU against an NVIDIA GPU for a specific workload, tune an ANN index to a recall target instead of a vibe, ship a model that fits in megabytes on a phone, convert and verify a model across runtimes without silent numerical drift, isolate tenants on shared GPUs and defend against prompt injection, and hold a RAG pipeline to per-stage latency budgets with a fallback for every stage.

This folder is the long-form version of [Phase 9 in the ROADMAP](../../ROADMAP.md#phase-9--beyond-llms-recsysvisionspeech-hardware-diversity-edge-and-security). Phases 1-8 went deep on exactly one workload shape: a large autoregressive model, tens of QPS, long streaming outputs, NVIDIA GPUs, one model per request. That shape is maybe a third of production inference by request count and a much smaller fraction by *service* count.

The intellectual core:

> **A serving system's architecture is determined by the shape of its workload, not by the fact that it contains a model. Four numbers — per-request FLOPs, latency budget, per-request state, and arrival rate — decide your batching strategy, your hardware, your failure modes, and which of Phases 2-8 even apply. Move those numbers by three orders of magnitude and every default flips: batching stops being a tradeoff and becomes free, the bottleneck moves from HBM bandwidth to network round trips, and the most expensive component stops being the model.**

---

## Prerequisites

- **[Phase 2 lesson 5](../phase-2/05-roofline-model.md)** — the roofline model. It is the tool that tells you a ranking model is neither compute- nor bandwidth-bound but *latency*-bound on a network hop, and that a vision batch job is the one workload in this repo that actually saturates tensor cores.
- **[Phase 3 lessons 2-3, 5](../phase-3/02-static-batching.md)** — static and dynamic batching, queueing theory. Everything Phase 3 called "the bad old way" is the *correct* answer for non-autoregressive models, and this phase explains why.
- **[Phase 4 lessons 2-3](../phase-4/02-quantization-fundamentals.md)** — quantization. On CPUs, NPUs, and phones, quantization is not an optimization; it is the entry ticket, and the constraints are far tighter than on a GPU.
- **[Phase 5 lesson 7](../phase-5/07-triton-inference-server.md)** — Triton Inference Server. Outside pure-LLM shops it is *the* serving layer; lessons 2-4 here are the workloads it was built for.
- **[Phase 7 lessons 5, 7, 9](../phase-7/05-slos-and-error-budgets.md)** — SLOs, degradation ladders, cost per unit of work. Lesson 10 applies all three to a multi-stage pipeline where each stage needs its own budget and its own fallback.
- **[Phase 8 lessons 3, 7](../phase-8/03-image-size-and-cold-start.md)** — image size, cold start, artifact identity. Lessons 7-8 here are the same discipline with the budget reduced by three orders of magnitude and the runtime no longer under your control.

### About hardware

- **This is the most laptop-friendly phase since Phase 1.** Lessons 2, 5, 6, 7 and 8 are *better* on a laptop: CPU inference, ANN indexes, on-device runtimes and format conversion are the native habitat of a MacBook or any x86 box. Apple Silicon users get Metal, Core ML and `llama.cpp` first-class.
- **A GPU makes three things real**: hardware video decode in lesson 3, the CPU-vs-GPU cost comparison in lesson 5, and MIG/time-slicing in lesson 9 (MIG needs A100/H100-class hardware; time-slicing works on anything).
- **Cloud accelerators are rentable by the hour**: a TPU v5e slice or an `inf2.xlarge` costs a few dollars for an afternoon, and the compile-then-run experience is the actual lesson — you cannot get it from reading.
- **No hardware at all?** Lessons 1, 6, 9 and 10 are complete on a laptop with no accelerator of any kind, and they are four of the six most interview-relevant lessons in the phase.

---

## The map of this phase

```
              WHAT SHAPE IS THE WORKLOAD?  (lesson 1)
                              │
     ┌────────────────┬───────┴────────┬────────────────┐
     ▼                ▼                ▼                ▼
  RANKING          VISION           SPEECH          RETRIEVAL
  lesson 2         lesson 3         lesson 4         lesson 6
  1M QPS, 10ms     throughput,      streaming,       recall vs
  TB embedding     decode-bound,    partial output,  latency, index
  tables, network  NVDEC/DALI       endpointing      params, memory
  round trips      preprocessing    real-time factor math
     │                │                │                │
     └────────────────┴───────┬────────┴────────────────┘
                              │  same model, different silicon
     ┌────────────────┬───────┴────────┬────────────────┐
     ▼                ▼                ▼                ▼
  HARDWARE        EDGE/DEVICE      FORMATS          TENANCY
  lesson 5        lesson 7         lesson 8         lesson 9
  TPU/XLA,        megabytes,       ONNX/GGUF/       MIG, time-slice,
  Inferentia,     thermal, NPU     ExecuTorch,      injection, rate
  CPU economics   int8-only        opsets, drift    limits, PII
     │                │                │                │
     └────────────────┴───────┬────────┴────────────────┘
                              ▼
                    lesson 10: it's a PIPELINE now
                    per-stage budgets, per-stage timeouts,
                    per-stage fallbacks, cache layers
                              ▼
                    lesson 11: build both projects
  ─────────────────────────────────────────────────────────────────────────
  THE ONE SLIDE: WHY LLM INTUITION MISLEADS ELSEWHERE
    LLM:      2 GFLOP/token × 500 tokens, 2 s budget, GBs of KV state
              ⇒ memory-bandwidth-bound, batching is a latency tradeoff
    RANKING:  ~1 MFLOP/candidate, 10 ms budget, TB of shared state
              ⇒ network-round-trip-bound, batching is nearly free
    VISION:   50 GFLOP/image, 1 s budget, no state
              ⇒ compute-bound — if you fixed the JPEG decode first
    SPEECH:   small model, budget measured against WALL CLOCK of audio
              ⇒ real-time factor, not throughput; output before input ends
```

Two conclusions most engineers reach only after being handed a non-LLM service:

1. **The model is often not the expensive part.** In ranking it's the feature fetch; in vision it's the decode; in RAG it can be the vector database ([Phase 7 lesson 9](../phase-7/09-cost-per-million-tokens.md)'s cost table already hinted at this). Optimizing the forward pass of a system whose forward pass is 15% of the budget is the most common expensive mistake in this part of the field.
2. **Portability is a cost, not a feature.** Every format conversion, execution provider and non-NVIDIA accelerator buys you price or reach and charges you in numerical drift, unsupported ops, static-shape recompiles, and a second debugging environment ([lesson 8](08-model-formats-and-runtimes.md)). Pay it deliberately.

---

## The lessons (read in order)

| # | File | What you'll be able to say afterwards |
|---|---|---|
| 1 | [The other inference workloads](01-the-other-inference-workloads.md) | "I classify a serving problem by four numbers — per-request FLOPs, latency budget, per-request state, arrival rate — and I can say which Phase 2-8 techniques transfer and which are actively wrong for it." |
| 2 | [Recommendation and ranking inference](02-recommendation-and-ranking.md) | "Ranking is a funnel with a 10 ms budget where the bottleneck is embedding lookup and feature fetch over the network. I know the fan-out tail-amplification math, the cache hierarchy, and why batching is free here." |
| 3 | [Vision and video serving](03-vision-serving.md) | "I profile the pipeline, not the model: decode, resize, normalize, transfer, infer. I know when to move decode onto NVDEC/DALI, and that a resize-semantics mismatch is an accuracy bug no latency test catches." |
| 4 | [Speech: streaming ASR and TTS](04-speech-and-streaming.md) | "Real-time factor and first-audio latency, not QPS. I can write the streaming contract — chunk size, lookahead, partial vs final, endpointing — and price a full voice-agent loop end to end." |
| 5 | [Hardware diversity: TPU, Inferentia, CPU](05-hardware-diversity.md) | "I choose silicon per workload on cost per unit of useful work, and I know what each choice costs me: static shapes and bucketing on XLA, a compiler and SDK on Neuron, throughput ceilings on CPU." |
| 6 | [Vector search and ANN](06-vector-search-and-ann.md) | "Recall@k is a tunable, not a property. I can set HNSW `M`/`ef` or IVF-PQ `nlist`/`nprobe` to hit a recall target at a latency target, compute the index memory, and measure recall against exact search." |
| 7 | [Edge and on-device inference](07-edge-and-on-device.md) | "The budget is megabytes, milliwatts and thermal headroom. I know the NPU quantization constraints, why the second inference is 10× faster than the first, and how model updates ship without an app release." |
| 8 | [Model formats and runtimes](08-model-formats-and-runtimes.md) | "A format is a contract about ops, opsets and numerics. Conversion is a build step with a numerical-tolerance test, and I debug unsupported-op and drift failures instead of hoping." |
| 9 | [Security and multi-tenancy](09-security-and-multi-tenancy.md) | "Prompt injection is a data/instruction confusion with no complete fix, so I constrain the blast radius. MIG partitions, time-slicing doesn't isolate, token buckets must be denominated in tokens, and logs are a PII surface." |
| 10 | [RAG and agentic serving](10-rag-and-agentic-serving.md) | "A pipeline's p99 is not the sum of its stage p50s. Every stage has a budget, a timeout, a fallback and a cache, and an agent loop multiplies all of it by the number of round trips." |
| 11 | [Build: CPU/GPU shootout + RAG with guardrails](11-build-rag-and-cpu-serving.md) | "I built both: a quantized CPU serving comparison with honest numbers, and a RAG service with per-stage budgets, a token-bucket limiter, and an injection guardrail I tried to break myself." |
| 12 | [Exercises & exit artifact](12-exercises-and-artifacts.md) | "Here are the workload-shape table, the recall/latency curve, the CPU-vs-GPU cost table, and the per-stage budget with its fallback ladder." |

---

## How to work through this phase

1. **Read lesson 1 properly, even if you skim the rest.** The four-number classification is the transferable skill; the remaining lessons are instances of it. It is also the lesson that shows up in interviews as "how would you serve X?" for an X that isn't an LLM.
2. **Pick depth by target job.** LLM-serving-specific target: read all of 1, 9, 10, skim 2-8 (≈1 week). General inference-engineering breadth: do every lesson and both projects (≈2-3 weeks). Recsys/ads/ranking target: lessons 2 and 6 become the core of the phase.
3. **Measure on the hardware you have, not the hardware you read about.** A CPU-vs-GPU table from your own laptop plus one rented GPU-hour is worth more than any vendor benchmark, and it is the artifact that makes lesson 5 stick.
4. **Break the guardrails yourself.** In lesson 9, the exercise is not "add a filter" — it is "write ten inputs that defeat your own filter." An injection guardrail you haven't attacked is a false sense of security you've shipped to production.
5. **Instrument per stage before optimizing anything.** In lessons 3, 6 and 10 the first action is always the same: split the measured latency into stages and look at which one is 60% of it. Do this before reading anyone's tuning guide.
6. **Keep the LLM comparison alive.** Every number you collect here — CPU tokens/sec, ANN p99, on-device latency — should be put next to your Phase 3/4 GPU numbers. The comparison *is* the learning.

**Time budget:** 1-3 weeks part-time depending on the choice in step 2. Lessons 1, 5, 9 and 10 are load-bearing for any inference role; lesson 11 produces the artifacts.

## Phase self-check (from the ROADMAP)

You're done when you can, without notes:

1. Explain **why a ranking model at 1M QPS / 10 ms and an LLM at 10 QPS / 2 s are different engineering problems**, and name which Phase 2-4 techniques transfer to each. ([lessons 1](01-the-other-inference-workloads.md), [2](02-recommendation-and-ranking.md))
2. Name the **five stages of a vision request** and which one is usually dominant before anyone optimizes. ([lesson 3](03-vision-serving.md))
3. State the **streaming ASR latency contract** — chunk, lookahead, partial, final, endpoint — and what real-time factor means. ([lesson 4](04-speech-and-streaming.md))
4. Give the **decision rule for CPU vs GPU vs TPU vs Inferentia** for a named workload, in cost per unit of useful work. ([lesson 5](05-hardware-diversity.md))
5. Tune an **ANN index to a recall target** and explain the memory and latency consequence of each knob. ([lesson 6](06-vector-search-and-ann.md))
6. Explain **why an NPU often needs static int8 per-tensor quantization** and what that does to accuracy. ([lessons 7](07-edge-and-on-device.md), [8](08-model-formats-and-runtimes.md))
7. Describe **what MIG isolates that time-slicing does not**, and when each is the right answer. ([lesson 9](09-security-and-multi-tenancy.md))
8. Explain **why prompt injection has no complete fix**, and name three blast-radius controls that work anyway. ([lesson 9](09-security-and-multi-tenancy.md))
9. Take a **RAG request's latency budget apart by stage** and place a timeout and a fallback on each. ([lesson 10](10-rag-and-agentic-serving.md))

## Projects that belong to this phase

- **[16 — CPU-only serving shootout: ONNX Runtime quantized vs `llama.cpp` GGUF vs GPU FP16](../../projects/README.md)** (small): one model, three runtimes, honest latency/throughput/cost-per-unit-work numbers, plus the accuracy check that makes the comparison legitimate. Built in [lesson 11](11-build-rag-and-cpu-serving.md), part 1.
- **[17 — Mini RAG pipeline with per-stage latency budgets, rate limiting, and prompt-injection guardrail](../../projects/README.md)** (large): embed → ANN search → LLM, with per-stage instrumentation, a budget and fallback per stage, a token-bucket limiter, and a guardrail you attacked yourself. Built in [lesson 11](11-build-rag-and-cpu-serving.md), part 2. **This is the Phase 9 exit artifact**, and the direct precursor to capstone 21.

Also do the [Phase 9 labs](../../labs/README.md#phase-9-lab--beyond-llms) — the GGUF CPU run, the ONNX Runtime execution-provider read, and the OWASP exercise pair directly with lessons 5, 8 and 9.

---

Next after this: **[Phase 10 — Capstones](../../ROADMAP.md#phase-10--capstones-this-is-where-top-1-gets-proven)**. Nine phases of mechanics; the capstones are where you prove them on something nobody assigned you. Two of the five capstones are direct continuations of this phase: the multi-modal pipeline (lessons 3-4 plus Phase 5's composition) and the production-grade RAG service (lessons 6, 9, 10 plus Phase 7's observability and Phase 8's gates).
