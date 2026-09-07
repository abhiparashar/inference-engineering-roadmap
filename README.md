# Inference Engineering Roadmap

A practical, project-driven path from newbie → **top 1% inference engineer**: someone who can build, profile, optimize, scale, and operate model-serving systems — not just call an API.

**Inference engineering** = taking a trained model and serving it to real users fast, cheap, and reliably. Training gets headlines; inference is where >90% of production AI compute cost lives (ChatGPT, Google AI Overviews, Instagram ranking, Netflix recommendations).

---

## Start here

1. **[`GETTING-STARTED.md`](GETTING-STARTED.md)** ← read this first. Prerequisites, and (critically) **how to get GPU access** — including what you can and cannot do on an Apple Silicon Mac.
2. **[`ROADMAP.md`](ROADMAP.md)** — the full 11-phase track. This is the main document.
3. **[`projects/README.md`](projects/README.md)** — 21 projects (small + large + capstones) mapped to phases. Nothing pre-built; you build them.
4. **[`labs/README.md`](labs/README.md)** — bite-sized "read this exact file in this real repo and answer this" exercises per phase.
5. **[`playbooks/`](playbooks/)** — reusable methodology: [benchmarking](playbooks/benchmarking.md), [profiling](playbooks/profiling.md), [production readiness](playbooks/production-readiness-checklist.md).
6. **[`GLOSSARY.md`](GLOSSARY.md)** — every acronym in this repo (TTFT, TPOT, HBM, MIG, TP/PP, roofline...) in plain English.
7. **[`resources/README.md`](resources/README.md)** — consolidated papers, books, engineering blogs, OSS repos, plus how to stay current and make your first OSS contribution.

## The 11 phases

| Phase | Topic | Core question it answers |
|---|---|---|
| 0 | Systems & ML Foundations | How does a computer actually run my model? |
| 1 | Transformer Internals & Inference Math | What happens, tensor by tensor, to generate one token? |
| 2 | GPU Architecture & Low-Level Perf | Why is my model slow — compute, memory, or overhead? |
| 3 | Serving Fundamentals (batching, queueing, scheduling) | How do I serve many users at once without wrecking tail latency? |
| 4 | Optimization (quantization, PagedAttention, speculative decoding) | How do I make it 10-100x cheaper? |
| 5 | Production Frameworks (vLLM, TensorRT-LLM, Triton, TGI, SGLang) | How do the real engines work internally? |
| 6 | Distributed Inference (TP/PP, disaggregation, load balancing) | How do I serve a model too big for one GPU, to millions of users? |
| 7 | Observability, Reliability, Cost (SRE for inference) | How do I know it's healthy, and what does it cost per token? |
| 8 | MLOps Glue (Docker, K8s, CI/CD, IaC) | How do I ship and roll it back safely? |
| 9 | Beyond LLMs (recsys/vision/speech, TPU/Inferentia/CPU, edge, security, RAG) | What about the 50% of inference that isn't an LLM? |
| 10 | Capstones | Can I prove all of the above with real artifacts? |

Full diagrams: [`assets/roadmap-diagram.md`](assets/roadmap-diagram.md).

## How this repo is meant to be used

- **Ratio discipline**: 1 hour reading → 3 hours building/profiling/reading source code. Top 1% comes from hours inside a profiler and inside other people's codebases, not papers read.
- **Every phase ends with a committed artifact** (a benchmark table, a profiler screenshot + writeup, a working service). If a phase produced no artifact, it isn't finished. See the exit-artifact rule in [`ROADMAP.md`](ROADMAP.md).
- **Be honest in your numbers.** Follow [`playbooks/benchmarking.md`](playbooks/benchmarking.md) — percentiles at fixed load, never averages of "as fast as possible" loops.

## Progress

Track status in [`projects/README.md`](projects/README.md) and commit as you go. Suggested pace for a part-time newbie (10-15 hr/week): **~7-8 months** end to end. Phases 2-4 are where real differentiation happens — don't rush them.
