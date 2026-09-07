# Phase 0 — Systems & ML Foundations (Deep Dive)

> **Goal:** stop being scared of "C-level" concepts and tensors. By the end you can explain, in plain words, *how your computer actually runs your code* and *what "inference" even is* — the vocabulary every later phase silently assumes.

This folder is the long-form version of [Phase 0 in the ROADMAP](../../ROADMAP.md#phase-0--systems--ml-foundations). The roadmap gives you the bullet points; these files teach them from scratch, in simple language, with pictures and tiny experiments you can run on any laptop (no GPU needed).

---

## Who this is for

You can already write Python and you know what a `for` loop is. That's the bar. Everything else — processes, memory, floating point, what a "forward pass" is — is explained here from the ground up. If a sentence uses a scary word, that word is defined the first time it appears (and everything lives in the [GLOSSARY](../../GLOSSARY.md) too).

## Why Phase 0 exists

Inference engineering is **not** mostly about machine learning. It's about *systems*: how bytes move, how work gets scheduled, where time actually goes. If you skip these foundations, later phases (KV-cache, batching, quantization, GPUs) will feel like memorized magic spells. With these foundations, they'll feel obvious.

The single mental shift Phase 0 gives you: **stop thinking "the code runs" and start thinking "who does the work, where does the data live, and what is everyone waiting for?"**

---

## The lessons (read in order)

| # | File | What you'll be able to say afterwards |
|---|---|---|
| 1 | [How a computer runs your code](01-how-a-computer-runs-your-code.md) | "Python is instructions the CPU executes; here's why it's slow and C/CUDA are fast." |
| 2 | [The memory hierarchy](02-memory-hierarchy.md) | "Data lives in registers → cache → RAM → disk, each ~10-100x slower; most performance work is about *where data lives*." |
| 3 | [Processes, threads, concurrency vs parallelism](03-processes-threads-concurrency.md) | "async I/O ≠ parallel compute; the GIL is why Python threads don't speed up math; here's what an inference server actually does." |
| 4 | [What "inference" actually is](04-what-is-inference.md) | "Inference = one forward pass through fixed weights. No gradients. That's why it's a totally different performance problem than training." |
| 5 | [Floating point & precision](05-floating-point-and-precision.md) | "FP32/FP16/BF16/INT8/INT4 — what the bits mean and why using fewer of them is the highest-leverage inference optimization." |
| 6 | [Build: a web server from scratch, then with a framework](06-build-a-server-from-scratch.md) | "I wrote a raw-socket HTTP server, then a FastAPI one, and I know exactly what the framework does for me." |
| 7 | [Exercises & exit artifact](07-exercises-and-artifacts.md) | The concrete thing you commit to prove Phase 0 is done. |

---

## How to work through this phase

1. **Read a lesson (~20-30 min).** Don't highlight; instead, after each section, close your eyes and re-explain it out loud. If you can't, re-read that section.
2. **Run the tiny experiments.** Every lesson has 2-4 short Python snippets. Type them (don't copy-paste) and run them. Watching numbers change is worth ten re-reads.
3. **Do the build (lesson 6).** This is where the abstract stuff becomes real.
4. **Produce the exit artifact (lesson 7).** No artifact = phase not finished — same rule as the rest of the repo.

**Time budget:** a comfortable Python programmer can do this phase in 1-2 weekends. If you're rusty, take a week. Don't rush *concepts 2, 3, and 5* — they are load-bearing for literally every later phase.

## Self-check for the whole phase

You're done with Phase 0 when you can answer all of these without notes:

1. Why is Python slower than C for a tight numeric loop? (lesson 1)
2. Roughly how much slower is reading from RAM than from CPU cache, and why do we care? (lesson 2)
3. What's the difference between concurrency and parallelism, and which one does the GIL block? (lesson 3)
4. What does "inference is a forward pass" mean, and why does that make it different from training? (lesson 4)
5. Why does representing a weight in 8 bits instead of 16 bits often make decoding *faster*, not just smaller? (lesson 5)
6. What did FastAPI/uvicorn do for you that your raw-socket server had to do by hand? (lesson 6)

If you can't answer #5 crisply, that's the most important one to revisit — it's the seed of the entire optimization story in Phase 4.

---

Next stop after this: **[Phase 1 — Transformer Internals & Inference Math](../phase-1/README.md)**, where you'll learn exactly what happens, tensor by tensor, to generate a single token.
