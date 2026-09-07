# 7 — Exercises & Exit Artifact

> **Rule of this repo:** no artifact = phase not finished. This file is how you *prove* Phase 0 is done — to yourself, and to a future interviewer.

Do the exercises to cement understanding; produce the exit artifact to close the phase.

---

## Warm-up exercises (short, run them)

Each maps to one lesson. Type the code, run it, write down the number you saw.

1. **Interpreter tax** ([lesson 1](01-how-a-computer-runs-your-code.md)): sum 10M numbers with a Python `for` loop vs `numpy.sum`. Record both times and the ratio. Explain the ratio in one sentence.
2. **Cache locality** ([lesson 2](02-memory-hierarchy.md)): sum an 8000×8000 float32 array by row (`axis=1`) vs by column (`axis=0`). Record both times. Explain which one respects the cache and why.
3. **Concurrency vs compute** ([lesson 3](03-processes-threads-concurrency.md)): run two `asyncio.sleep(1)` tasks concurrently (total ≈ 1s), then replace the sleeps with a busy compute loop and observe the total become the *sum*. Explain in one sentence why compute didn't overlap.
4. **GIL** ([lesson 3](03-processes-threads-concurrency.md)): run a CPU-heavy function across 4 **threads** vs 4 **processes** (`concurrent.futures`). Record both wall-clock times. Explain why threads gave ~no speedup.
5. **A forward pass** ([lesson 4](04-what-is-inference.md)): implement the tiny 2-layer NumPy `forward()` from the lesson and run it on one input. State where the weights are and confirm there are no gradients anywhere.
6. **Quantization by hand** ([lesson 5](05-floating-point-and-precision.md)): quantize a small float array to INT8 and back (scale + round + dequantize). Record the max error. State the memory saving vs FP32.

Keep these in a scratch notebook or `labs/phase0/` — they don't all need to ship, but the *numbers* feed your writeup.

---

## Conceptual self-check (answer without notes)

If any answer is fuzzy, re-read the linked lesson before moving to Phase 1.

1. Why is a pure-Python numeric loop ~10-100x slower than the same loop in C? ([1](01-how-a-computer-runs-your-code.md))
2. Why does AI in Python stay fast *despite* Python being slow? What's the rule that keeps it fast? ([1](01-how-a-computer-runs-your-code.md))
3. Roughly how much slower is RAM than L1 cache, and what are the two "localities" caches exploit? ([2](02-memory-hierarchy.md))
4. Define **arithmetic intensity**. How do you use it to decide whether you're compute-bound or memory-bound? ([2](02-memory-hierarchy.md))
5. Concurrency vs parallelism — one sentence each. Which does the GIL block? Which does async provide? ([3](03-processes-threads-concurrency.md))
6. Why would putting a heavy compute loop inside a FastAPI `async` endpoint hurt *every other* connection? ([3](03-processes-threads-concurrency.md))
7. What exactly is "inference" in one sentence, and name two things training does that inference doesn't. ([4](04-what-is-inference.md))
8. Why is an LLM answer a *sequence* of forward passes rather than one? ([4](04-what-is-inference.md))
9. FP16 vs BF16 — both 16 bits; what's the trade and why does ML usually prefer BF16? ([5](05-floating-point-and-precision.md))
10. **The keystone question:** why does quantizing weights to INT8 often make *decoding* ~2x faster, not merely smaller? (Chain the answer through: decode is memory-bandwidth-bound → time ≈ bytes ÷ bandwidth → half the bytes → half the time.) ([5](05-floating-point-and-precision.md) + [2](02-memory-hierarchy.md))

Question 10 is the one that matters most. If you can deliver that chain crisply, you have the core intuition that the entire optimization half of this roadmap is built on.

---

## Exit artifact (this is what "finishing Phase 0" means)

Produce **one** of the following and commit it to the repo. Either is fine; pick the one you'll actually do well.

### Option A — "What runs my code?" writeup + microbenchmarks (recommended)

A single markdown file, `projects/phase0-foundations/README.md`, containing:

- **A results table** from exercises 1-4 (Python-vs-NumPy, row-vs-column, async-vs-compute, threads-vs-processes) with *your machine's* numbers and a one-line explanation of each.
- **One diagram** (ASCII or mermaid) of the memory hierarchy *or* the "async front-end + compute back-end" server anatomy, drawn from memory.
- **A 150-word explanation** answering self-check question 10 end to end (quantization → speed), in your own words.
- The scripts you used to get the numbers, committed alongside.

This is genuinely portfolio-worthy: it shows you can measure, not just recite.

### Option B — The two servers + comparison ([lesson 6](06-build-a-server-from-scratch.md))

Commit `raw_server.py`, `app.py`, and `NOTES.md` where:

- Both servers respond to `GET /health`.
- `NOTES.md` has your measured 5-concurrent-requests numbers (raw blocking ≈ 5s vs FastAPI async ≈ 1s) and a bullet list of *everything the framework did for you*.
- Bonus: include the "busy loop inside async endpoint" experiment and explain the collapse.

---

## How you know you're ready for Phase 1

You're ready when:

- [ ] All 10 self-check answers are crisp (especially #10).
- [ ] You ran the microbenchmarks and *saw* the numbers yourself — you're not taking the ratios on faith.
- [ ] Your exit artifact is committed.
- [ ] You can draw the memory hierarchy and the server anatomy from memory.

Then go to **[Phase 1 — Transformer Internals & Inference Math](../phase-1/README.md)**. Phase 0 gave you the systems vocabulary (waiting vs working, memory hierarchy, concurrency vs parallelism, forward pass, precision). Phase 1 uses *all* of it to explain exactly what happens, tensor by tensor, when a model writes a single token — and why the KV-cache exists.

---

## Where these ideas come back (so you know it wasn't busywork)

| Phase 0 idea | Comes back as |
|---|---|
| Waiting vs working; syscalls | Async serving, TTFT vs TPOT (Phase 3) |
| Memory hierarchy; bandwidth | HBM/SRAM, roofline, FlashAttention (Phase 2) |
| Arithmetic intensity | The roofline "knee"; why decode is memory-bound (Phases 1-2) |
| Concurrency vs parallelism | Continuous batching, the scheduler (Phase 3) |
| Forward pass; frozen weights | The whole KV-cache + quantization story (Phases 1, 4) |
| Precision / fewer bits | GPTQ, AWQ, KV-cache quantization (Phase 4) |

Nothing here is throwaway. Phase 0 is the grammar; the rest of the roadmap is the language.
