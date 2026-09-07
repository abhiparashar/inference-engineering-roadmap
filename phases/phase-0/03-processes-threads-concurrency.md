# 3 — Processes, Threads, and Concurrency vs Parallelism

> **You'll be able to say:** "Concurrency is *dealing with* many things at once; parallelism is *doing* many things at once. An inference server needs both — async I/O to juggle thousands of connections, and true parallel compute (on the GPU) to run the model."

This distinction is the backbone of Phase 3 (serving). Get it now and continuous batching will later feel inevitable instead of clever.

---

## Processes vs threads (the containers your code runs in)

- **Process:** an independent running program with its *own* private memory. Two processes can't accidentally stomp on each other's data — the OS keeps them separate. Starting one is relatively heavy. Example: your web server and your database are separate processes.
- **Thread:** a "line of execution" *inside* a process. All threads in one process **share the same memory**. Lighter to create than a process, and they can cooperate by reading/writing shared variables — which is also exactly how they get into trouble (two threads editing the same variable = bugs).

```
┌─ Process A ───────────────┐   ┌─ Process B ───────────────┐
│  private memory           │   │  private memory           │
│  ┌────────┐ ┌────────┐    │   │  ┌────────┐               │
│  │thread 1│ │thread 2│    │   │  │thread 1│               │
│  └────────┘ └────────┘    │   │  └────────┘               │
│  (share A's memory)       │   │                           │
└───────────────────────────┘   └───────────────────────────┘
     isolated from each other by the OS ─────────────────────
```

**Rule of thumb:** use *threads* when tasks need to share data cheaply; use *processes* when you want isolation or need to sidestep Python's GIL (next section).

---

## Concurrency vs parallelism (the idea people mix up most)

They sound like synonyms. They're not.

- **Concurrency = dealing with many tasks at once by switching between them.** One worker, many jobs, rapidly interleaved. Progress on all of them, but only one is actually *executing* at any instant. Great when tasks spend time *waiting* (for the network, disk, etc.) — you switch to another task during the wait.
- **Parallelism = actually doing many tasks at the exact same instant** on multiple workers (multiple CPU cores, or thousands of GPU cores). Great when tasks are *computing* and there's genuinely more work than one worker can do.

> **Analogy — one barista (concurrency):** a single barista takes your order, starts your espresso, and *while it's pulling* takes the next person's order and steams milk. One worker, many orders in flight, lots of switching during waits. Nobody is served literally simultaneously, but the line moves fast because no one stands idle during the waits.
>
> **Analogy — many baristas (parallelism):** four baristas each make one drink at the same time. Genuinely four drinks being made in the same instant.

You use concurrency to hide *waiting*. You use parallelism to get through *work*. A real inference server does **both**: concurrency to juggle thousands of open HTTP connections that are mostly waiting, and parallelism (on the GPU) to crunch the actual model math.

---

## Async I/O — concurrency without threads

The most common way to get concurrency in a server is **async I/O** (Python's `asyncio`; what FastAPI/uvicorn use under the hood). The idea: a single thread runs an **event loop**. When a task hits a wait ("send this over the network and wait for the reply"), instead of blocking, it says "wake me when the reply arrives" and the event loop immediately runs some *other* ready task. When the reply lands, the paused task resumes.

```
Event loop timeline (ONE thread):
 req1: read ▓░░░░ (waiting on network) ...................... resume ▓
 req2:        read ▓░░░ (waiting) ................ resume ▓
 req3:              read ▓░░░░░ (waiting) .................... resume ▓
        ▲ while req1 waits, the loop runs req2, then req3, etc.
```

One thread, thousands of connections, because at any moment almost all of them are *waiting* and only need attention for microseconds when their data is ready. This is why a tiny FastAPI server can hold 10,000 open connections — most cost nothing while idle.

The catch: **async only helps with waiting, never with computing.** If one task sits and does heavy math (like a model forward pass) inside the event loop, the whole loop freezes — every other connection stalls until that math finishes. That's why serving frameworks run the model on a GPU and/or in a separate worker, keeping the async loop free to keep juggling connections. Remember this; it directly motivates the architecture of every server you'll build in Phase 3.

---

## The GIL — why Python threads don't speed up math

Python (specifically CPython, the standard one) has a **Global Interpreter Lock (GIL)**: a rule that *only one thread can execute Python bytecode at a time*, even on a 16-core CPU. So spinning up 8 threads to do 8 chunks of pure-Python math gives you… roughly the speed of 1, because they take turns holding the GIL.

Two honest clarifications, because this trips everyone up:

1. **Threads still help for I/O-bound work.** When a thread is *waiting* on the network or disk, it releases the GIL, letting another thread run. So threads are fine for "many things waiting" — they're just useless for "many things computing" in pure Python.
2. **The GIL does not chain your GPU or NumPy.** Libraries like NumPy, PyTorch, and CUDA release the GIL while they run their compiled C/CUDA code. So `torch.matmul` on a big tensor *does* use all your cores / the whole GPU — the GIL is only held during the Python bits around it. This is (again) why "keep Python out of the inner loop" works.

**Consequences you'll actually use:**
- To get true CPU parallelism in Python, use **multiple processes** (`multiprocessing`, or just run multiple server workers), not threads.
- For I/O concurrency, use **async** (`asyncio`) — it's lighter than threads and avoids most shared-memory bugs.
- For model compute, lean on the library (PyTorch/CUDA) — it already parallelizes and drops the GIL.

> Note: newer CPython has an experimental "no-GIL" build, and the GIL's details evolve. For this roadmap, assume the classic GIL behavior above — it's what the tools and servers you'll use are designed around.

---

## Putting it together: the anatomy of an inference server

Here's the whole picture this lesson was building toward:

```
                 ┌──────────────── async event loop (1 thread) ─────────────┐
 many clients ──▶│ accept connections, read requests, stream responses      │
   (mostly       │  — concurrency: thousands of connections, all mostly     │
    waiting)     │    waiting, cheap to juggle                              │
                 └───────────────┬──────────────────────────────────────────┘
                                 │ hands the heavy math off
                                 ▼
                 ┌──────────── GPU / model worker ──────────────────────────┐
                 │ runs the forward pass — parallelism: thousands of GPU     │
                 │ cores doing matrix math at once. Must NOT block the loop. │
                 └───────────────────────────────────────────────────────────┘
```

- The **front** is I/O-bound → solved with **concurrency** (async).
- The **model** is compute-bound → solved with **parallelism** (GPU).
- The art of Phase 3 is feeding the expensive GPU efficiently (batching requests together) *without* making waiting clients suffer. That tension — throughput vs latency — is the entire serving discipline, and it grows directly out of this one distinction.

---

## Tiny experiments

**1. Watch async juggle waits.** Two "slow network calls" that each sleep 1 second, run concurrently, finish in ~1s total (not 2):

```python
import asyncio, time

async def fake_request(name):
    print(name, "start")
    await asyncio.sleep(1)      # pretend we're waiting on the network
    print(name, "done")

async def main():
    t = time.perf_counter()
    await asyncio.gather(fake_request("A"), fake_request("B"))
    print("total:", time.perf_counter() - t)   # ~1.0s, not 2.0s

asyncio.run(main())
```

The waits overlapped — that's concurrency hiding I/O. Now change `asyncio.sleep` to a *computing* loop (`sum(range(50_000_000))`) and watch the total become the *sum* of both — because compute can't overlap on one thread.

**2. Feel the GIL.** Run a CPU-heavy function in 4 threads, then in 4 processes, and compare wall-clock time. Threads ≈ no speedup (GIL); processes ≈ ~4x speedup (true parallelism). (`concurrent.futures.ThreadPoolExecutor` vs `ProcessPoolExecutor`.)

---

## Key takeaways

- **Process** = isolated program with private memory; **thread** = line of execution sharing its process's memory.
- **Concurrency** = interleaving many tasks (hides *waiting*); **parallelism** = truly simultaneous execution (gets through *work*).
- **Async I/O** gives cheap concurrency for I/O-bound servers, but heavy compute inside the event loop freezes everything.
- The **GIL** prevents true parallel *Python* execution, so use processes for CPU parallelism — but NumPy/PyTorch/CUDA release the GIL, so your model math still uses the hardware fully.
- An inference server = async concurrency at the front + GPU parallelism for the model. Every Phase 3 design decision lives in the gap between those two.

**Next:** [What "inference" actually is →](04-what-is-inference.md)
