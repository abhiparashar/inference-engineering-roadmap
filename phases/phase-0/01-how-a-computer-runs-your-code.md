# 1 — How a Computer Actually Runs Your Code

> **You'll be able to say:** "My Python is a list of instructions something has to execute. Here's *what* executes them, *why* Python is slow, and *why* C and CUDA are fast."

---

## The one-paragraph version

A CPU is a machine that does one incredibly simple thing, billions of times per second: read a tiny instruction ("add these two numbers", "copy this byte there"), do it, move to the next. Your Python program is *not* those instructions — it's a high-level description that another program (the Python interpreter) reads and turns into work on the fly. That "on the fly" translation is why Python is flexible but slow. C code is translated into raw CPU instructions *once, ahead of time*, so at run time there's no translator in the loop. That's the whole story. The rest of this file is just filling it in.

---

## Layer by layer: from your text file to electrons

Think of it as a stack. Each layer only talks to the one below it.

```
Your Python source (.py)           ← what you write
        │  (interpreter reads it)
Python bytecode (.pyc)             ← simplified instructions for the "Python VM"
        │  (CPython evaluates each bytecode op in a big C loop)
Machine code (the CPython binary)  ← real CPU instructions, written in C, compiled once
        │
CPU instructions (x86 / ARM)       ← "add", "load", "store", "jump" — the real thing
        │
Transistors / electrons            ← physics; not our problem
```

Key insight: **your Python never becomes CPU instructions directly.** It becomes *bytecode*, and then a big loop written in C (inside CPython) looks at each bytecode op and runs the corresponding real machine code. So every `+` in your Python loop pays for: fetch bytecode → figure out what it means → check the types of the operands → find the right C function → call it. A C program's `+` is *one* CPU instruction. That overhead ratio — dozens of steps vs one — is why pure-Python numeric loops are ~10-100x slower than C.

### See it yourself

```python
import dis
def add(a, b):
    return a + b
dis.dis(add)
```

You'll see something like:

```
  LOAD_FAST   a
  LOAD_FAST   b
  BINARY_ADD
  RETURN_VALUE
```

Those are bytecode ops. `BINARY_ADD` isn't "add two ints" — it's "figure out what a and b are, find their `__add__`, and call it." That figuring-out is the tax.

---

## What "compiled" vs "interpreted" really means

- **Compiled (C, C++, Rust, CUDA):** a compiler reads your whole program *once*, before it ever runs, and produces a file full of raw CPU instructions. Running it = the CPU executing those instructions directly. No translator present at run time. Fast, but you must compile for a specific CPU/OS, and mistakes are less forgiving.
- **Interpreted (Python, by default):** a program (the interpreter) reads and executes your code step by step *every time it runs*. Flexible, portable, forgiving — but there's always a translator standing between your code and the CPU.

> There's a middle ground called **JIT** (Just-In-Time compilation): watch the code run, notice the hot parts, and compile *those* to machine code on the fly. That's what PyTorch's `torch.compile`, JAX, and Java's JVM do. You'll meet it again in Phase 4.

### Why we still use Python for AI, if it's slow

Because **the slow part isn't in Python.** When you write `torch.matmul(a, b)`, Python spends microseconds setting up the call, then hands a giant matrix multiply to compiled C/CUDA code that runs for milliseconds. Python is the *manager*; C/CUDA are the *workers*. As long as each Python instruction kicks off a big chunk of compiled work, Python's slowness doesn't matter. It only bites you when you do lots of tiny operations in a Python loop (element by element) — then you're paying the interpreter tax on every element. **Rule of thumb: keep Python out of the inner loop; hand big arrays to compiled code.**

---

## Syscalls: how your program asks the OS for anything real

Your program can't touch the network card, the disk, or another process's memory directly — the operating system (OS) guards all of that. When you want something real (read a file, send bytes over a socket, allocate memory), you make a **system call (syscall)**: a formal request to the OS kernel.

Why you care as an inference engineer: **syscalls are expensive** (they cross from your program into the kernel and back), and many involve *waiting* (the network hasn't replied yet; the disk is still spinning). That waiting is the difference between "I/O-bound" and "compute-bound" work — the single most important classification in this whole roadmap, covered in [lesson 3](03-processes-threads-concurrency.md).

A concrete picture for a server:

```
Client ──HTTP request──▶  your server calls recv()  ← syscall, may WAIT for bytes
                          your server runs the model ← pure compute, no waiting
       ◀──HTTP response── your server calls send()  ← syscall, may WAIT to flush
```

Waiting on `recv`/`send` is I/O. Running the model is compute. A good inference server overlaps them so the CPU/GPU is never idle while the network dawdles.

---

## A mental model you'll reuse forever

For any slow thing, ask: **"Is it waiting, or is it working?"**

- **Working** (compute-bound): the CPU/GPU is busy doing math. To go faster: do less math, or use faster hardware, or use fewer bits per number ([lesson 5](05-floating-point-and-precision.md)).
- **Waiting** (I/O-bound or memory-bound): the processor is idle, blocked on the network, disk, or memory. To go faster: overlap the waiting with other work, or move the data closer ([lesson 2](02-memory-hierarchy.md)).

Almost every optimization in this entire roadmap is one of those two moves. You just learned the whole game's grammar.

---

## Tiny experiments

**1. Feel the interpreter tax.** Sum a million numbers two ways:

```python
import time, numpy as np

n = 10_000_000
xs = list(range(n))

t = time.perf_counter()
s = 0
for x in xs:          # pure Python: interpreter runs every iteration
    s += x
print("python loop:", time.perf_counter() - t, "s")

arr = np.arange(n)
t = time.perf_counter()
s = int(arr.sum())    # one call into compiled C, loops internally
print("numpy sum:   ", time.perf_counter() - t, "s")
```

You'll typically see NumPy 20-100x faster. Same math — the difference is *who runs the loop*: the Python interpreter vs compiled C.

**2. Look at the bytecode** of a function you wrote today with `dis.dis`. Notice there's no such thing as "just add" — every op is bookkeeping around the real work.

---

## Key takeaways

- The CPU only runs simple machine instructions. Everything above that is layers of translation.
- Python is slow because a translator (the interpreter) stands between your code and the CPU at run time; C/CUDA translate once, ahead of time.
- AI in Python is fast anyway because Python only *dispatches* big compiled kernels — keep Python out of the inner loop.
- Syscalls are how you talk to the outside world, and they often involve *waiting*.
- The master question for all performance: **"Is it waiting, or is it working?"**

**Next:** [The memory hierarchy →](02-memory-hierarchy.md) — because "working" is often really "waiting for data to arrive."
