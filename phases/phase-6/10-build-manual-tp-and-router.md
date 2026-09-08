# 10 — Build: Manual Tensor Parallelism + a Prefix-Aware Router

> **You'll be able to say:** "I sharded a linear layer, a GQA attention block and a full transformer block across ranks with `torch.distributed`, proved the output matches the unsharded reference within tolerance, and counted exactly two all-reduces per layer. Then I put multiple replicas behind a router that does consistent-hash → two-candidate → least-loaded with bounded loads, drove it with a session-shaped workload, and produced the four-metric table — prefix-cache hit rate, p50/p99 TTFT, load imbalance, goodput — for round-robin vs sticky vs an oracle, including a mid-test scale event."

Two deliverables, both of which run **entirely on a laptop** for the correctness and policy work. **Part A** is [project 11](../../projects/README.md) (small). **Part B** is [project 12](../../projects/README.md) (large) and is the **Phase 6 exit artifact**. A GPU is optional in both: it changes the numbers, not the code.

Keep everything in `labs/phase6/`.

---

# Part A — tensor parallelism by hand

Goal: remove all remaining mystery from [lesson 3](03-tensor-parallelism.md). Success is a printed max-abs-diff at fp32 round-off scale and a collective count of exactly `2 × num_layers`.

## A1. Harness and instrumented collectives

```python
#!/usr/bin/env python3
# labs/phase6/tp_shard.py — run with: torchrun --nproc-per-node 4 tp_shard.py
import datetime, os, torch, torch.distributed as dist
import torch.nn.functional as F

STATS = {"collectives": 0, "bytes": 0}

def all_reduce(t):
    """Every TP collective in this file goes through here, so we can count them."""
    STATS["collectives"] += 1
    STATS["bytes"] += t.numel() * t.element_size()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t

def setup():
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    # a SHORT timeout is deliberate: lesson 9's hang becomes an error, not a mystery
    dist.init_process_group(backend, timeout=datetime.timedelta(seconds=30))
    rank, tp = dist.get_rank(), dist.get_world_size()
    if backend == "nccl":
        torch.cuda.set_device(rank % torch.cuda.device_count())
    return rank, tp, ("cuda" if backend == "nccl" else "cpu")
```

Two deliberate choices worth copying into real code: **every collective goes through one wrapper** (so counting and tracing are free), and the process group has a **short timeout** so a divergence shows up as an exception with a rank number instead of a hang ([lesson 9](09-multi-node-operations.md)).

## A2. The two sharded linears

```python
class ColumnParallelLinear:
    """W: [K, M] full weight (given to every rank ONLY so the test can compare).
    Output is SHARDED on M. No communication."""
    def __init__(self, W, rank, tp):
        M = W.shape[1]
        assert M % tp == 0, f"M={M} not divisible by tp={tp}"   # lesson 3's constraint
        s = M // tp
        self.w = W[:, rank * s:(rank + 1) * s].contiguous()

    def __call__(self, x):            # x: [T, K] replicated -> [T, M/tp]
        return x @ self.w

class RowParallelLinear:
    """W: [K, M]. Input is SHARDED on K, output is FULL after one all-reduce."""
    def __init__(self, W, rank, tp):
        K = W.shape[0]
        assert K % tp == 0
        s = K // tp
        self.w = W[rank * s:(rank + 1) * s, :].contiguous()

    def __call__(self, x_shard):      # x: [T, K/tp] -> [T, M]
        return all_reduce(x_shard @ self.w)
```

That's it. `ColumnParallelLinear` is a slice; `RowParallelLinear` is a slice plus a sum. Everything else in TP is arranging matmuls so those two alternate.

## A3. The MLP pair — one all-reduce for two matmuls

```python
class TPSwiGLU:
    def __init__(self, Wg, Wu, Wd, rank, tp):
        self.gate = ColumnParallelLinear(Wg, rank, tp)   # [H, I] -> [T, I/tp]
        self.up   = ColumnParallelLinear(Wu, rank, tp)
        self.down = RowParallelLinear(Wd, rank, tp)      # [I, H], input sharded on I

    def __call__(self, x):                               # [T, H] -> [T, H]
        return self.down(F.silu(self.gate(x)) * self.up(x))
```

Note what does *not* happen: the SwiGLU element-wise product runs entirely on the sharded `I/tp` dimension, so no collective is needed between the pair. **Three matmuls, one collective.**

## A4. GQA attention, including the KV-replication fallback

```python
class TPAttention:
    def __init__(self, Wq, Wk, Wv, Wo, n_q, n_kv, rank, tp):
        H, hd = Wq.shape[0], Wq.shape[1] // n_q
        assert n_q % tp == 0, "num_attention_heads must be divisible by tp"
        self.hq, self.hd, self.tp = n_q // tp, hd, tp
        q0 = rank * self.hq
        self.wq = Wq[:, q0 * hd:(q0 + self.hq) * hd].contiguous()      # column-parallel

        if n_kv >= tp:                        # clean case: shard the KV heads
            assert n_kv % tp == 0, "num_key_value_heads must be divisible by tp"
            self.hkv = n_kv // tp
            k0 = rank * self.hkv
        else:                                 # tp > n_kv: REPLICATE kv heads (lesson 3)
            self.hkv, k0 = 1, rank // (tp // n_kv)
        self.wk = Wk[:, k0 * hd:(k0 + self.hkv) * hd].contiguous()
        self.wv = Wv[:, k0 * hd:(k0 + self.hkv) * hd].contiguous()
        self.o  = RowParallelLinear(Wo, rank, tp)                      # row-parallel

    def __call__(self, x):                    # [T, H] replicated -> [T, H]
        T = x.shape[0]
        q = (x @ self.wq).view(T, self.hq,  self.hd).transpose(0, 1)   # [hq, T, hd]
        k = (x @ self.wk).view(T, self.hkv, self.hd).transpose(0, 1)
        v = (x @ self.wv).view(T, self.hkv, self.hd).transpose(0, 1)
        rep = self.hq // self.hkv                                      # GQA grouping
        k, v = k.repeat_interleave(rep, 0), v.repeat_interleave(rep, 0)
        # NO COMMUNICATION HERE: each rank owns whole heads, so softmax is local
        att = torch.softmax(
            (q @ k.transpose(-1, -2)) / self.hd ** 0.5
            + torch.full((T, T), float("-inf"), device=x.device).triu(1), dim=-1)
        out = (att @ v).transpose(0, 1).reshape(T, self.hq * self.hd)
        return self.o(out)                    # collective #2 of the layer
```

The comment on the softmax line is the point of the whole lesson: **attention needs no collective under TP** because heads are independent. Only the output projection reduces.

## A5. Verify against the unsharded reference

```python
def reference_block(x, Wq, Wk, Wv, Wo, Wg, Wu, Wd, n_q, n_kv):
    """Single-process ground truth: same math, no sharding, no collectives."""
    T, H = x.shape; hd = Wq.shape[1] // n_q
    q = (x @ Wq).view(T, n_q,  hd).transpose(0, 1)
    k = (x @ Wk).view(T, n_kv, hd).transpose(0, 1)
    v = (x @ Wv).view(T, n_kv, hd).transpose(0, 1)
    k, v = k.repeat_interleave(n_q // n_kv, 0), v.repeat_interleave(n_q // n_kv, 0)
    att = torch.softmax((q @ k.transpose(-1, -2)) / hd ** 0.5
                        + torch.full((T, T), float("-inf")).triu(1), dim=-1)
    h = x + ((att @ v).transpose(0, 1).reshape(T, n_q * hd) @ Wo)
    return h + (F.silu(h @ Wg) * (h @ Wu)) @ Wd

if __name__ == "__main__":
    rank, tp, dev = setup()
    torch.manual_seed(0)                      # identical weights on every rank
    H, I, n_q, n_kv, T, L = 256, 704, 8, 2, 32, 4
    W = {n: torch.randn(*s, dtype=torch.float32) / 16 for n, s in
         dict(q=(H, H), k=(H, n_kv * 32), v=(H, n_kv * 32), o=(H, H),
              g=(H, I), u=(H, I), d=(I, H)).items()}
    x = torch.randn(T, H) / 4

    attn = TPAttention(W["q"], W["k"], W["v"], W["o"], n_q, n_kv, rank, tp)
    mlp  = TPSwiGLU(W["g"], W["u"], W["d"], rank, tp)

    h = x.to(dev)
    for _ in range(L):                        # L layers => expect 2*L collectives
        h = h + attn(h)
        h = h + mlp(h)

    ref = x
    for _ in range(L):
        ref = reference_block(ref, W["q"], W["k"], W["v"], W["o"],
                              W["g"], W["u"], W["d"], n_q, n_kv)
    if rank == 0:
        print(f"tp={tp} layers={L} max|diff|={(h.cpu() - ref).abs().max():.2e} "
              f"collectives={STATS['collectives']} (expected {2 * L}) "
              f"bytes={STATS['bytes']}")
    dist.destroy_process_group()
```

Run it and read the three numbers:

```
torchrun --nproc-per-node 1 tp_shard.py   # sanity: diff exactly 0, collectives 8
torchrun --nproc-per-node 2 tp_shard.py
torchrun --nproc-per-node 4 tp_shard.py   # n_kv=2 < tp=4 -> replication path exercised
# no torchrun on PATH?  python3 -m torch.distributed.run --nproc-per-node 4 tp_shard.py
```

| What you should see | Meaning |
|---|---|
| `max|diff|` = 0.00e+00 at `tp=1`, ~1e-5 at `tp` > 1 (fp32, 4 layers) | your sharding is mathematically correct — the residual is **reduction order**, not a bug, and it grows with depth |
| `collectives == 2 × L` at every `tp` | column→row pairing worked; a count of `4 × L` means you all-gathered where you should have reduced |
| `bytes == 2 × L × T × H × 4` | matches [lesson 2](02-collectives-and-interconnects.md)'s `tokens × hidden × dtype` formula exactly |
| diff grows by orders of magnitude in bf16 | the nondeterminism from [lesson 3](03-tensor-parallelism.md) — this is why changing TP degree changes logits, and why you compare with `allclose` under greedy decoding rather than string equality |

## A6. The experiments that turn code into understanding

1. **Break a constraint on purpose.** Set `n_q = 6` with `tp = 4` and read the assertion. Then set `n_kv = 1, tp = 4`: the replication path runs, output is still correct, and **per-rank KV memory stopped shrinking** — compute the per-rank KV bytes/token both ways and confirm.
2. **Get the sharding wrong in the classic way.** Make `down_proj` column-parallel instead of row-parallel and observe garbage output with the *same* collective count. This is what a real TP bug looks like: no crash, wrong numbers.
3. **Measure `α` in situ** (GPU, `nccl`): time 1,000 iterations of the `L`-layer loop, then subtract the same loop with `all_reduce` stubbed out. Divide by `2 × L × 1000` → your per-collective latency. Compare against the `all_reduce_perf` number from lesson 2. They should agree within ~2×; a bigger gap means launch overhead you'd fix with CUDA graphs ([Phase 2 lesson 6](../phase-2/06-overhead-bound-and-cuda-graphs.md)).
4. **Sweep `T`** (token count) from 1 to 4096 and plot per-collective time. Find your own latency/bandwidth crossover and mark where decode (`T ≈ batch`) and prefill (`T ≈ prompt`) sit on it.
5. **Optional, high value:** load a real small model's weights (e.g. `Qwen/Qwen2.5-0.5B`) into this structure and check that the sharded forward matches `transformers`' logits. That's the step from "toy" to "I can shard a real model."

---

# Part B — the prefix-aware router (exit artifact)

Goal: reproduce [lesson 7](07-prefix-aware-routing.md)'s trade with numbers. Build the router once; run it against **mock backends** to explore policy space cheaply, then against **real vLLM replicas** for the headline table.

## B1. The ring and the policy

```python
#!/usr/bin/env python3
# labs/phase6/ring.py
import bisect, hashlib

BLOCK = 16   # match the engine's KV block size, or your key granularity is a lie

def h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")

class Ring:
    """Consistent hash ring with virtual nodes: a scale event moves ~1/R of keys."""
    def __init__(self, replicas=(), vnodes=160):
        self.vnodes, self.points, self.owner = vnodes, [], {}
        for r in replicas:
            self.add(r)

    def add(self, r):
        for i in range(self.vnodes):
            p = h64(f"{r}#{i}")
            bisect.insort(self.points, p)
            self.owner[p] = r

    def remove(self, r):
        for i in range(self.vnodes):
            p = h64(f"{r}#{i}")
            j = bisect.bisect_left(self.points, p)
            if j < len(self.points) and self.points[j] == p:
                self.points.pop(j)
                self.owner.pop(p, None)

    def candidates(self, key, k=2):
        """k DISTINCT replicas walking clockwise from the key's position."""
        if not self.points:
            return []
        out, i, n = [], bisect.bisect(self.points, h64(key)), len(self.points)
        for j in range(n):
            r = self.owner[self.points[(i + j) % n]]
            if r not in out:
                out.append(r)
                if len(out) == k:
                    break
        return out

def chained_block_hashes(tokens):
    """vLLM's scheme: hash of (previous block hash, this block's tokens).
    Longest common run of these == longest cacheable prefix."""
    out, prev = [], 0
    for i in range(0, len(tokens) - len(tokens) % BLOCK, BLOCK):
        prev = h64(f"{prev}:{tuple(tokens[i:i + BLOCK])}")
        out.append(prev)
    return out

def route_key(session_id, tokens, prefix_tokens=512):
    if session_id:                                  # exact, free, survives prompt edits
        return f"sid:{session_id}"
    n = (min(len(tokens), prefix_tokens) // BLOCK)  # BLOCK-ALIGNED prefix hash
    return "pfx:" + str(chained_block_hashes(tokens)[n - 1] if n else 0)

def choose(ring, key, load, k=2, eps=0.25):
    """Two-level policy: hash -> k candidates -> least-loaded, with bounded loads."""
    cands = ring.candidates(key, k) or list(load)
    mean = sum(load.values()) / max(len(load), 1)
    cap = (1 + eps) * mean + 1                      # +1 so an idle fleet isn't capped at 0
    ok = [c for c in cands if load[c] <= cap]
    return min(ok or list(load), key=lambda r: load[r])   # spill globally if both hot
```

`choose` is the whole policy: five lines, three tunables (`k`, `eps`, and the key), and every failure mode from lesson 7 is a specific setting of them.

## B2. The proxy

```python
#!/usr/bin/env python3
# labs/phase6/router.py — streaming reverse proxy over N OpenAI-compatible replicas
import argparse, asyncio, collections, json, time
import aiohttp
from aiohttp import web
from ring import Ring, choose, route_key

class Router:
    def __init__(self, replicas, policy="sticky", k=2, eps=0.25):
        self.replicas, self.policy = list(replicas), policy
        self.k, self.eps = k, eps
        self.ring = Ring(self.replicas)
        self.load = {r: 0 for r in self.replicas}          # outstanding requests
        self.routed = collections.Counter()
        self.rr = 0

    def pick(self, key):
        if self.policy == "rr":
            self.rr += 1
            return self.replicas[self.rr % len(self.replicas)]
        if self.policy == "hash":                          # policy ③: locality only
            return self.ring.candidates(key, 1)[0]
        return choose(self.ring, key, self.load, self.k, self.eps)   # policy ⑥

    def scale(self, replica, add=True):
        (self.ring.add if add else self.ring.remove)(replica)
        if add:
            self.replicas.append(replica); self.load[replica] = 0
        else:
            self.replicas.remove(replica); self.load.pop(replica)

async def handle(request):
    R, body = request.app["router"], await request.json()
    key = route_key(request.headers.get("X-Session-Id"),
                    request.app["tok"](body.get("prompt") or body["messages"][-1]["content"]))
    target = R.pick(key)
    R.load[target] += 1
    R.routed[target] += 1
    t0 = time.perf_counter()
    try:
        async with request.app["sess"].post(f"{target}/v1/completions", json=body) as up:
            resp = web.StreamResponse(status=up.status, headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            first = None
            async for chunk in up.content.iter_any():
                if first is None:
                    first = time.perf_counter() - t0                 # TTFT, measured here
                await resp.write(chunk)
            await resp.write_eof()
            request.app["log"].append({"target": target, "key": key, "ttft": first,
                                       "total": time.perf_counter() - t0})
            return resp
    finally:
        R.load[target] -= 1

async def metrics(request):
    R = request.app["router"]
    lines = [f'router_routed_total{{replica="{r}"}} {n}' for r, n in R.routed.items()]
    lines += [f'router_outstanding{{replica="{r}"}} {n}' for r, n in R.load.items()]
    mx, mean = max(R.routed.values(), default=0), (sum(R.routed.values()) / max(len(R.routed), 1)) or 1
    lines.append(f"router_imbalance {mx / mean:.4f}")
    return web.Response(text="\n".join(lines) + "\n")
```

Wire it up with `web.Application()`, an `aiohttp.ClientSession`, a tokenizer (`transformers.AutoTokenizer` for the real run; `str.split` for mocks), health-checking that calls `Router.scale(r, add=False)` on failure, and `/admin/scale` so you can add a replica mid-test.

## B3. Mock backends: explore the policy space without a GPU

```python
# labs/phase6/mock_backend.py — TTFT = a + b·uncached_tokens, with a real prefix cache
import collections
from ring import BLOCK, chained_block_hashes

class MockReplica:
    def __init__(self, name, capacity_blocks=4000, a=0.020, b=4e-5, tpot=0.020):
        self.name, self.a, self.b, self.tpot = name, a, b, tpot
        self.cache = collections.OrderedDict()     # block hash -> None (LRU)
        self.capacity, self.queries, self.hits = capacity_blocks, 0, 0

    def serve(self, tokens, out_tokens):
        hs = chained_block_hashes(tokens)
        matched = 0
        for x in hs:                               # longest PREFIX run present
            if x in self.cache:
                self.cache.move_to_end(x); matched += 1
            else:
                break
        self.queries += len(hs); self.hits += matched
        for x in hs[matched:]:
            self.cache[x] = None
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)         # LRU eviction: stale router views
        uncached = len(tokens) - matched * BLOCK
        return self.a + self.b * uncached + out_tokens * self.tpot, matched * BLOCK
```

Calibrate `a`, `b` and `tpot` from one real run of your engine (three requests: empty cache short prompt, empty cache long prompt, warm cache long prompt) so the simulator's numbers are anchored to your hardware rather than invented.

## B4. The session-shaped workload

This is the part that determines whether the experiment means anything. Requirements:

| Property | Why | Suggested default |
|---|---|---|
| Multi-turn sessions | the whole effect lives in turn ≥ 2 | turns ~ `1 + Geometric(p)`, mean 5 |
| Growing prompts | turn `n` contains turns `1…n-1` | 300-500 new tokens per turn |
| Shared system prompts, Zipf-distributed | models real products (a few hot prompts) | 20 prompts, `s = 1.1`, 1-2k tokens each |
| Think time between turns | otherwise you've built a closed loop | `Exp(mean 3 s)` |
| Open-loop session arrivals | [Phase 3 lesson 7](../phase-3/07-measuring-honestly.md) | Poisson at the target session rate |
| Fixed seed | policies must see identical traffic | `random.Random(0)` per run |
| A single-turn/unique-prompt control workload | proves your harness isn't fabricating gains | same generator, `turns = 1`, no shared prefix |

## B5. The experiment matrix and the table

Run every cell with the same seed and the same generator:

| Policy | Prefix hit rate | TTFT p50 | TTFT p99 | Imbalance (max/mean routed) | Goodput @ SLO |
|---|---|---|---|---|---|
| round-robin | | | | ~1.0 | |
| pure hash (③) | | | | | |
| **two-level, k=2, ε=0.25 (⑥)** | | | | | |
| oracle (best true match, load-blind) | | | | | |

Then four sweeps, each one a figure:

1. **`k` ∈ {1, 2, 3, R}** — locality vs balance; `k = R` degenerates to least-loaded.
2. **`ε` ∈ {0.05, 0.25, 1.0, ∞}** — the SLO knob. Show p99 TTFT and hit rate moving in opposite directions.
3. **Load ladder** — sessions/sec from 25% to 110% of capacity. The sticky policy's advantage should *shrink* near saturation (queueing dominates locality) — if it doesn't, suspect your generator.
4. **Scale event** — add a replica at t=120 s. Plot TTFT over time, and report (a) the spike height and duration, (b) the fraction of sessions rehashed (should be ≈`1/R`, and ≈`1` if you "accidentally" use `hash mod R` — run that variant deliberately, it's the most persuasive plot in the report).

Predict every cell before running. The control workload must show **no** improvement; if it shows one, your key is leaking load information.

## B6. Then do it for real

```bash
# two replicas, one GPU each (or two ports on one GPU with a small model)
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8001 \
  --enable-prefix-caching --max-model-len 8192 &
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8002 \
  --enable-prefix-caching --max-model-len 8192 &
python router.py --replicas http://127.0.0.1:8001 http://127.0.0.1:8002 \
  --policy sticky --k 2 --eps 0.25 --port 8000
python gen_sessions.py --url http://127.0.0.1:8000 --rate 2 --duration 300 --seed 0
```

Scrape hit rate **from the engines, not your router**: `vllm:gpu_prefix_cache_queries_total` and `vllm:gpu_prefix_cache_hits_total` per replica, differenced across the run. The gap between your router's *predicted* hits and the engines' *actual* hits is itself a finding — it's eviction and staleness, and quantifying it is what separates this from a toy.

---

## Deliverables checklist

**Part A** (`projects/11-manual-tensor-parallel/`): `tp_shard.py`; the verification output at `tp` ∈ {1, 2, 4}; the collective count and bytes matching `2·L` and `2·L·T·H·dtype`; the KV-replication finding at `tp > n_kv`; the "wrong sharding" negative result; and (if GPU) your measured `α` next to lesson 2's `nccl-tests` number.

**Part B** (`projects/12-sticky-load-balancer/`): `ring.py`, `router.py`, `mock_backend.py`, `gen_sessions.py`; the four-policy table on mocks **and** on real replicas; the four sweep figures; the scale-event plot with rehash fraction; the router-predicted vs engine-measured hit-rate gap; a limits section (what your router doesn't do: KV-event awareness, disaggregation-aware routing, multi-region); and exact commands + seeds + versions so someone else can rerun it.

---

**Next:** [Exercises & exit artifact →](11-exercises-and-artifacts.md) — the full exercise set, the twenty-question self-check, and what "Phase 6 done" means.
