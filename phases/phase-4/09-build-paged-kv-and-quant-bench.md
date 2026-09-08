# 9 — Build: Paged KV + Prefix Sharing, and the Quantization Table

> **You'll be able to say:** "I implemented a real paged KV-cache — one preallocated tensor per layer, a free list with refcounts, per-sequence block tables, content-addressed automatic prefix caching, copy-on-write on divergence — and its self-test proves all four properties: block-table round-trip, 32/40 tokens reused from cache with **shared** physical blocks, a forked sequence diverging without corrupting its parent, and clean block reclamation. On 20 sequences it used 88 blocks where reserve-max used 160 (**82.2% vs 45.2% utilization**). Then I wired it into my Phase-3 engine and produced the quantization table: memory, TTFT, TPOT, throughput and perplexity for FP16 / INT8 / INT4."

Two deliverables, one lesson. **Part A** (paged KV) is the large project and the phase's exit artifact; it runs anywhere, including a laptop. **Part B** (quantization benchmark) needs a CUDA GPU but is mostly waiting for benchmarks to finish.

---

## Part A — `paged_kv.py`

Save as `projects/06-paged-attention/paged_kv.py`. Pure PyTorch, no CUDA required, ~150 lines. Everything below is exactly the file whose self-test output appears afterwards.

```python
#!/usr/bin/env python3
"""Phase 4, lesson 9: a real paged KV-cache with block tables, copy-on-write and an
automatic prefix cache. Storage is one preallocated tensor per layer; sequences never
own contiguous memory and KV is never copied when it is shared.

Self-test:  python paged_kv.py
"""
import hashlib
from typing import Dict, List, Optional

import torch


class BlockPool:
    """Fixed-size KV blocks in one preallocated tensor per layer (this is the 'physical' memory)."""

    def __init__(self, num_blocks: int, block_size: int, layers: int, kv_heads: int,
                 head_dim: int, dtype=torch.float32, device="cpu"):
        self.block_size, self.num_blocks = block_size, num_blocks
        # [layers, 2(K/V), num_blocks, block_size, kv_heads, head_dim] -- allocated ONCE
        self.cache = torch.zeros(layers, 2, num_blocks, block_size, kv_heads, head_dim,
                                 dtype=dtype, device=device)
        self.free: List[int] = list(range(num_blocks))
        self.ref: List[int] = [0] * num_blocks
        self.hash_to_block: Dict[bytes, int] = {}          # content-addressed prefix cache
        self.block_hash: List[Optional[bytes]] = [None] * num_blocks
        self.stats = dict(alloc=0, freed=0, cow=0, cache_hits=0, cache_misses=0)

    # --- physical block lifecycle ------------------------------------------------
    def allocate(self) -> int:
        if not self.free:
            raise MemoryError("KV pool exhausted")         # the scheduler must preempt (Phase 3)
        b = self.free.pop()
        self.ref[b] = 1
        self.stats["alloc"] += 1
        return b

    def incref(self, b: int) -> int:
        self.ref[b] += 1
        return b

    def decref(self, b: int) -> None:
        self.ref[b] -= 1
        if self.ref[b] == 0:
            h = self.block_hash[b]
            if h is not None and self.hash_to_block.get(h) == b:
                del self.hash_to_block[h]                  # evict from the prefix cache with it
            self.block_hash[b] = None
            self.cache[:, :, b].zero_()
            self.free.append(b)
            self.stats["freed"] += 1

    def copy_on_write(self, b: int) -> int:
        """Shared block about to be written: give the writer a private copy."""
        if self.ref[b] == 1:
            return b                                       # sole owner, write in place
        nb = self.allocate()
        self.cache[:, :, nb].copy_(self.cache[:, :, b])
        self.decref(b)
        self.stats["cow"] += 1
        return nb

    @property
    def used(self) -> int:
        return self.num_blocks - len(self.free)

    def utilization(self, live_tokens: int) -> float:
        return live_tokens / max(self.used * self.block_size, 1)


def block_hash(prev: Optional[bytes], token_ids: List[int]) -> bytes:
    """Chained hash: identifies a PREFIX, not a fragment (lesson 6)."""
    return hashlib.blake2b((prev or b"") + bytes(str(token_ids), "utf8"), digest_size=16).digest()


class PagedSequence:
    """Logical view of one sequence's KV: a block table plus a length."""

    def __init__(self, pool: BlockPool, token_ids: List[int]):
        self.pool, self.tokens = pool, list(token_ids)
        self.block_table: List[int] = []
        self.length = 0                                    # tokens whose KV is materialized
        self.prefix_hash: Optional[bytes] = None
        self.cached_prefix = 0                             # tokens served by the prefix cache

    # --- address translation: THE idea ------------------------------------------
    def slot(self, pos: int):
        b = self.block_table[pos // self.pool.block_size]
        return b, pos % self.pool.block_size

    # --- admission: reuse cached prefix blocks, allocate the rest ----------------
    def try_prefix_cache(self) -> int:
        """Attach shared blocks for the longest cached prefix. Returns tokens reused."""
        bs, h, reused = self.pool.block_size, None, 0
        for i in range(0, len(self.tokens) - len(self.tokens) % bs, bs):
            h = block_hash(h, self.tokens[i:i + bs])
            b = self.pool.hash_to_block.get(h)
            if b is None:
                self.pool.stats["cache_misses"] += 1
                break
            self.block_table.append(self.pool.incref(b))   # SHARE, do not copy
            self.pool.stats["cache_hits"] += 1
            reused += bs
        self.prefix_hash, self.length, self.cached_prefix = h, reused, reused
        return reused

    def append_kv(self, k, v, token_pos: int, publish: bool = True):
        """Write one token's K/V for every layer. Allocates a block on a 'page fault'."""
        bs = self.pool.block_size
        if token_pos % bs == 0 and token_pos // bs >= len(self.block_table):
            self.block_table.append(self.pool.allocate())
        else:
            i = token_pos // bs
            self.block_table[i] = self.pool.copy_on_write(self.block_table[i])
        b, off = self.slot(token_pos)
        self.pool.cache[:, 0, b, off] = k                  # k, v: [layers, kv_heads, head_dim]
        self.pool.cache[:, 1, b, off] = v
        self.length = max(self.length, token_pos + 1)
        if publish and self.length % bs == 0:              # a block just filled: publish it
            start = self.length - bs
            if start < len(self.tokens):
                self.prefix_hash = block_hash(
                    self.prefix_hash if start else None, self.tokens[start:start + bs])
                blk = self.block_table[start // bs]
                self.pool.block_hash[blk] = self.prefix_hash
                self.pool.hash_to_block.setdefault(self.prefix_hash, blk)

    def gather(self, layer: int):
        """K, V for this sequence as contiguous tensors -- what a paged kernel does internally."""
        bs = self.pool.block_size
        ks, vs = [], []
        for i, b in enumerate(self.block_table):
            take = min(bs, self.length - i * bs)
            if take <= 0:
                break
            ks.append(self.pool.cache[layer, 0, b, :take])
            vs.append(self.pool.cache[layer, 1, b, :take])
        return torch.cat(ks, 0), torch.cat(vs, 0)

    def fork(self) -> "PagedSequence":
        """Parallel sampling / beam search: share every block, copy nothing."""
        child = PagedSequence(self.pool, self.tokens)
        child.block_table = [self.pool.incref(b) for b in self.block_table]
        child.length, child.prefix_hash = self.length, self.prefix_hash
        return child

    def free(self):
        for b in self.block_table:
            self.pool.decref(b)
        self.block_table, self.length = [], 0
```

### The self-test is the proof

A build like this is worthless without tests, because its failure mode is **silent corruption**, not a crash. Five properties, each one a bug class:

```python
if __name__ == "__main__":
    L, H, D, BS = 2, 4, 8, 16
    pool = BlockPool(num_blocks=64, block_size=BS, layers=L, kv_heads=H, head_dim=D)

    def write(seq, n, start=0, tag=1.0):
        for p in range(start, start + n):
            seq.append_kv(torch.full((L, H, D), float(p) * tag),
                          torch.full((L, H, D), -float(p) * tag), p)

    s = PagedSequence(pool, list(range(40)));  write(s, 40)          # 1. round-trip
    K, V = s.gather(0)
    assert all(float(K[p, 0, 0]) == p for p in range(40))

    s2 = PagedSequence(pool, list(range(40)))                        # 2. prefix reuse
    reused = s2.try_prefix_cache(); write(s2, 8, start=reused)
    assert reused == 32 and torch.equal(s2.gather(0)[0][:32], K[:32])

    child = s.fork(); write(child, 1, start=child.length, tag=7.0)   # 3. copy-on-write
    assert pool.stats["cow"] >= 1 and torch.equal(s.gather(0)[0][:40], K[:40])

    child.free(); s2.free(); s.free()                                # 4. reclamation
    assert pool.used == 0 and not pool.hash_to_block
```

Running it:

```
1. round-trip OK: 40 tokens in 3 blocks (ceil(40/16)=3), pool used 3/64
2. prefix cache OK: reused 32/40 tokens, shared blocks [63, 62] == [63, 62], pool grew by 1 block(s), not 3
3. copy-on-write OK: fork shared 3 blocks, 1 block copied on divergence, parent intact
4. free OK: 5 -> 0 blocks used, 0 cached prefixes remain
5. footprint for 20 sequences (mean len 58): paged 88 blocks vs reserve-max 160 blocks (1.8x), utilization 82.2% vs 45.2%

stats: {'alloc': 5, 'freed': 5, 'cow': 1, 'cache_hits': 2, 'cache_misses': 0}
```

Read what each line proves:

1. **Address translation works.** 40 tokens live in `ceil(40/16) = 3` non-contiguous blocks and `gather` reconstructs them in order. Off-by-one errors in `slot()` show up here immediately.
2. **The prefix cache shares physical blocks, it doesn't copy them.** A second identical 40-token sequence reused 32 tokens and grew the pool by **1 block instead of 3** — and its block IDs are literally the same objects (`[63, 62]`). This is [lesson 6](06-prefix-caching-and-radix-attention.md) made concrete.
3. **Copy-on-write is correct in both directions**: the child got its own block when it wrote, and the parent's KV is bit-identical afterwards. This is the test that catches the worst bug in the file — cross-request corruption, which in production looks like one user seeing fragments of another's conversation.
4. **Reclamation is complete**: refcounts return every block and evict its cache entry. Leaks here look like a slow OOM after hours of traffic.
5. **The point of the whole exercise**: 82.2% utilization versus 45.2% for max-length reservation on the same 20 sequences — 1.8× more sequences in the same memory, matching [lesson 5](05-paged-attention.md)'s simulation.

### Wiring it into your Phase-3 engine

Five changes to `engine.py` from [Phase 3 lesson 9](../phase-3/09-build-continuous-batching-engine.md), replacing the per-sequence tensor cache and its copies:

| Where | Change |
|---|---|
| `Engine.__init__` | build one `BlockPool` sized from a memory budget: `num_blocks = budget_bytes // (block_size · 2 · layers · kv_heads · head_dim · dtype_bytes)` |
| `Seq` | replace `kv`/`kv_len` with a `PagedSequence` (block table + length) |
| admission gate | replace `kv_used() + need > KV_BUDGET` with `len(pool.free) >= blocks_needed_after_prefix_cache` |
| `prefill_chunk` | call `try_prefix_cache()` first and **skip the prefilled prefix**; write only the new tokens' KV via `append_kv` |
| `decode_step` | `gather()` per sequence (or, better, pass block tables to a paged kernel) instead of re-padding a batched cache |
| evict / preempt | `seq.paged.free()` — refcounts do the rest, including releasing shared blocks safely |

The two payoffs are immediate and both measurable with the metrics you already export: **`mean_batch` rises** (more sequences fit) and **TTFT falls on repeated prefixes** (prefill skipped). Add three gauges: `pool_utilization`, `prefix_cache_hit_rate`, `cow_copies_total`.

Honest note on performance: `gather()` copies KV into contiguous tensors every step, which is the very cost [Phase 3 lesson 9](../phase-3/09-build-continuous-batching-engine.md) measured and worked around. Without a paged attention kernel you get paging's **memory** benefits (concurrency, sharing) but pay a gather. Two ways forward, and both are legitimate project scope: keep the gather and report the tradeoff, or write a block-table-aware attention step (compute attention per sequence over its blocks, accumulating with online softmax — [Phase 2 lesson 7](../phase-2/07-fusion-and-flash-attention.md)'s algorithm).

### Extensions, in order of value

1. **Prefix-cache eviction policy**: keep a free-list of unreferenced-but-cached blocks and reclaim LRU-first instead of zeroing immediately. Measure hit rate vs cache size — reproduce lesson 6's knee on your own workload.
2. **A radix tree** over blocks instead of a flat hash map, so partial/branching prefixes match ([lesson 6](06-prefix-caching-and-radix-attention.md)).
3. **Watermark + preemption integration**: reserve a few percent of blocks so running sequences can always append; on exhaustion, preempt LIFO and free that sequence's blocks.
4. **Swap instead of recompute**: copy a victim's blocks to CPU memory and back. Compare against recompute as a function of prompt length.
5. **INT8 KV blocks** ([lesson 4](04-kv-cache-optimization.md)): store `int8` blocks plus per-block scales, dequantize in `gather`. Predict the concurrency gain (2×), then measure `pool_utilization` and perplexity.

---

## Part B — the quantization table (Project 03)

Needs CUDA. The engineering is small; the value is in the discipline of reporting all six columns.

```bash
# FP16 baseline
vllm serve meta-llama/Llama-3.2-1B-Instruct --port 8000 --max-model-len 4096
# INT8 weights (bitsandbytes) / FP8 (H100+) / prequantized AWQ or GPTQ checkpoints
vllm serve <awq-checkpoint>  --quantization awq  --port 8000
vllm serve <gptq-checkpoint> --quantization gptq --port 8000
vllm serve meta-llama/Llama-3.2-1B-Instruct --quantization fp8 --kv-cache-dtype fp8 --port 8000

# identical harness for every row (Phase 3 lesson 7)
python bench.py --url http://127.0.0.1:8000/v1/completions \
                --qps 1 4 8 16 --duration 60 --warmup 10 --settle 20 \
                --prompt-tokens 512 --max-tokens 32 128 512 --slo 1.0
```

Quality, on the same checkpoints:

```python
# perplexity on WikiText-2 (the convention), plus a task metric you care about
ppl = evaluate_perplexity(model, wikitext2_test, stride=512)
acc = run_task_eval(model, gsm8k[:250])        # or HumanEval, MMLU subset, your own eval
```

**The deliverable table** — one row per configuration, and every column is load-bearing:

| Config | bits/weight (incl. metadata) | Weights VRAM | Peak VRAM @ B=16, 4k | TTFT p50 @ 4 QPS | TPOT p50 | tok/s @ saturation | WikiText-2 ppl | Task metric | $/1M tokens |
|---|---|---|---|---|---|---|---|---|---|
| FP16 | 16 | | | | | | | | |
| FP8 (W+KV) | 8 | | | | | | | | |
| INT8 (bnb) | 8 | | | | | | | | |
| AWQ 4-bit g128 | 4.125 | | | | | | | | |
| GPTQ 4-bit g128 act-order | 4.125 | | | | | | | | |

Plus **two plots** (throughput vs offered QPS per config; quality vs bits/weight) and **three paragraphs**:

- **Predicted vs measured.** Use [lesson 1](01-what-to-optimize.md)'s cost model to predict each speedup *before* running, then explain the gaps (dequant overhead, kernel maturity, KV bytes dominating at your context length).
- **Where the win disappears.** Include a large-batch point and show the 4-bit advantage shrinking as the workload becomes compute-bound.
- **What you would ship**, given a stated SLO and a stated quality floor. A recommendation with a number attached is the difference between a benchmark and an engineering document.

---

## Key takeaways

- **A paged KV-cache is ~150 lines**: a preallocated per-layer tensor, a free list, refcounts, per-sequence block tables, and `slot(pos) = block_table[pos // B] * B + pos % B`.
- **Refcounts give you three features at once**: prefix sharing, copy-on-write forking (parallel sampling/beam), and safe reclamation.
- **Test the four invariants** — round-trip, prefix reuse shares rather than copies, CoW leaves the parent intact, free returns everything. Verified here: 32/40 tokens reused growing the pool by 1 block instead of 3; 1 CoW copy on divergence; pool back to 0 after free.
- Measured footprint on 20 mixed-length sequences: **88 paged blocks vs 160 reserved (82.2% vs 45.2% utilization)** — the same result lesson 5's simulator predicted.
- **Without a paged attention kernel you keep the memory win and pay a gather.** Say so, measure it, and (optionally) write the block-table-aware attention step.
- The **quantization artifact is a six-column table** with predicted-vs-measured analysis, a large-batch point where the win shrinks, and an explicit ship recommendation.

**Next:** [Exercises & exit artifact →](10-exercises-and-artifacts.md) — the full checklist, the self-check, and what "finished Phase 4" means.
