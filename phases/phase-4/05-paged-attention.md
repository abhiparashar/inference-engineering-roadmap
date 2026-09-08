# 5 — PagedAttention

> **You'll be able to say:** "PagedAttention is OS virtual memory for the KV-cache. Instead of one contiguous slab per sequence sized to `max_len`, the cache is fixed-size blocks (16 tokens) from a shared pool, and each sequence has a **block table** mapping logical positions to physical blocks. I simulated all three allocators on the same workload: reserving `max_len` gives **30.1%** memory utilization, oracle-contiguous gives 76.6%, paging gives **98.8%** — 54.9 → 171.5 mean concurrent sequences, a **3.1× throughput multiplier from an allocator change alone**. And because blocks are shareable, prefix sharing with copy-on-write falls out for free."

This is the idea that made vLLM famous (SOSP '23), and it is the clearest example in the whole roadmap of systems knowledge — not ML knowledge — producing a large win.

---

## The problem: three kinds of waste

Before paging, an engine allocated each sequence one contiguous KV buffer. It cannot know the output length in advance, so it must reserve for the worst case:

```
  sequence A: prompt 512, will generate 87 tokens, max_len 2048
  ┌───────────────────────────────────────────────────────────┐
  │ prompt 512 │ generated 87 │ ← reserved but never used → 1449 │
  └───────────────────────────────────────────────────────────┘
    used 599 of 2048 = 29%          INTERNAL FRAGMENTATION (over-reservation)

  the pool, with three such sequences and one free gap of 300:
  [ A: 2048 ][ B: 2048 ][ 300 free ][ C: 2048 ]
                          ↑ too small for a new 2048-slab even though total free is plenty
                            EXTERNAL FRAGMENTATION

  and one more: a system prompt shared by 50 requests is stored 50 times
                            NO SHARING
```

The vLLM paper measured exactly this on production-style workloads: existing systems wasted **60-80%** of KV memory. And since [lesson 4](04-kv-cache-optimization.md) showed concurrency = memory ÷ bytes, wasted memory is wasted throughput, one for one.

---

## The idea: blocks and a block table

Steal the OS solution to exactly this problem — virtual memory.

```
   LOGICAL view (what the model sees)          PHYSICAL view (what HBM holds)
   sequence A, 40 tokens                       a pool of fixed 16-token blocks
   ┌────┬────┬────┐                            ┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐
   │0-15│16-31│32-39│                          │ b0 ││ b1 ││ b2 ││ b3 ││ b4 ││ b5 │…
   └──┬─┴──┬──┴──┬──┘                          └────┘└────┘└────┘└────┘└────┘└────┘
      │    │     │        block table A = [b3, b0, b7]   (any order, non-contiguous)
      └────┴─────┴──────▶ block table B = [b3, b1]       (b3 SHARED: same prefix!)

   OS term            engine term
   ─────────────────  ────────────────────────────
   virtual page       logical block (16 tokens of one sequence)
   physical frame     physical KV block in the pool
   page table         block table (per sequence)
   page fault         allocate-on-demand when the last block fills
   copy-on-write      fork a shared block when two sequences diverge
```

Consequences, in the order they matter:

1. **Allocation is on demand, one block at a time.** A sequence holds `ceil(len/16)` blocks — internal waste is at most 15 tokens, *not* `max_len − len`.
2. **No contiguity requirement**, so external fragmentation disappears entirely. Any free block fits any sequence.
3. **Blocks can be shared** between sequences (identical prefixes, parallel samples of one prompt, beam search) with reference counts and copy-on-write on divergence.
4. **The attention kernel must gather** K/V through the block table instead of reading a contiguous tensor. That is the cost, and it's why PagedAttention is a *kernel* as well as an allocator.

---

## Measured: what the allocator is worth

Same workload (4,000 requests, prompts 64-1024, generations 16-512, `max_len` 2048), same 120,000-token pool (≈62 GB at 512 KB/token), three allocators:

```
pool = 120,000 KV tokens, max_len = 2048, 4000 requests

allocator               KV utilization  effective waste
reserve_max                      30.1%            69.9%
reserve_final                    76.6%            23.4%
paged                            98.8%             1.2%

allocator               mean concurrent seqs
reserve_max                             54.9
reserve_final                          129.8
paged                                  171.5
```

- **30.1% utilization for `max_len` reservation** reproduces the paper's 60-80%-waste claim on a synthetic-but-plausible workload. Two thirds of an expensive GPU's memory, holding nothing.
- **`reserve_final` is an oracle** — it reserves exactly the length the request will end up needing, which no real system can know. Even that only reaches 76.6%, because the reservation is held from the start while the sequence is still short.
- **Paging reaches 98.8%** and turns 54.9 concurrent sequences into 171.5: a **3.1× throughput multiplier with no change to the model, the kernel math, or the schedule.**

Block size sensitivity, same simulation:

```
paged block=1                          171.5
paged block=8                          171.5
paged block=16                         171.5
paged block=32                         170.1
paged block=128                        159.1
```

Blocks of 8-16 tokens are indistinguishable from perfect (block=1) allocation, while 128 costs ~7%. That is why **16 is the default in vLLM**: small enough that internal waste is negligible, large enough that the block table stays small and the kernel gets a contiguous run of 16 tokens to read at a time. This measured flat region is the justification for the constant you'll see in every config file.

The simulation is ~40 lines and worth writing yourself — it's exercise 1.

---

## What it costs: the kernel

A standard attention kernel reads `K[seq, :, :, :]` as one contiguous tensor. A paged kernel receives a **block table** and gathers:

```
  for each block index b in block_table[seq]:
      load K_block = kv_cache[b]        # 16 tokens, contiguous within the block
      accumulate partial attention (online softmax, FlashAttention-style)
```

So each block read is still coalesced; only the *sequence* of reads is indirect. The overheads are one extra indirection per block, a slightly more complex kernel, and reduced ability to use off-the-shelf attention implementations — which is precisely why vLLM wrote its own, and why FlashAttention gained paged-KV support later. **The measured cost is a few percent of attention time; the benefit is 2-4× concurrency.** That trade is not close.

Two structural bonuses fall out of the same design:

- **Copy-on-write sharing.** Parallel sampling (`n=4`) or beam search share the prompt's blocks with a refcount; only the diverging block gets copied. The vLLM paper reports ~55% memory saving on parallel sampling/beam workloads.
- **Prefix caching.** If block contents are content-addressed (hash of the tokens), two *different requests* with the same system prompt reuse the same physical blocks — [lesson 6](06-prefix-caching-and-radix-attention.md).

---

## How it looks in code (and in vLLM)

The data structures are simple enough to write on a whiteboard, which is exactly what an interviewer may ask for:

```python
class BlockPool:
    def __init__(self, num_blocks, block_size):
        self.block_size = block_size
        self.free = list(range(num_blocks))          # a free list, nothing fancier
        self.ref  = [0] * num_blocks                 # refcounts enable sharing + CoW

    def allocate(self):                              # one block
        b = self.free.pop(); self.ref[b] = 1; return b

    def fork(self, b):                               # share instead of copy
        self.ref[b] += 1; return b

    def free_block(self, b):
        self.ref[b] -= 1
        if self.ref[b] == 0: self.free.append(b)

class Sequence:
    block_table: list[int]                           # logical block -> physical block
    length: int

    def append_token(self, pool):
        if self.length % pool.block_size == 0:       # "page fault": need a new block
            self.block_table.append(pool.allocate())
        self.length += 1

    def slot(self, pos, pool):                       # logical position -> physical slot
        return self.block_table[pos // pool.block_size] * pool.block_size + pos % pool.block_size
```

That `slot()` function — the address translation — *is* PagedAttention. Everything else is bookkeeping and kernels.

**Where to read the real thing:** `vllm/core/block_manager.py` and `vllm/core/block/` (V0), `vllm/v1/core/block_pool.py` and `kv_cache_manager.py` (V1); the kernel in `csrc/attention/paged_attention_v1.cu` / `v2.cu` and the newer FlashAttention-backed paths in `vllm/attention/backends/`. Look specifically for:

- `can_allocate` returning `OK / LATER / NEVER` — [Phase 3 lesson 6](../phase-3/06-scheduling-policies-and-admission-control.md)'s memory gate, implemented.
- The **watermark** — a few percent of blocks held back so running sequences can always append.
- `append_slots`, `fork`, `swap_in`/`swap_out`, and the refcount/CoW logic.
- `--block-size` and `--gpu-memory-utilization` in the server args: the two flags that expose this whole design.

---

## Where paging does *not* help

Say this too, or you'll oversell it:

- **Nothing changes for a single sequence.** Paging is a multi-tenancy win; batch 1 sees only kernel overhead.
- **It doesn't reduce bytes per token** — that's quantization and GQA. It reduces *wasted* bytes.
- **Uniform, short, known-length workloads** (classification, embeddings, fixed-length generation) have little fragmentation to reclaim; `reserve_final` was already 76.6%, and with uniform lengths it approaches 100%.
- **It adds failure modes**: block-table bugs are silent corruption rather than crashes, and preemption/eviction policy now interacts with sharing refcounts.

---

## Try it

1. **Write the fragmentation simulator** (the three allocators above) and reproduce the table for *your* workload's length distribution. Report utilization and mean concurrency. Then sweep block size and find your flat region.
2. **Sweep `max_len`** in the `reserve_max` case: utilization is roughly `mean_len / max_len`, so a client library that defaults `max_tokens=4096` costs you a specific, computable amount of GPU memory. Quantify it — it's a great "we changed one default and got 2× throughput" story.
3. **Implement `BlockPool` + block tables** in your Phase-3 engine ([lesson 9](09-build-paged-kv-and-quant-bench.md) does this properly): allocate per block, free on eviction, and report the pool's utilization in `/metrics`.
4. **Add copy-on-write** and serve `n=4` parallel samples of one prompt. Measure the memory saving versus four independent sequences; predict it first from `prompt_len / total_len`.
5. **Read `block_manager.py` with your own implementation open** and list five things vLLM handles that yours doesn't (watermark, swapping, sliding-window blocks, prefix hashing, block-table tensors for the kernel). That list is your Phase 5 reading plan.

---

## Key takeaways

- **PagedAttention = OS virtual memory for the KV-cache**: fixed-size blocks from a shared pool, a per-sequence block table, allocation on demand, no contiguity requirement.
- Measured on one workload: utilization **30.1% (reserve max_len) → 76.6% (oracle contiguous) → 98.8% (paged)**; mean concurrency **54.9 → 171.5 (3.1×)**. Pure allocator change, zero model change.
- **Block size 8-16 is indistinguishable from perfect**; 128 costs ~7%. That's why the default is 16.
- Three wastes eliminated: **over-reservation** (`max_len` you'll never reach), **external fragmentation** (unusable gaps), and **duplication** (shared prefixes stored once, via refcounts + copy-on-write).
- **The cost is a gather-based attention kernel** — a few percent of attention time for 2-4× concurrency, plus the loss of drop-in third-party attention implementations.
- **Address translation is the whole idea**: `physical_slot = block_table[pos // B] * B + pos % B`. If you can write that on a whiteboard and explain the refcount/CoW path, you understand vLLM's core.
- It reduces *wasted* bytes, not bytes per token, and does nothing for a single sequence or for fixed-length workloads.

**Next:** [Prefix caching & RadixAttention →](06-prefix-caching-and-radix-attention.md) — now that blocks are shareable, stop recomputing the same 2,000-token system prompt for every request.
