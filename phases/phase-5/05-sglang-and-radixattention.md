# 5 — SGLang and RadixAttention

> **You'll be able to say:** "RadixAttention promotes the prefix cache from an optimization to a *scheduling input*. KV blocks live in a radix tree keyed by token sequence; `match_prefix()` returns the longest matched node, nodes in use are lock-refed so they can't be evicted, and the waiting queue is sorted by **longest prefix match** so that requests which share a prefix run together and hit the cache while it's still hot. Same data structure I built in Phase 4, but the scheduler consults it before deciding what to run — which is what makes SGLang strong on agent, few-shot and multi-turn workloads. Its other differentiator is structured output: a grammar compiled to an FSM that masks logits per step, with jump-forward for deterministic spans."

If vLLM is the general-purpose engine, SGLang is the engine that assumes **your prompts repeat**, and builds the scheduler around that assumption.

---

## The two ideas

```
  1. RADIXATTENTION — the KV cache is a radix tree over token sequences
                     (not a flat hash map, not per-request slabs)

        root
         ├── "You are a helpful assistant.\n"            ← 8 requests share these blocks
         │      ├── "Q: What is 2+2?"                     ← request A
         │      └── "Q: Summarize this doc: ..."          ← requests B, C (share more)
         └── "<|system|> You are a code reviewer..."      ← a different tenant/app

     match_prefix(tokens) → longest matching path, its KV blocks, and the node
     insert(tokens, kv)   → extend/split the tree as sequences grow
     evict(n)             → LRU over LEAVES only; lock-refed nodes are protected

  2. CACHE-AWARE SCHEDULING — the waiting queue is sorted by match length
                              (policy "lpm": longest prefix match first)

     naive FCFS:   A(sys1) B(sys2) C(sys1) D(sys2)  → cache thrashes between prefixes
     lpm ordering: A(sys1) C(sys1) B(sys2) D(sys2)  → each prefix loaded once, reused hot
```

Idea 1 you already implemented ([Phase 4 lesson 6](../phase-4/06-prefix-caching-and-radix-attention.md)). **Idea 2 is the part that's easy to miss and hard to retrofit**: it changes the *order* requests run in, which trades fairness for hit rate. That's a scheduling policy decision in the sense of [Phase 3 lesson 6](../phase-3/06-scheduling-policies-and-admission-control.md), with the same starvation hazard, which is why SGLang ships aging policies (`hrrn`) alongside it.

---

## The code

| Component | File |
|---|---|
| Scheduler loop (the event loop of the whole engine) | `python/sglang/srt/managers/scheduler.py` |
| Queue ordering policies + `PrefillAdder` | `python/sglang/srt/managers/schedule_policy.py` |
| Batch representation (`Req`, `ScheduleBatch`) | `python/sglang/srt/managers/schedule_batch.py` |
| Radix tree prefix cache | `python/sglang/srt/mem_cache/radix_cache.py` |
| Hierarchical (GPU→CPU→disk) cache | `python/sglang/srt/mem_cache/hiradix_cache.py` |
| Sliding-window variant | `python/sglang/srt/mem_cache/swa_radix_cache.py` |
| Token/page allocators | `python/sglang/srt/mem_cache/allocator/{token,paged}.py` |
| Grammar backends | `python/sglang/srt/constrained/{xgrammar,outlines,llguidance}_backend.py` |

### `radix_cache.py`, the parts that matter

- **`match_prefix(...) → MatchResult`**: walks the tree token-by-token (page-aligned when `page_size > 1`) and returns the matched KV indices plus the last matched node. The scheduler uses the *length* of this match as a priority signal, not just as a compute saving.
- **`insert(...)`**: extends the tree, splitting a node when two sequences diverge mid-node. Node splitting is the operation a flat block-hash map (vLLM's design) doesn't need — vLLM gets the same effect by chaining block hashes. Tree vs chained-hash-map is a genuine design fork with the same big-O and different constants; SGLang's tree makes *subtree* operations (eviction, accounting, ordering) natural, which is exactly what cache-aware scheduling needs.
- **`inc_lock_ref(node)` / `dec_lock_ref(node)`**: pin every node on the path of a *running* request so eviction can't reclaim KV that's in use. This is the refcount from your `BlockPool`, expressed on tree nodes.
- **`evict(...)`**: LRU over evictable leaves. Evicting an interior node is impossible while children exist — the tree structure enforces "you cannot drop a prefix that a longer cached sequence depends on", which in the flat design has to be enforced by hashing discipline.
- **`cache_unfinished_req()` / `cache_finished_req()`**: the write path. A running request's blocks are inserted into the tree as it goes, so a *concurrent* request with the same prefix can attach to it immediately, not only after the first one finishes.

### `schedule_policy.py`, the ordering

The queue policies, from the enum in the source:

| Policy | Behavior | Use when |
|---|---|---|
| `lpm` (default, cache-aware) | Longest prefix match first | Shared system prompts, few-shot, agents, multi-turn |
| `dfs-weight` | Depth-first over the radix tree, weighted by subtree size | Heavily branched prompt trees (parallel sampling, tree-of-thought) |
| `hrrn` | Highest response ratio next — token-based aging | Cache-aware ordering without starving old requests |
| `fcfs` | First come, first served | Fairness/latency predictability; unique prompts |
| `lof` | Longest output first | Throughput-oriented offline batches |
| `random` | Baseline | Experiments |

`PrefillAdder` is the counterpart of vLLM's token budget: it decides how many waiting requests can join this prefill given the remaining token budget and KV space, and it subtracts the *cached* prefix from a request's cost — a request whose 2,000-token prompt is 90% cached costs ~200 tokens of budget, so many more of them fit in one step. **That's the compounding effect**: cache hits don't just skip compute, they make the batch bigger.

Also read `--schedule-conservativeness` (how aggressively new requests are admitted given a risk of running out of KV) — it's the same admission-control dial as TGI's `waiting_served_ratio`, aimed at a different failure.

---

## Structured / constrained output

The second thing SGLang is known for, and increasingly the reason people pick it, is **grammar-constrained decoding**: guarantee JSON that matches a schema, or output that matches a regex/EBNF grammar.

```
   per decode step:
     logits ──▶ apply token mask from the grammar's current FSM state ──▶ sample
                  (disallowed tokens set to −inf)
                        │
                        └── advance the FSM by the sampled token

   optimization: JUMP-FORWARD DECODING
     if the FSM has only one legal continuation for the next k tokens
     (e.g. after `{"name":` the string `"` is forced), emit them WITHOUT
     a forward pass — several free tokens, and lower latency.
```

Implementation notes worth carrying to interviews:

- The mask is computed on the **CPU** from a compiled grammar (xgrammar, Outlines, or llguidance backends live under `srt/constrained/`) and applied to logits on the GPU. Grammar *compilation* is the expensive part and is cached per schema — first request with a new schema pays it. That's a real latency cliff to know about.
- Constrained decoding changes the token distribution deliberately; it is **not** lossless in the speculative-decoding sense ([Phase 4 lesson 7](../phase-4/07-speculative-decoding.md)). It removes the tail rather than preserving it.
- Jump-forward decoding interacts with tokenization: emitting forced characters can produce a different token boundary than the model would have chosen, and backends handle re-tokenization carefully. This is a good "read the code to see the ugly detail" target.
- vLLM has the same feature (`structured_outputs` / `guided_json`, xgrammar backend); TRT-LLM has a logits-processor path. The mechanism is universal; the maturity differs.

---

## Operating it

Flags you'll actually set (confirm against `python -m sglang.launch_server --help` for your version):

| Flag | Equivalent to | Notes |
|---|---|---|
| `--mem-fraction-static` | vLLM `--gpu-memory-utilization` | Fraction reserved for weights + KV pool |
| `--chunked-prefill-size` | vLLM `--max-num-batched-tokens` | Prefill chunk budget; `-1` disables chunking |
| `--max-running-requests` | vLLM `--max-num-seqs` | Concurrency cap |
| `--max-total-tokens` | TGI `--max-batch-total-tokens` | KV pool in tokens |
| `--schedule-policy` | (no vLLM equivalent) | `lpm` / `fcfs` / `hrrn` / … — the cache-aware dial |
| `--schedule-conservativeness` | TGI `waiting_served_ratio` (in spirit) | Lower = admit more aggressively |
| `--disable-radix-cache` | vLLM `--no-enable-prefix-caching` | Use it once, to measure what the cache is worth |
| `--page-size` | vLLM `--block-size` | Tokens per page |
| `--grammar-backend` | vLLM `--structured-outputs-config` | `xgrammar` / `outlines` / `llguidance` |
| `--speculative-algorithm` | vLLM `--speculative-config` | EAGLE variants, n-gram |
| `--tp-size`, `--dp-size` | same | Parallelism ([Phase 6](../../ROADMAP.md#phase-6--distributed-inference-at-scale)) |

Serving is `python -m sglang.launch_server --model-path <model> --port 30000` with an OpenAI-compatible API, so your Phase-3 harness points at it unchanged — which is the whole reason lesson 10's shootout is possible.

**Where it wins, concretely:** workloads with large shared prefixes (a 2k-token system prompt across every request; agent loops that resend a growing transcript; few-shot classification with a fixed 20-example preamble; batch evaluation over one long document). On those, the published RadixAttention results are large, and you can reproduce the *shape* yourself: measure hit rate and TTFT with `--disable-radix-cache` vs default, with an lpm vs fcfs policy, at fixed offered load.

**Where it doesn't:** every prompt unique and short. Then the tree is overhead, and any engine's numbers converge to the same kernels.

---

## The frontend language (optional, but know it exists)

SGLang started as a *language*: a Python DSL (`@sgl.function`, `gen()`, `fork()`, `select()`) for multi-call LLM programs, where the runtime knows the program structure and can therefore share KV between branches, parallelize forks, and skip re-prefilling shared context between calls. The runtime (SRT) is what everyone deploys; the frontend is the part that explains the design.

The transferable idea: **when the server knows the *program*, not just the request, it can schedule better.** That's the same insight behind prefix-aware routing ([Phase 6](../../ROADMAP.md#phase-6--distributed-inference-at-scale)) and behind why an agent framework that reuses one conversation ID beats one that rebuilds prompts from scratch.

---

## Do this (90 minutes reading, plus one measurement if you have a GPU)

1. Read `radix_cache.py` end to end (~700 lines) with your Phase-4 `paged_kv.py` open. List three things the tree makes easy that your hash map made awkward, and one thing that's harder.
2. Read `schedule_policy.py::calc_priority` and `PrefillAdder`. Write down exactly how a cached prefix reduces a request's cost in the budget.
3. Find where `inc_lock_ref` is called in `scheduler.py`. Explain what breaks if it isn't.
4. **Measure (GPU):** the shared-prefix workload from Phase 4 lesson 6, at fixed offered load, four configurations: `{radix on, off} × {lpm, fcfs}`. Report hit rate, TTFT p50/p99, throughput. Predict the ordering of the four before running.
5. Compare with vLLM on the *same* workload and harness — that's a legitimate shootout axis, and it's the interesting one because it's a design difference rather than a defaults difference.

---

**Next:** [TensorRT-LLM and compiled engines →](06-tensorrt-llm-and-compiled-engines.md) — what you gain by moving work from run time to build time, and what it costs you operationally.
