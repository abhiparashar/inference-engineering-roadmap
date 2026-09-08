# 6 — Prefix Caching & RadixAttention

> **You'll be able to say:** "If two requests share a token prefix, its KV is identical — so compute it once. With content-addressed 16-token blocks and LRU eviction I measured **93% of prefill eliminated** on a shared-2k-system-prompt workload and **66.8%** on multi-turn chat, versus 0% on independent prompts. Cache size decides everything (11% hit rate at 2k tokens of cache, 91% at 128k), and so does routing: the same chat workload across 4 replicas gets **33.7% round-robin vs 65.0% with prefix-affinity routing**. SGLang's RadixAttention is this idea with a radix tree so that *partial* prefixes match, not just whole ones."

Every other technique in this phase makes work cheaper. This one **deletes the work**, which is why its ceiling is so much higher — and why it's the only technique in Phase 4 that dramatically improves TTFT.

---

## Why it is correct (and where it isn't)

A decoder-only transformer's KV for position `i` depends **only on tokens `0..i`** — not on what comes after, not on sampling parameters, not on the request that asked for it. So:

> Two requests with the same token prefix have **bit-identical** KV for that prefix. Recomputing it is pure waste.

Which means prefix caching is **exact**, not approximate. No quality tradeoff, unlike everything in lessons 2-4. The conditions that must hold, and each is a real bug someone has shipped:

- **Same tokens, from position 0.** A prefix match must start at the beginning; positions are baked into the KV via RoPE/positional encoding. "The same paragraph appearing in the middle of two prompts" is *not* a hit.
- **Same model, dtype, and adapter.** A LoRA adapter changes the KV; cache keys must include the adapter id. (This is a classic multi-tenant bug: correct-looking text, wrong model.)
- **Tokenization must match exactly** — the same string tokenized differently (leading space, chat-template whitespace) is a different prefix and a silent miss.
- **Only complete blocks are cacheable.** A 16-token block is hashed when it fills; the partial tail is recomputed.

Where the sharing comes from in practice, roughly in order of value:

| Source | Shared portion | Typical hit rate |
|---|---|---|
| System prompt / policy preamble | 500-4,000 tokens, on *every* request | very high |
| Few-shot examples | 1-10k tokens | very high |
| Multi-turn chat history | everything before the new user turn | high, grows with turn count |
| Agent loops (ReAct, tool calls) | whole conversation replayed each step | very high |
| RAG with a shared corpus chunk order | document prefixes | medium |
| Parallel sampling / beam search | the entire prompt | ~100% |
| Independent one-shot prompts | nothing | ~0% |

---

## Measured: hit rate is the whole ballgame

Simulation: content-addressed 16-token blocks (each block's hash chains the previous ones, so a hash identifies a *prefix*, not a fragment), LRU eviction, 4,000 blocks (~64k tokens) of cache.

```
workload           requests  prompt tokens  prefill computed  hit rate  prefill saved
independent            1200        719,616           719,616      0.0%           0.0%
shared_system          1200      2,640,640           185,088     93.0%          93.0%
chat                   1200      2,676,704           888,672     66.8%          66.8%
```

- **A 2,048-token system prompt shared across requests: 93% of all prefill tokens disappear.** TTFT for those requests collapses to the cost of the ~100 unique tokens plus a block-table lookup. If your product has a long system prompt — and every serious LLM product does — this is the single largest optimization available to you, and it costs nothing in quality.
- **Multi-turn chat: 66.8%**, because each turn re-sends the whole conversation and only the new turn is novel. The longer the conversation, the higher the hit rate — the workload that is most expensive without caching is the one that benefits most from it.
- **Independent prompts: 0%.** Prefix caching is a *workload* property. Measure your traffic before promising anything; a benchmark with random prompts will show you nothing, and a benchmark that reuses one prompt will show you a fantasy.

### Cache capacity decides your hit rate

```
cache capacity sweep on the chat workload (blocks of 16 tokens):
  capacity    128 blocks (     2k tokens): hit rate  11.4%
  capacity    512 blocks (     8k tokens): hit rate  28.3%
  capacity   2000 blocks (    32k tokens): hit rate  47.2%
  capacity   8000 blocks (   128k tokens): hit rate  91.1%
  capacity  40000 blocks (   640k tokens): hit rate  94.0%
```

Hit rate is **sharply nonlinear in cache size and then saturates**: 128k tokens of cache captures 91%, and 5× more capacity adds 3 points. That shape is the whole capacity-planning argument — and it's in direct competition with running sequences, since both live in the same block pool. The decision is quantitative: a block held for reuse is a block unavailable for concurrency. Measure the knee (here: ~128k tokens) and give the cache exactly that much.

### Routing decides it too

Four replicas, same chat traffic, cache split evenly:

```
  round_robin             hit rate  33.7%
  sticky_by_prompt_prefix hit rate  65.0%
```

**Prefix-affinity routing nearly doubles the hit rate** for free, by sending a conversation back to the replica that already holds its KV. This is why production gateways hash a prefix of the prompt to choose a replica, and it's the exact tension flagged in [Phase 3 lesson 5](../phase-3/05-queueing-theory.md): sticky routing gives up some queue-pooling benefit (shared queues beat private ones) to win cache hits. Which wins depends on your hit-rate curve versus your utilization — and now you can measure both sides.

---

## RadixAttention: matching *partial* prefixes

Hash-per-block gets you exact prefix matching in O(blocks). SGLang's **RadixAttention** generalizes it: keep all cached sequences in a **radix tree (compressed trie) over tokens**, where each edge holds a run of tokens and each node points at its KV blocks.

```
                     [ system prompt: 2048 tokens ]           ← shared by everyone
                        ├── "summarize this: " ──── [doc A tokens] ── (user 1 turns…)
                        ├── "summarize this: " is shared, so it splits here
                        │        └───────────────── [doc B tokens] ── (user 2 turns…)
                        └── "translate: " ───────── …
```

What the tree buys over a flat hash map:

- **Longest-prefix match in one traversal**, including matches that diverge mid-way — you reuse everything up to the divergence point automatically.
- **Automatic sharing of intermediate prefixes** (system prompt + instruction, but different documents) without enumerating them.
- **A natural eviction unit**: LRU on *leaves*, so a shared internal prefix is never evicted while a child still needs it (refcounts again — [lesson 5](05-paged-attention.md)).
- **Cache-aware scheduling**: SGLang orders the waiting queue by longest prefix match, so requests that share a prefix run together and the shared blocks stay hot. That is a scheduling policy driven by the cache — [Phase 3 lesson 6](../phase-3/06-scheduling-policies-and-admission-control.md)'s fifth axis, now concrete.

vLLM's automatic prefix caching (`--enable-prefix-caching`, on by default in recent versions) uses the chained block-hash approach; SGLang uses the radix tree. Both are exact; the tree wins on workloads with rich branching (agents, tool loops, many shared sub-prefixes), the hash map is simpler and cheaper to maintain.

---

## What it does to your metrics

- **TTFT falls by roughly the hit fraction of prefill**, which for a long shared prompt is dramatic: a 2,000-token prefill at ~50 µs/token is ~100 ms; a 93% hit turns that into ~7 ms plus lookup.
- **Throughput rises** because prefill FLOPs — the compute-bound half of your workload — largely vanish, freeing the GPU for decode. It also relieves the [prefill interference](../phase-3/04-continuous-batching.md) problem: cached prefills don't stall decoders.
- **Decode is unchanged.** Cached KV still has to be read every step; caching saves the *computation* of the cache, not its bandwidth.
- **Memory is consumed by the cache**, in direct competition with concurrency — the tradeoff above.

The metric to expose and alert on is **`prefix_cache_hit_rate`** (vLLM: `vllm:prefix_cache_hits_total` / `queries_total`). A drop usually means a client changed its prompt template — a one-character change in a system prompt invalidates *every* cached entry, and TTFT triples. That is a real incident pattern, and it is instantly diagnosable if you have the metric and instantly baffling if you don't.

---

## Try it

1. **Reproduce the three-workload table** with your own traffic's shape. If you have real prompt logs, replay them; hit rate is a property of your users, not of the algorithm.
2. **Sweep cache capacity** and find your knee. Then convert it: at 128 KB/token, 128k tokens of cache = 16 GB of HBM, which is *N* fewer concurrent sequences. State both sides and pick.
3. **Compare routing policies** on a replayed trace: round-robin, least-outstanding-requests, and prefix-affinity. Report hit rate *and* p99 TTFT — affinity can win on cache and lose on queueing, and you should be able to show which dominates at your load.
4. **Build the radix tree version** (insert, longest-prefix lookup, LRU on leaves with refcounts) and compare its hit rate to flat block hashing on an agent-style workload with branching. This is a satisfying afternoon and a strong portfolio addition.
5. **Break it on purpose:** change one character in the system prompt mid-run and watch the hit rate and TTFT. Then add the metric and alert you'd have wanted.
6. **Measure it for real:** `vllm serve --enable-prefix-caching` vs `--no-enable-prefix-caching` with the Phase 3 harness and a shared-system-prompt workload. Report TTFT p50/p99 and throughput for both.

---

## Key takeaways

- Prefix caching is **exact, not approximate**: identical token prefixes have identical KV. The only quality risk is a correctness bug (adapter/tokenizer/position mismatch), not degradation.
- Measured savings in prefill tokens: **93% (shared 2k system prompt), 66.8% (multi-turn chat), 0% (independent prompts)**. It is a workload property — measure your traffic.
- **Hit rate saturates in cache size**: 11.4% at 2k tokens of cache → 91.1% at 128k → 94.0% at 640k. Give the cache the knee, not more; every block held is a block not serving concurrency.
- **Routing is worth almost as much as the cache itself**: 33.7% (round-robin) vs 65.0% (prefix affinity) across 4 replicas — and it trades against queue pooling, so measure both.
- **RadixAttention** = radix tree over tokens: longest-prefix matching including mid-way divergence, natural refcounted eviction, and cache-aware scheduling that groups requests sharing a prefix.
- It is the **only technique in this phase that strongly improves TTFT**, and it also relieves prefill interference; decode bandwidth is unaffected.
- **Alert on `prefix_cache_hit_rate`.** A prompt-template change silently invalidates the cache and multiplies TTFT.

**Next:** [Speculative decoding →](07-speculative-decoding.md) — every technique so far moved fewer bytes. This one stops generating one token at a time.
