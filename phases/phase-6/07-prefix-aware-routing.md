# 7 — Routing for Stateful Serving

> **You'll be able to say:** "An LLM replica with a prefix cache is *stateful*, so round-robin is wrong: with R replicas, random routing gives a multi-turn conversation roughly a `1/R` chance of landing on the replica that already holds its KV, and every miss re-prefills the entire conversation. Consistent hashing on a session or token-prefix key fixes locality but creates hot spots; the production answer is a two-level policy — hash to a small candidate set, then pick the least-loaded member — with bounded loads, spill, and a cache-hit-rate metric next to the p99 TTFT metric. I can build it and measure both sides of the trade."

Phase 4 built a prefix cache; Phase 6 can throw it away with one line of load-balancer config. This lesson is the fix, and it's the [Phase 6 exit artifact](../../projects/README.md).

---

## Why round-robin is a bug here

A stateless HTTP service has no memory, so any replica is equally good. An LLM replica has two kinds of state:

1. **KV blocks for in-flight sequences** — hard state. Routing a follow-up token elsewhere is impossible mid-request; this is why streaming requests are pinned.
2. **The prefix cache** — *soft* state: reusable KV blocks for prompts already seen ([Phase 4 lesson 6](../phase-4/06-prefix-caching-and-radix-attention.md)). This is the state routing decisions destroy.

```
  A 6-turn conversation, R = 4 replicas, random/round-robin routing

  turn 1  (300 tok)  → R2   miss, prefill 300
  turn 2  (700 tok)  → R4   MISS, prefill 700   (R2 had the first 300)
  turn 3  (1100 tok) → R1   MISS, prefill 1100
  turn 4  (1500 tok) → R4   partial hit (300+? no — R4 has turn 2's 700) → prefill 800
  turn 5  (1900 tok) → R3   MISS, prefill 1900
  turn 6  (2300 tok) → R2   partial, prefill ~2000
                            ───────────────────────
                            ~6,800 prompt tokens prefilled

  Same conversation, sticky to R2
  turn 1 → R2 prefill 300; turns 2-6 hit the cached prefix and prefill only the
  new user turn (~400 tok each)     ─────  ~2,300 prompt tokens prefilled  (3× less)
```

The generalization: a conversation's turn-`n` prompt contains all of turns `1…n-1`, so **the value of a cache hit grows with conversation length while the cost of a miss grows quadratically in total prefill work.** With random routing over `R` replicas, the chance the longest available prefix is on the replica you picked is ~`1/R`, so:

| What you lose to round-robin | Effect |
|---|---|
| TTFT | a hit prefills only the new tokens. Turn 6 of the example: ~400 tokens instead of 2,300 → TTFT falls roughly 5× (prefill is linear in tokens) |
| Prefill FLOPs / cost per token | strictly wasted compute, and it steals decode bandwidth ([lesson 6](06-disaggregated-prefill-decode.md)) |
| Effective capacity | the wasted prefill is capacity you paid for; at high shared-prefix ratios this is a 2-3× throughput difference |
| KV memory | `R` replicas each caching the *same* conversation instead of one — worse hit rate *and* worse memory efficiency |

**Shared system prompts make it worse and easier at the same time.** A 2,000-token system prompt shared by all traffic is cached everywhere after a few requests, so routing doesn't matter for *that* segment — it's the per-session tail that needs locality. Knowing which of the two shapes your traffic has decides your key.

---

## The policy ladder

```
  ①  ROUND ROBIN / LEAST-OUTSTANDING
     locality: none        balance: perfect        use: stateless, no prefix cache
  ②  SESSION AFFINITY  (hash conversation_id mod R)
     locality: great       balance: luck           breaks: any scale event reshuffles ALL keys
  ③  CONSISTENT HASHING  (ring + virtual nodes, or maglev)
     locality: great       balance: ~ok            scale event moves only ~1/R of keys
  ④  PREFIX HASHING  (hash the first N block-aligned tokens)
     locality: great for shared prefixes           hot spot: one popular prefix → one replica
  ⑤  CACHE/KV-AWARE ROUTING  (router knows what each replica holds)
     locality: best        needs: engine → router cache events, more moving parts
  ⑥  TWO-LEVEL: consistent hash → candidate set of k → least-loaded of k   ★ recommended
     locality: near-⑤      balance: bounded        this is "power of two choices" + locality
```

### Why ⑥ is the default answer

Pure locality and pure balance are in direct conflict, and you don't have to pick a corner:

```
  key = session_id (or hash of the first 512 block-aligned prompt tokens)
  candidates = consistent_hash_ring.lookup(key, k=2 or 3)     # stable under scaling
  target = argmin(candidates, by = queue_depth + λ · running_seqs)
  if load(target) > (1 + ε) · mean_load:  spill to global least-loaded   # bounded loads
```

- **Consistent hashing** keeps `k` stable across replica churn, so a scale-up invalidates ~`1/R` of the keys, not all of them.
- **Choosing among `k = 2`** collapses tail imbalance dramatically (the classic "power of two choices" result) while still hitting a warm cache in one of two places.
- **The `(1+ε)` cap** ("consistent hashing with bounded loads") makes the pathological case — one whale session, one viral prefix — degrade into extra prefill rather than a 30-second queue on one replica.

Pick `ε` from your SLO: small `ε` protects latency and sacrifices hit rate; large `ε` does the opposite. **This one knob is the whole lesson, and it should be a config value with a dashboard, not a constant in code.**

### Getting the key right

- **Block-align the prefix hash.** Engines cache in blocks of 16 tokens ([Phase 4 lesson 5](../phase-4/05-paged-attention.md)); hashing a prefix that doesn't end on a block boundary gives you a key whose granularity doesn't match the thing it's predicting. Hash the *tokenized*, block-truncated prefix — which means the router has to tokenize (cheap, but it's now on your TTFT path; cache the tokenizer's work per session).
- **Prefer an explicit session/conversation ID when the client can give you one** (`X-Session-Id`, or the OpenAI `user` field). It's exact, it's free, and it survives prompt edits. Fall back to prefix hashing for stateless one-shot traffic.
- **Never key on client IP.** Kubernetes `sessionAffinity: ClientIP` sits behind NAT and load balancers; you'll pin an entire office to one replica and nothing else.

---

## Cache-aware routing (⑤), and what it costs

The router keeps an approximate model of each replica's cache — either a radix tree of known prefixes per replica (SGLang's router) or a subscription to the engine's KV-cache events (vLLM publishes block hash add/remove events; llm-d's endpoint picker and Dynamo's KV router consume exactly this) — and scores each replica by *matched prefix length* and current load.

```
  score(replica) = matched_prefix_tokens × w_hit  −  queue_depth × w_load  −  ...
```

| Pros | Cons |
|---|---|
| Best possible hit rate; handles shared prefixes and sessions with one mechanism | The router's view is stale by construction — replicas evict without asking |
| Can route to *partial* matches (longest prefix wins) | Needs an event stream per replica; a new failure mode when it lags |
| Load and locality in one objective function | Router becomes stateful and memory-hungry at scale; needs its own eviction |

Use it when shared prefixes dominate and you already run one of the stacks that ships it. Otherwise ⑥ gets most of the benefit for a fraction of the operational surface, and ⑥ is what you can build and defend in an interview.

---

## Failure modes (each one has bitten a real deployment)

| Failure | Symptom | Fix |
|---|---|---|
| **Hot prefix** — every request shares one system prompt, keyed by prefix hash | one replica at 100%, others idle | key on session for the tail; the shared prefix will be cached on *all* replicas anyway (it's cheap and hot) |
| **Whale session** — one conversation at huge QPS | pinned replica saturates, p99 explodes | bounded loads (`1+ε`) + spill |
| **Scale event** — `hash mod R` after adding a replica | global cache miss storm, TTFT spike for minutes | consistent hashing (moves ~`1/R`), plus pre-warm the new replica before adding it to the ring ([lesson 8](08-autoscaling-gpu-fleets.md)) |
| **Queue-blind stickiness** | one replica queues while a warm-but-idle peer sits free | always include load in the decision; locality is worth ~a prefill, not ~a 5 s queue |
| **Stale router view** | router routes for hits that were evicted | treat hits as probabilistic; measure actual hit rate from the engine, not the router's guess |
| **Sticky to a dying replica** | a slice of sessions all fail together | health-check + eject from the ring; sessions rehash to the next node and re-prefill |
| **Prefix cache disabled on the engine** | routing work buys nothing | verify `--enable-prefix-caching` (on by default in current vLLM) and check the hit-rate metric is non-zero before tuning routing |

---

## Measuring it honestly

You need **four** numbers, and reporting fewer is how people convince themselves a router works:

1. **Prefix cache hit rate** — from the engine, not the router: vLLM exposes `vllm:gpu_prefix_cache_queries_total` and `vllm:gpu_prefix_cache_hits_total` (plus the derived hit-rate gauge). Aggregate across replicas.
2. **p50/p99 TTFT** — the user-visible payoff.
3. **Load imbalance** — `max_replica_load / mean_replica_load` over a window. This is what stickiness costs you, and it must be in the same table as the hit rate.
4. **Throughput at SLO (goodput)** — because a policy that improves hit rate while blowing p99 on one replica is a regression.

And the workload generator has to be *session-shaped*, or the experiment is meaningless: multi-turn conversations with a realistic turn count distribution, a Zipf-distributed set of system prompts, and a realistic think-time between turns. Extend the [Phase 3 lesson 7](../phase-3/07-measuring-honestly.md) harness — open-loop arrivals, sessions as first-class objects, and the same generator driving every policy so the comparison is leveled ([Phase 5 lesson 10](../phase-5/10-build-shootout-and-ensemble.md)'s discipline).

Expected shape of the result (predict before you measure): hit rate up several-fold, p99 TTFT down 2-5× on multi-turn traffic, imbalance up from ~1.05 to ~1.2-1.4, and *no* improvement at all on single-turn traffic with unique prompts — which is the control experiment that proves your harness isn't lying to you.

---

## What's available off the shelf

| Layer | What it does |
|---|---|
| NGINX `hash $http_x_session_id consistent;` | policy ③ in one line. Legitimate baseline; no load awareness, no cache awareness |
| Envoy ring-hash / maglev + `least_request` | ③/⑥-ish; maglev has better disruption properties on churn |
| Kubernetes `Service` + `sessionAffinity: ClientIP` | wrong granularity — don't |
| **Gateway API Inference Extension** (endpoint picker) / **llm-d** | ⑤ on Kubernetes: KV-aware endpoint selection driven by engine metrics/events |
| **SGLang router** | ⑤ with an approximate per-replica radix tree; cache-aware scheduling end to end |
| **NVIDIA Dynamo KV router** | ⑤ plus disaggregation-aware routing to prefill/decode pools ([lesson 6](06-disaggregated-prefill-decode.md)) |
| **vLLM production stack** router | ⑥/⑤ options over vLLM replicas, consuming its KV events |

Read one of them after you've built yours — the comparison is where the learning is, and "I wrote a sticky router, then read SGLang's and found three things it handles that I didn't" is the Phase 5 source-reading skill applied to your own code.

---

## Do this now (90 minutes, no GPU required)

1. **Build the ring.** ~80 lines: an `asyncio` reverse proxy with a consistent-hash ring (150-200 virtual nodes per replica), key = `X-Session-Id` else the block-aligned hash of the first 512 prompt tokens, `k = 2` candidates, pick by least outstanding requests, `(1+ε)` cap with spill, `/metrics` exposing per-replica routed counts and outstanding requests. Health-check and eject on failure.
2. **Prove the invariant without a GPU.** Replace the replicas with mock backends that simulate TTFT as `a + b × uncached_tokens` and hold a per-replica LRU set of prefix hashes. Run your session-shaped generator through all four policies (round-robin, ③, ⑥, and an oracle that always picks the true best match) and produce the four-metric table. The oracle row tells you how much headroom your policy left on the table.
3. **Then do it for real** ([project 12](../../projects/README.md)): 2+ vLLM replicas (one GPU each, small model, `--enable-prefix-caching`), scrape `vllm:gpu_prefix_cache_*` from each, and reproduce the table with real numbers. Add a scale event mid-test — add a third replica — and show the TTFT spike, its duration, and that only ~`1/R` of sessions were affected. That plot is the artifact.

---

**Next:** [Autoscaling and capacity for GPU fleets →](08-autoscaling-gpu-fleets.md) — the replica count your router balances over is itself a control problem, and model load times make the naive controller unstable.
