# 7 — Speculative Decoding

> **You'll be able to say:** "A cheap draft model proposes `k` tokens; the target model verifies all `k+1` positions in **one** forward pass; a rejection-sampling rule accepts a prefix and guarantees the output distribution is *identical* to the target's — I verified that empirically to a total-variation distance of 0.0017 against a very different draft. Expected speedup is `(Σᵢ₌₀ᵏ αⁱ) / (k·c + 1)`: at α = 0.8 and c = 0.15 the optimum is k ≈ 4-5 for **2.1×**; at α = 0.3 it's *unprofitable* once the draft costs more than 10% of the target. And it works because decode is memory-bound — it spends idle FLOPs. At high batch there are no idle FLOPs, so the win evaporates."

Every other technique in this phase reduces bytes. This one changes the *structure* of generation: from N sequential forward passes to roughly N/2-N/3 of them.

---

## The mechanism

```
  standard decode:  [target] → t1 → [target] → t2 → [target] → t3 → …     N passes for N tokens

  speculative:
    1. DRAFT   small model runs k times (cheap):        d1 d2 d3 d4
    2. VERIFY  target runs ONCE over the whole guess:   p(·|…), p(·|…d1), p(·|…d1d2), …
               ↑ one forward pass scores k+1 positions, because they're all in the prompt now
    3. ACCEPT  walk left to right; accept dᵢ with probability min(1, p(dᵢ)/q(dᵢ));
               on the first rejection, sample one token from the residual (p − q)⁺ and stop
    4. BONUS   if all k are accepted, the target's own next-token prediction is free → k+1 tokens
```

The trick that makes verification cheap is the same fact that makes prefill cheap: **a transformer scores every position of a sequence in one pass.** Verifying 5 guessed tokens costs about as much as generating 1, because decode is bandwidth-bound and you were reading all those weights anyway ([Phase 2 lesson 5](../phase-2/05-roofline-model.md)). Speculative decoding **converts spare FLOPs into fewer sequential steps.**

---

## It is exact — and you can verify that

The acceptance rule is not a heuristic; it's rejection sampling, and it makes the output distribution **provably identical** to sampling from the target model. Draft `x ~ q`; accept with probability `min(1, p(x)/q(x))`; otherwise sample from the normalized residual `max(p − q, 0)`. Empirically, 400,000 samples with a deliberately mismatched draft:

```
  target p     : 0.2677 0.2157 0.0664 0.0252 0.0982 0.0616 0.2307 0.0345
  draft  q     : 0.0828 0.1241 0.3007 0.0929 0.0290 0.2083 0.1394 0.0229
  spec-sampled : 0.2674 0.2166 0.0664 0.0258 0.0975 0.0617 0.2307 0.0339
  total variation distance from target: 0.00168  (0 = identical; sampling noise ~ 0.0016)
  theoretical acceptance rate alpha = sum_i min(p_i, q_i) = 0.551
```

The distance equals the Monte-Carlo noise floor: the distributions are the same. **Speculative decoding is lossless**, which is what separates it from every other technique in this phase and makes it the easiest optimization to get approved.

Two corollaries worth internalizing:

- **Acceptance rate has a closed form:** `α = Σᵢ min(p_i, q_i)` — the overlap between draft and target distributions, averaged over contexts. A better draft model literally means more distributional overlap.
- **Greedy decoding is the special case** where acceptance means "the draft's argmax equals the target's argmax." Same math, degenerate distributions.

```python
def spec_sample(p, q):                       # one speculative step, exactly
    x = sample(q)
    if random.random() < min(1.0, p[x] / q[x]):
        return x                              # accepted
    resid = [max(p[i] - q[i], 0.0) for i in range(len(p))]
    return sample([v / sum(resid) for v in resid])     # rejected: sample the residual
```

---

## The speedup formula (memorize this)

With acceptance rate `α` (per token, assumed independent), speculation length `k`, and cost ratio `c = draft_pass / target_pass`:

```
  E[tokens per round] = Σᵢ₌₀..ᵏ αⁱ = (1 − α^(k+1)) / (1 − α)      ← includes the bonus token
  cost per round      = k·c + 1                                    (k draft passes + 1 target pass)
  speedup             = E[tokens] / (k·c + 1)
```

Evaluated (c = 0.15, i.e. a draft ~7× cheaper than the target):

```
 alpha      k=1     k=2     k=3     k=4     k=5     k=7    k=10
   0.3     1.13    1.07    0.98    0.89    0.82    0.70    0.57
   0.5     1.30    1.35    1.29    1.21    1.12    0.97    0.80
   0.6     1.39    1.51    1.50    1.44    1.36    1.20    1.00
   0.7     1.48    1.68    1.75    1.73    1.68    1.53    1.31
   0.8     1.57    1.88    2.04    2.10    2.11    2.03    1.83
   0.9     1.65    2.08    2.37    2.56    2.68    2.78    2.74
  0.95     1.70    2.19    2.56    2.83    3.03    3.28    3.45
```

And the optimal `k` as the draft gets cheaper (α = 0.8):

```
  c=0.02  best k=11  speedup=3.82x
  c=0.05  best k= 8  speedup=3.09x
  c=0.1   best k= 6  speedup=2.47x
  c=0.2   best k= 4  speedup=1.87x
  c=0.3   best k= 3  speedup=1.55x
  c=0.5   best k= 2  speedup=1.22x
```

Everything you need to know about tuning is in those two tables:

1. **There is an optimal `k`, and it's small.** Speculating further is exponentially less likely to pay (`αᵏ`) while costing linearly (`k·c`). At α = 0.8, c = 0.15, k = 4-5 is optimal at 2.1× — and k = 10 is *worse* than k = 2.
2. **Low acceptance makes it harmful.** At α = 0.3, k = 4 the "optimization" is a **0.89× slowdown**, and it becomes unprofitable as soon as `c > 0.10`. Break-even thresholds measured: α = 0.3 → c ≤ 0.10; α = 0.5 → c ≤ 0.23; α = 0.7 → c ≤ 0.44.
3. **Draft cost and draft quality trade off.** A bigger draft raises α and c together. The product is what matters, which is why "use a 1B draft for a 70B target" (c ≈ 0.015, α ≈ 0.7-0.8) works so well and "use a 7B draft for a 13B target" doesn't.
4. **These are ceilings.** Real systems pay draft-model KV memory, an extra scheduler path, and per-round Python overhead. Expect to land below the formula, and report both.

---

## Variants: how to make the draft cheaper or better

| Variant | Draft comes from | α (typical) | Notes |
|---|---|---|---|
| **Vanilla (Leviathan/Chen)** | a separate small model of the same family | 0.6-0.8 | needs an aligned tokenizer + a checkpoint that exists |
| **Self-speculation / layer skip** | the target's own early layers | 0.5-0.7 | no extra model or memory |
| **Medusa** | extra decoding heads on the target predicting t+1, t+2, … | 0.6-0.8 | needs training the heads; tree-attention verification |
| **EAGLE / EAGLE-2/3** | a lightweight head over the target's *features*, autoregressive in feature space | 0.8+ | current quality leader; tree drafts |
| **Lookahead decoding** | Jacobi iteration + n-gram pool, no draft model | varies | training-free |
| **Prompt lookup / n-gram** | copy from the prompt itself | very high on summarization/RAG/code-edit | ~free; brilliant when output quotes input |
| **Tree/multi-candidate drafts** | several candidate continuations verified at once | raises effective α | more verification FLOPs per round |

**Prompt-lookup decoding deserves special attention** because it's nearly free to implement: when the task is summarization, RAG-grounded answering, or code editing, much of the output is copied verbatim from the input. Search the prompt for the last few generated tokens, propose the continuation that followed there, verify it. No draft model, no extra memory, and on copy-heavy workloads the acceptance rate is very high. Try this before anything with a second checkpoint.

---

## Why it dies at high batch (the part people miss)

Speculative decoding spends **extra FLOPs** to save **sequential steps**. That's a good trade only when FLOPs are free — i.e., when the GPU is memory-bound and mostly idle, which is exactly batch-1 to small-batch decode ([lesson 1](01-what-to-optimize.md): arithmetic intensity ≈ 1 against a ridge of ~300).

Now raise the batch size. Verification of `k+1` positions for `B` sequences is `B·(k+1)` tokens of work per round, so the batch's effective size is multiplied by `k+1`. Once that crosses the roofline ridge, you are compute-bound and the extra work costs real time:

```
  B = 1,  k = 4  → 5 token-positions per pass  → still bandwidth-bound → ~2x faster
  B = 64, k = 4  → 320 token-positions per pass → compute-bound → speculation is a TAX
```

Which gives the operating rule:

> **Speculative decoding is a latency optimization for low-concurrency serving** (single user, on-device, interactive agents, latency-SLO tiers). It is **not** a throughput optimization for a busy multi-tenant server — and vLLM/TGI expose exactly this by letting you disable speculation above a batch threshold.

Two more practical caveats:

- **Acceptance rate is workload-dependent**, and it drops on exactly the hard content you care about (novel reasoning, rare domains) while being high on boilerplate. Report α measured on *your* traffic; a vendor's α is a different distribution.
- **The draft model consumes memory and its own KV-cache**, competing with concurrency (lesson 4's arithmetic). Include that in the accounting.

---

## Try it

1. **Reproduce both tables.** Then plot speedup vs k for your (α, c) and mark the optimum. This is the tuning procedure — `num_speculative_tokens` is not a number you guess.
2. **Measure α for a real pair** (e.g. `gpt2` drafting for `gpt2-large`, or Llama-3.2-1B for Llama-3.1-8B): generate a few hundred tokens, count how often the draft's sample would be accepted. Plot α by content type — code vs prose vs math — and watch it move.
3. **Measure c honestly**: time a single draft forward pass and a single target forward pass at your batch size, warmed up. `c` is not the parameter ratio; it's the *time* ratio, and it's dominated by weight bytes (a 1B draft against a 7B target is c ≈ 0.14, not 0.14 of anything else).
4. **Implement prompt-lookup decoding** in your Phase-3 engine (search the prompt for the last 2-3 generated tokens, propose the next 5, verify). On a summarization workload, measure tokens/sec and acceptance. This is the highest ratio of speedup to code in the whole phase.
5. **Sweep batch size with speculation on and off** and find your crossover. Report the batch size where speculation stops paying — that number is the config you'd ship.
6. **Verify losslessness yourself** on a real model: generate with a fixed seed, greedy, with and without speculation; the token sequences must be *identical*. If they aren't, your acceptance rule is wrong.

---

## Key takeaways

- **Draft k tokens, verify all k+1 in one target pass, accept a prefix by rejection sampling.** Verification is cheap because decode is bandwidth-bound — you already paid for the weight reads.
- **It is exact.** Measured TV distance from the target distribution: **0.0017**, equal to the Monte-Carlo noise floor. Acceptance rate has a closed form: `α = Σ min(p_i, q_i)`.
- **`speedup = (Σᵢ₌₀ᵏ αⁱ) / (k·c + 1)`.** Memorize it; it answers every tuning question.
- **Optimal k is small and depends on c**: α = 0.8 gives k ≈ 4-5 at c = 0.15 (2.1×), k = 11 at c = 0.02 (3.8×), k = 2 at c = 0.5 (1.2×).
- **Low acceptance makes it a slowdown**: α = 0.3, k = 4 → 0.89×. Break-even needs `c ≲ 0.10` at α = 0.3, `c ≲ 0.44` at α = 0.7.
- **Variants raise α or lower c**: Medusa (heads), EAGLE (feature-space drafts, α ≈ 0.8+), self-speculation, lookahead, and **prompt-lookup** — which is free and excellent on copy-heavy tasks.
- **It fails at high batch**: verification multiplies effective batch by `k+1`, crossing the roofline ridge. Latency tool for low concurrency, not a throughput tool for a busy server.
- Report **measured α on your traffic**, measured `c`, the draft's memory cost, and the batch-size crossover — not the paper's numbers.

**Next:** [Compilation, kernels & graphs →](08-compilation-and-kernels.md) — the last layer: fuse the ops, delete the launch overhead, and make attention IO-aware.
