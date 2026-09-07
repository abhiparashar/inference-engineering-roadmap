# 2 — Static Batching (and Exactly Why It Hurts)

> **You'll be able to say:** "Static batching pads every prompt to the longest prompt and runs every sequence until the longest *output* finishes, so a realistic batch of 8 wastes 60-70% of its decode slots on padding. Worse, the batch is a closed door: requests that arrive one millisecond late wait for the whole thing to drain."

Static batching is what you get for free. `model.generate()` with a list of prompts is static batching; so is every "collect N items, call the model, return N results" worker ever written. It is also **exactly right** for a large class of models — and catastrophically wrong for autoregressive text generation. Knowing which is which, and being able to quantify the waste, is the point of this lesson.

---

## What it does, precisely

```
  1. collect B requests                     (fixed set — the batch is now closed)
  2. LEFT-PAD every prompt to max_prompt_len, build an attention mask
  3. prefill all B at once                  (one big GEMM: this part is genuinely great)
  4. loop: one decode step for all B rows   (still B×1 tokens, no matter who's finished)
     - a row that emitted EOS keeps being computed; its output is masked/discarded
  5. stop when ALL rows hit EOS or max_new_tokens
  6. return all B results together
```

Steps 2 and 4 contain the two independent sources of waste, and they're worth naming separately because they have different magnitudes and different fixes:

- **Prompt padding waste (step 2)** — you compute attention over pad tokens. Wasted *prefill* FLOPs.
- **Output-length waste (step 4)** — the batch runs for as many steps as the *longest* generation, so a 20-token answer occupies a slot for 500 steps. Wasted *decode* steps.

For chat traffic the second one dominates by an order of magnitude, because decode is ~97% of the wall clock ([lesson 1](01-what-a-serving-system-is.md)) and because output lengths vary far more wildly than prompts do.

*(Practical detail that bites everyone once: decoder-only models must be **left**-padded for generation — `tokenizer.padding_side = "left"` — so that the last real token is at the final position where the next-token logits are read. Right-padding silently produces garbage. Correct masks are non-negotiable too, or pad tokens leak into attention.)*

---

## Do the arithmetic: how much do you actually waste?

Eight requests in one static batch, with a length spread typical of real chat traffic (some "yes/no", some essays):

| Request | Prompt tokens | Output tokens |
|---|---|---|
| A | 50 | 20 |
| B | 120 | 30 |
| C | 90 | 50 |
| D | 400 | 80 |
| E | 210 | 120 |
| F | 150 | 200 |
| G | 60 | 350 |
| H | 900 | 500 |
| **sum** | **1,980** | **1,350** |
| **max** | **900** | **500** |

```
  prefill positions computed = B × max_prompt = 8 × 900 = 7,200
  prefill positions needed                     =        1,980
  prefill waste                = 1 − 1980/7200 =        72%

  decode slot-steps computed  = B × max_output = 8 × 500 = 4,000
  decode slot-steps needed                      =        1,350
  decode waste                = 1 − 1350/4000 =        66%   ← the expensive one
```

**Two-thirds of your decode compute produces tokens nobody asked for.** Put the other way: with perfect packing this GPU would serve ~3× the traffic on the same hardware. That factor of ~3 is the low end of what continuous batching recovers, and it's *before* considering latency.

The distribution matters more than the mean here. Because output length is heavy-tailed — a few very long generations among many short ones — waste grows with batch size for a fixed traffic mix: bigger B means a higher expected maximum, while the mean stays put. Formally, waste `= 1 − E[len] / E[max of B lens]`, and `E[max]` keeps climbing. **Static batching gets *worse* at exactly the batch sizes Phase 2 told you that you need** (B ≈ 150-300 for compute-bound decode). That is the trap.

---

## The latency problem is worse than the waste problem

Wasted FLOPs cost money. Head-of-line blocking costs users. Look at arrival times:

```
  t=0.00  A,B,C,D arrive  → batch closes, starts running
  t=0.01  E arrives       → the batch is CLOSED. E waits.
  t=0.02  F arrives       → waits.
  ...
  t=5.00  batch finishes (H needed 500 tokens)  → E has waited 4.99 s to even START
                                                   its own TTFT ≈ 5.05 s
```

Three named pathologies, all visible in that trace:

1. **Head-of-line blocking.** A request that misses the batch by 10 ms pays the *full* duration of the batch in queue wait. Nothing about E's own work is slow; it's stuck behind unrelated work.
2. **Convoy effect.** The single longest generation in the batch sets the release time for all eight *and* delays everyone queued behind them. One user asking for a 2,000-token essay degrades everybody. Your p99 becomes a function of other people's `max_new_tokens`.
3. **Zero admission flexibility.** Slots free up at step 21 (A finished) and stay empty for 479 steps. The scheduler *knows* there's queued work and *knows* there's free capacity, and can't act, because the batch is the scheduling unit.

Notice what these have in common: the batch boundary is a **commitment made once, for the entire lifetime of the longest request in it.** Requests have wildly different lifetimes (20 to 2,000 steps), so any decision made at request granularity is wrong for most of them almost immediately. Lesson 4 is what happens when you stop making it.

---

## When static batching is exactly right

Don't over-learn the lesson. Static batching's problems come *entirely* from autoregression — variable, unknown-in-advance numbers of steps per item. Remove that and it becomes optimal:

| Model type | One request = | Static batching? |
|---|---|---|
| Image classifier / detector | one forward pass, fixed shape | ✅ optimal — every item costs the same and finishes together |
| Text embedding / reranker | one forward pass (pad to max, cheap) | ✅ yes; bucket by length to cut pad waste |
| Recommender ranking (score 500 candidates) | one big fixed batch | ✅ yes — this *is* the natural batch |
| ASR / Whisper-style encoder-decoder | variable output length | ⚠️ partially — encoder batches fine, decode has the same problem |
| LLM chat/completion | 1 prefill + N unknown decode steps | ❌ no |

This is why NVIDIA Triton Inference Server's **dynamic batcher** (a queue + static batching, lesson 3) is the standard answer for vision and embedding models and is *not* how LLMs are served — Triton uses a separate in-flight batching path (TensorRT-LLM backend) for those. Roughly half of production inference by request count isn't autoregressive at all ([Phase 9](../../ROADMAP.md#phase-9--beyond-llms-recsysvisionspeech-hardware-diversity-edge-and-security)), and for that half this lesson's "bad" technique is best practice. Know which workload you're holding.

---

## The mitigations people try first (and their ceilings)

Before continuous batching existed, teams reached for these. They're worth knowing because you'll meet them in code, and because two of them are still genuinely useful:

| Mitigation | What it fixes | Ceiling |
|---|---|---|
| **Length bucketing** — separate queues per prompt-length range | prompt padding waste | does nothing for output-length waste; splits your traffic, so each bucket batches worse |
| **Sort queued requests by length** | pads less within a batch | needs a deep queue (adds latency), and reorders users → unfairness, starvation of long requests |
| **Cap `max_new_tokens` aggressively** | bounds the convoy | truncates real answers; a product regression, not an engineering fix |
| **Small batches (B=4-8)** | bounds head-of-line delay | throws away the throughput you batched for; you're back near the memory-bound floor |
| **Many model replicas, one per request** | isolates users | B=1 per replica → <1% GPU math utilization each ([Phase 2 lesson 5](../phase-2/05-roofline-model.md)); the most expensive possible answer |

Every row is a *tradeoff between throughput and tail latency* — you're sliding along one curve. The reason continuous batching was a genuine breakthrough rather than another tuning knob is that it **moves the curve**: it improves throughput *and* p50 latency at once (the Anyscale write-up's headline result), because it eliminates the waste instead of rebalancing who eats it.

---

## Try it (laptop; any small model)

Measure the waste directly. Predict the numbers from the table above first.

```python
import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "gpt2"                                    # 124M: runs on CPU/MPS fine
tok = AutoTokenizer.from_pretrained(name)
tok.pad_token, tok.padding_side = tok.eos_token, "left"     # ← required for decoder-only
model = AutoModelForCausalLM.from_pretrained(name).eval()

prompts = ["Hi", "Explain gravity in one sentence.",
           "Write a detailed essay about the history of the steam engine: " + "context " * 200,
           "Name a color"] * 2                   # deliberately ragged, like real traffic

enc = tok(prompts, return_tensors="pt", padding=True)
print("padded prompt matrix:", tuple(enc.input_ids.shape),
      "| real tokens:", int(enc.attention_mask.sum()),
      "| pad waste:", f"{1 - enc.attention_mask.sum().item()/enc.input_ids.numel():.0%}")

# --- static batch: all 8 together, everyone runs until the longest finishes
t0 = time.perf_counter()
with torch.no_grad():
    out = model.generate(**enc, max_new_tokens=200, do_sample=False,
                         pad_token_id=tok.eos_token_id)
static_s = time.perf_counter() - t0
new = (out.shape[1] - enc.input_ids.shape[1])

# --- what each request ALONE would have needed (the ideal, unbatched)
alone = []
with torch.no_grad():
    for p in prompts:
        e = tok(p, return_tensors="pt")
        t0 = time.perf_counter()
        o = model.generate(**e, max_new_tokens=200, do_sample=False,
                           pad_token_id=tok.eos_token_id)
        alone.append((time.perf_counter() - t0, o.shape[1] - e.input_ids.shape[1]))

print(f"static batch: {static_s:.2f}s for {len(prompts)} reqs, {new} steps for every row")
print(f"steps actually needed: {sum(n for _, n in alone)}  vs computed: {new*len(prompts)}"
      f"  → decode waste {1 - sum(n for _, n in alone)/(new*len(prompts)):.0%}")
print(f"sequential total: {sum(t for t, _ in alone):.2f}s   (batching still wins on throughput)")
```

Two results to sit with. **(1)** The batch is far faster than running the eight sequentially — batching works, that's the whole point, don't lose it. **(2)** Most of the decode steps it computed were padding, and the request that needed 8 tokens didn't return until the one needing 200 was done. You now have both halves of the tradeoff measured on your own machine, which is exactly the framing you need for the next two lessons: keep the throughput win, delete the waiting.

---

## Key takeaways

- **Static batching = a fixed set of requests, run together, released together.** The batch is the scheduling unit and the commitment lasts as long as the longest member.
- **Two wastes:** prompt padding (wasted prefill FLOPs) and output-length padding (wasted decode steps). For chat, the second dominates — **60-70% is normal**.
- Waste **grows with batch size** for heavy-tailed output lengths (`E[max]` rises, `E[len]` doesn't), so it fails worst at the large B you need for GPU efficiency.
- The latency damage — **head-of-line blocking, convoy effect, no admission into freed slots** — is worse than the compute waste, and shows up entirely in p99.
- Left-pad decoder-only models and pass a correct attention mask, or your outputs are wrong before they're slow.
- **It's the right technique for fixed-shape, single-pass models** (vision, embeddings, ranking) — roughly half of production inference. It's wrong only for autoregressive decode.
- Bucketing, sorting, and small batches only *slide along* the throughput/latency curve. Continuous batching moves it.

**Next:** [Dynamic batching →](03-dynamic-batching.md) — add a queue and a time window. It fixes arrival misalignment, has a crisp `λ·T ≈ B` condition for when it's worth anything, and still leaves the convoy problem untouched.
