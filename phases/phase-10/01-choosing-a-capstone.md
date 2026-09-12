# 1 — Choosing and Scoping a Capstone

> **You'll be able to say:** "I pick a capstone by what it proves to the specific team I want to join, not by what looks impressive, and I scope it so that a narrow version is finished and measured within six weeks. I know the four anti-patterns — the unmeasured demo, the framework wrapper, the infinite-scope rewrite, and the benchmark that flatters itself — and I check every candidate project against them before writing code."

Nine phases of mechanics are behind you. The remaining risk is not technical: it is choosing badly and spending six weeks producing something that proves nothing.

---

## What a capstone has to do

A hiring manager spends five to fifteen minutes on your repository. In that window the project must answer three questions:

```
  1. CAN THEY BUILD?    does a non-trivial system exist and run?
  2. CAN THEY MEASURE?  are there numbers, a method, and a baseline?
  3. CAN THEY REASON?   do they explain the gap, the tradeoffs, and what
                        they'd do next with a specific, informed answer?
```

Question 3 is the discriminator. Most candidate projects answer 1, some answer 2, very few answer 3 — and question 3 is the one that maps onto the actual job, where nobody asks you to write a serving engine but everybody asks you why the p99 moved.

**Corollary:** a capstone that is technically smaller but answers all three beats a sprawling one that answers only the first. This is why lesson 2's bar matters more than the project choice.

---

## Selection rule: pick for the job, and pick for contrast

| Target role | Primary capstone | Secondary | Why |
|---|---|---|---|
| Inference/serving engineer (vLLM/TGI-shaped shops) | nano-vLLM ([L3](03-capstone-nano-vllm.md)) | cost study ([L5](05-capstone-cost-optimization.md)) | depth in the engine, then the economics conversation |
| Inference cost optimization | cost study ([L5](05-capstone-cost-optimization.md)) | paper repro of quantization/speculation ([L6](06-capstone-paper-reproduction.md)) | this *is* the job description |
| ML platform / MLOps | multi-modal pipeline ([L4](04-capstone-multimodal-pipeline.md)) | production RAG ([L7](07-capstone-production-rag.md)) | composition, observability, delivery |
| AI product backend / applied AI | production RAG ([L7](07-capstone-production-rag.md)) | multi-modal pipeline ([L4](04-capstone-multimodal-pipeline.md)) | the shape of most real work today |
| Research engineer / MLSys | paper repro ([L6](06-capstone-paper-reproduction.md)) | nano-vLLM ([L3](03-capstone-nano-vllm.md)) | reading and implementing primary sources |
| Recsys/ads inference | cost study on a ranking model ([L5](05-capstone-cost-optimization.md)) | multi-modal pipeline ([L4](04-capstone-multimodal-pipeline.md)) | latency-budget and fan-out engineering ([Phase 9 L2](../phase-9/02-recommendation-and-ranking.md)) |

Then apply the contrast rule: **one project that proves you understand the machine, one that proves you can ship.** Depth-only candidates get asked "have you ever run this in production?" Shipping-only candidates get asked "do you know what's inside?" Two projects, one of each, closes both questions in advance.

A third consideration that is worth more than people expect: **pick something you will still care about in week five.** Every capstone has a boring middle — the week of fixing a tokenizer mismatch or chasing a memory leak. Interest is the resource that gets you through it.

---

## Scoping so it finishes

The method is to fix the *proof* and let everything else shrink.

```
  1. Write the one-sentence claim you intend to prove.
       "A 1.5k-line engine with continuous batching and block-based KV
        management reaches within 3× of vLLM's throughput at equal p99
        on a 1B model, and I can attribute the remaining gap."

  2. Write the numbers table that would prove it — column headers only.

  3. Write the smallest system that can fill that table.
       one model · one GPU · one precision · one workload shape ·
       one baseline · synthetic-but-realistic traffic

  4. Everything else is a "future work" bullet, decided NOW, in writing.

  5. Timebox: 4-6 weeks part-time. At the halfway point, if the table is
     not fillable, cut a column — never lower the measurement standard.
```

**Deliberate narrowness is a senior signal.** "Single GPU, 1B model, fp16, one traffic pattern, because that isolates the scheduler's effect" is a better sentence than a list of four half-supported configurations. State the restriction and its reason; reviewers read it as control, not as limitation.

### What to cut first, in order

1. **Model variety** — one model proves the mechanism; three prove you had time.
2. **Hardware variety** — one GPU (or one CPU) unless hardware comparison *is* the claim ([Phase 9 lesson 5](../phase-9/05-hardware-diversity.md)).
3. **Feature breadth** — LoRA, multi-modal inputs, tool calling: all optional unless central.
4. **Scale** — 10k documents instead of 10M, 4 replicas instead of 40. Scale rarely changes the mechanism you're demonstrating, and you can state the extrapolation.
5. **UI** — a `curl` example beats a React front end, always, for this audience.

What you may **never** cut: the baseline, the honest measurement method, the failure-mode tests, the cost statement, and the write-up. Those are lesson 2, and they are what makes the rest evidence.

---

## The four anti-patterns

### 1. The unmeasured demo

Symptoms: a README with architecture diagrams and no tables; "blazing fast"; a demo GIF; no baseline anywhere.

Why it fails: there is nothing to discuss. An interviewer cannot probe a claim that was never quantified, so the conversation defaults to "walk me through the code," which any tutorial follower can do.

Fix: one baseline, one metric, one measured comparison, before adding any feature.

### 2. The framework wrapper

Symptoms: the project is 300 lines of glue around vLLM or LangChain; the interesting behavior all lives in a dependency.

Why it fails: it demonstrates API literacy, which is assumed. Note that a *composition* project (L4, L7) is not a wrapper — its substance is the budgets, timeouts, degradation, observability and gates around the models, which you build. The failure mode is specifically a wrapper with no engineering of its own: no measurement, no budget, no failure handling.

Fix: either build the mechanism (L3, L6) or own the production engineering around it (L4, L5, L7) — and make the numbers table be about *your* contribution.

### 3. The infinite-scope rewrite

Symptoms: "I'm writing a serving framework"; six weeks in, the tokenizer works and nothing has been measured; a `TODO.md` with 40 items.

Why it fails: it never reaches the write-up, and an unfinished project with no numbers is indistinguishable from no project.

Fix: the five-step scoping method above, and a hard halfway checkpoint where cutting a column is the expected action.

### 4. The self-flattering benchmark

Symptoms: your system compared against a misconfigured baseline; throughput at batch 512 versus a baseline at batch 1; latency measured without the client's queueing time; a "2× faster than vLLM" claim with vLLM's defaults untouched and its warmup skipped.

Why it fails worst of all: reviewers in this field configure these systems for a living. An unfair comparison destroys trust in every other number in the repo, including the correct ones.

Fix: tune the baseline as hard as you tune yourself, state its configuration, and expect to lose. **Losing honestly and explaining the gap is the strongest result you can produce as a single engineer competing with a funded project.** [Phase 3 lesson 7](../phase-3/07-measuring-honestly.md) and lesson 2 here are the standard.

---

## A pre-flight checklist for your chosen project

Answer all seven in writing before you start. If any answer is vague, the project is not yet scoped.

- [ ] **The claim**: one sentence, falsifiable, with a number in it.
- [ ] **The baseline**: what you compare against, and how you will configure it fairly.
- [ ] **The table**: exact columns, and the units.
- [ ] **The method**: workload, warmup, run count, how you'll report variance ([`playbooks/benchmarking.md`](../../playbooks/benchmarking.md)).
- [ ] **The environment**: hardware, versions, cost per hour, and total expected spend.
- [ ] **The failure tests**: which three things you will break on purpose, and what "correct" looks like.
- [ ] **The out-of-scope list**: at least five items, written down now.

Then: create the repository, commit the README skeleton with the empty tables, and commit the benchmark harness *before* the system. The harness-first order is what makes lesson 2 achievable instead of aspirational.

---

## Do this now (45 minutes)

1. **Name your target role** in one line, and pick your two capstones from the selection table, one depth and one shipping.
2. **Write the one-sentence claim** for each, with a number in it. Rewrite until it is falsifiable.
3. **Fill the pre-flight checklist** for the first one. The out-of-scope list must have five items.
4. **Create the repo and commit the skeleton**: README with empty numbers tables, `bench/` with the harness stub, `RESULTS.md`, and an `ENVIRONMENT.md` recording hardware, versions and prices. That commit is the real start of the project.

---

**Next:** [Engineering standards →](02-engineering-standards.md) — the bar every capstone must clear, and the definition of "done" that makes the difference between a demo and evidence.
