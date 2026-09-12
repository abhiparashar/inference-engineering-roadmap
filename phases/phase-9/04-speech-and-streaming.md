# 4 — Speech: Streaming ASR and TTS

> **You'll be able to say:** "Speech serving is judged against the clock of the audio itself. The primary metric is real-time factor — compute seconds per audio second — and the contract is chunk size, lookahead, partial versus final hypotheses, and endpointing, because output must start before the input has finished arriving. I can price a full voice-agent loop (ASR → LLM → TTS) end to end, name which stage owns each millisecond, and explain why the parallelism available is concurrent streams, not longer batches."

Speech is the archetype that breaks the request/response mental model entirely: the request has no defined end when it starts, the response begins before the request finishes, and both sides are on a wall clock set by human speech, not by your scheduler.

---

## The two metrics that replace QPS

```
  REAL-TIME FACTOR (RTF) = compute_time / audio_duration
     RTF 0.05 → 20× faster than real time (transcribe 1 h in 3 min)
     RTF 1.0  → exactly keeps up; ZERO headroom; any jitter = falling behind
     RTF > 1  → the backlog grows without bound. Not "slow" — broken.

  streams_per_GPU  ≈  1 / RTF_per_stream  × utilization_derate (0.5-0.7)
     RTF 0.05 ⇒ ~20 streams before the math breaks, call it 10-14 in practice

  FIRST-AUDIO / FIRST-PARTIAL LATENCY = perceptual latency
     ASR partial:  < 300 ms feels live, < 150 ms feels instant
     TTS first audio: < 200 ms or the conversation feels broken
```

RTF is the capacity-planning number; first-audio/first-partial is the user-experience number. **They trade against each other**, and every tuning decision below is somewhere on that curve. Offline (non-streaming) transcription cares only about RTF and cost; live captioning and voice agents care about both.

The key structural consequence: **a stream occupies a slot for its whole duration** whether or not it is speaking. Ten concurrent callers who are silent 60% of the time still hold ten sessions. Utilization comes from (a) batching *across* streams and (b) not running the model on silence at all (VAD, below).

---

## The streaming contract

If you take one artifact from this lesson, take this specification. Every streaming ASR system has these parameters, and most integration bugs are a disagreement about one of them.

| Parameter | Meaning | Typical | Effect if you change it |
|---|---|---|---|
| **Chunk size** | audio per inference step | 100-640 ms | ↑ = better RTF/accuracy, ↑ latency |
| **Lookahead (right context)** | future audio the model may peek at | 0-500 ms | ↑ = better accuracy, adds directly to latency |
| **Left context** | past audio/state retained | 5-30 s or cached states | ↑ = better accuracy, ↑ memory per stream |
| **Partial (interim) hypothesis** | revisable text, emitted every chunk | every chunk | consumers must handle *replacement*, not append |
| **Final (stable) hypothesis** | committed text, never revised | at endpoint or after N chunks | the only thing safe to act on |
| **Endpointing / VAD** | decision that the speaker stopped | 300-800 ms of silence | too short: interrupts; too long: dead air |
| **Max utterance length** | forced cut | 30-60 s | prevents unbounded state |
| **Sample rate / format** | 16 kHz mono PCM16 is the norm | 8/16 kHz | mismatch = garbage output, no error |

```
  time →
  audio  ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇  (arriving in real time)
  chunks [c1][c2][c3][c4][c5][c6][c7][c8]
  partials    "wha"  "what is" "what is the wether"  ← revisable, may change
  final                              ┌──────────────────────────┐
                                     │ "what is the weather"     │  ← at endpoint
                                     └──────────────────────────┘
  ▲                                  ▲
  first partial ≈ chunk + lookahead + compute
                                     endpoint decision adds 300-800 ms
```

**The two mistakes clients make**, both worth writing into your API docs:

1. **Treating partials as appendable.** Partial N+1 replaces partial N; "what is the wether" becomes "what is the weather". A UI that appends produces gibberish; a downstream LLM triggered on partials acts on text that no longer exists.
2. **Acting on text before the endpoint.** Sending a partial to an LLM to save latency is a legitimate *speculative* optimization, but you must be prepared to cancel and redo — treat it exactly like speculative decoding ([Phase 4 lesson 7](../phase-4/07-speculative-decoding.md)): cheap when right, wasted work when wrong, and it needs an explicit cancel path.

### Streaming vs offline models are different models

Non-streaming architectures (encoder-decoder like Whisper, full bidirectional attention) see the whole utterance and are more accurate; streaming architectures (RNN-T / transducer, chunked-attention conformers, CTC with limited context) are causal or nearly so. You cannot make a full-attention model stream by cutting the audio into pieces and calling it repeatedly — that discards cross-chunk context and produces word errors at every boundary. What you *can* do:

| Approach | How | Cost |
|---|---|---|
| **Native streaming model** (RNN-T, chunked conformer) | causal/limited right context, state carried across chunks | 1-3% relative WER worse than offline |
| **Chunked offline model with overlap** | 30 s windows with 2-5 s overlap, stitch by alignment | works for *near*-live (seconds late), not for conversation |
| **Two-pass** | streaming model for partials, offline model rescores for the final | best quality/latency point; 2 models to serve |
| **Attention-state caching** | keep encoder KV/conv states per stream, feed only new frames | this is the prefix-cache analogue; essential for efficiency |

**Attention/conv state caching is the direct analogue of the KV cache** ([Phase 1 lesson 5](../phase-1/05-kv-cache.md)): without it, each chunk recomputes the whole left context and your RTF grows with utterance length. With it, per-chunk compute is constant — and you now have per-stream state that pins a session to a replica (affinity, [Phase 6 lesson 7](../phase-6/07-prefix-aware-routing.md)) and must be drained on shutdown ([Phase 8 lesson 4](../phase-8/04-kubernetes-for-gpu-serving.md)).

---

## Batching across streams

Your batch dimension is **concurrent sessions**, filled on a chunk-arrival deadline:

```
  every CHUNK_PERIOD (e.g. 160 ms):
      collect the chunks that arrived from all active streams
      pad/bucket them to a common frame count
      one batched forward pass (encoder step), per-stream state in/out
      emit partials
```

This is dynamic batching with a *fixed* tick rather than a max-delay timer, and it has properties worth knowing:

- **The batch is bounded by concurrency, not by throughput ambition.** Twelve active streams means batch 12, however powerful the GPU. Low-concurrency services therefore run at terrible utilization — which is precisely when CPU serving or a smaller model wins on cost ([lesson 5](05-hardware-diversity.md)).
- **Per-stream state makes the batch ragged.** Streams start and end constantly, so state tensors must be gathered/scattered per step. This is the same bookkeeping as continuous batching ([Phase 3 lesson 4](../phase-3/04-continuous-batching.md)) with a fixed step period, and it is where most home-grown streaming servers get slow or subtly wrong.
- **One slow step delays every stream in the batch.** Head-of-line blocking is shared across tenants; a pathological 60 s utterance in the batch pushes everyone's partials late. Cap utterance length and bucket by expected cost.
- **Silence is free if you skip it.** A VAD gate in front of the model typically removes 40-70% of audio in conversational traffic. It is the highest-leverage optimization in speech serving and it costs almost nothing (energy/GMM VAD is microseconds; a small neural VAD is ~1% of the ASR cost).

---

## TTS: the mirror image

Text-to-speech inverts the shape — short input, long output, and the consumer plays it back at a fixed rate.

```
  text → [1 acoustic/LM stage] → [2 vocoder] → audio frames → player buffer
         autoregressive or NAR      HiFi-GAN/          plays at 1× real time
         (often a transformer)      WaveRNN/codec
```

| Property | Consequence |
|---|---|
| Output is consumed at 1× real time | you only need RTF < 1 *sustained*; being 20× faster buys nothing for a single stream except headroom |
| First-audio latency is the UX metric | < 200 ms; so synthesize the **first sentence/chunk** and stream it while producing the rest |
| Autoregressive acoustic models behave like LLMs | continuous batching, KV cache, and speculative tricks all genuinely apply here — the one non-LLM archetype where Phase 3-4 transfers directly |
| Underrun is the failure mode | a gap in audio is far worse than a slightly later start; keep a jitter buffer of 200-500 ms and prioritize *sustained* rate over burst |
| Chunk boundaries are audible | prosody/energy discontinuities; overlap-add or sentence-level chunking, never mid-word |
| Per-stream cost is predictable | duration ≈ characters × rate, so admission control can be exact, unlike LLM output length |

**Design rule:** for a live TTS stream, once you are meeting real time with the jitter buffer full, extra speed should be spent on *more streams*, not lower latency. The player cannot consume it.

---

## The voice-agent loop: where the budget actually goes

The capstone-shaped system (also [ROADMAP capstone 2](../../ROADMAP.md#phase-10--capstones-this-is-where-top-1-gets-proven)) is ASR → LLM → TTS in a duplex loop. Human turn-taking tolerance is roughly 500-800 ms of silence before the conversation feels broken, and every stage eats into it:

```
  user stops speaking
  ├─ 400 ms   endpoint detection (waiting to be SURE they stopped)   ← biggest single item
  ├─  60 ms   final ASR hypothesis (two-pass rescore, if used)
  ├─  40 ms   network + orchestration hops
  ├─ 250 ms   LLM TTFT (streaming; first token only)                 ← Phases 3-4 apply
  ├─ 120 ms   enough tokens for the first TTS chunk (a clause)
  ├─ 130 ms   TTS first audio
  ├─  50 ms   network + client jitter buffer
  └─ ≈ 1050 ms perceived response latency  — already over budget
```

The levers, in order of effect:

1. **Shrink endpointing, adaptively.** Semantic endpointing (is this utterance syntactically complete?) beats a fixed silence timer; 400 ms → 150 ms for clearly complete utterances is the single biggest win available.
2. **Speculate on partials.** Start the LLM on a stable partial; cancel and restart if the final hypothesis differs. Needs real cancellation plumbing, and you pay for the wasted prefill.
3. **Stream between every pair of stages.** No stage may wait for its predecessor to finish: TTS starts on the first clause, not the first paragraph. The pipeline's latency should be the *first-token* latency of each stage summed, not the total.
4. **Allow barge-in.** The user interrupting means you must cancel LLM generation and TTS playback immediately; without cancellation you keep billing for tokens nobody will hear.
5. **Co-locate stages.** Three network hops at 40 ms each is 12% of the budget; same-host or same-rack placement is free latency.

This budget table is the deliverable. Write it with *your* measured numbers and you can answer any "how would you build a voice assistant" question with arithmetic instead of adjectives.

---

## Quality metrics and how they fail quietly

| Metric | What it is | Gotcha |
|---|---|---|
| **WER** (word error rate) | (S+I+D)/N against a reference | text normalization dominates the number: "$5" vs "five dollars", punctuation, casing. Always report the normalizer |
| **Streaming WER** | WER of *finals* in the streaming configuration | always worse than offline WER; benchmarks that quote offline numbers for a streaming product are lying by omission |
| **Latency-to-final** | endpoint → committed text | the number users feel; usually unreported |
| **Deletion rate under noise** | model silently drops speech | a "great WER" model can be a disaster in noisy audio |
| **MOS / intelligibility** (TTS) | human-rated naturalness | expensive; proxy with objective metrics but don't trust them across architectures |
| **Underrun rate** (TTS) | fraction of streams with an audio gap | the real TTS SLI; not latency |

The silent-degradation story for speech: **a sample-rate or channel-layout mismatch.** Feed 8 kHz audio to a 16 kHz model, or interleaved stereo to a mono model, and you get fluent, confident, wrong transcripts with zero errors logged. Validate format at the edge and reject, don't resample silently — and if you do resample, log it as a first-class metric because it changes accuracy.

---

## Failure modes table

| Symptom | Cause | Fix |
|---|---|---|
| Backlog grows, latency climbs without bound | RTF ≥ 1 for the offered concurrency | shed new streams at admission; never accept a stream you cannot keep up with |
| Partials flicker/rewrite excessively | too little lookahead / no stabilization | emit partials only when stable for K chunks; add lookahead |
| Words lost at every ~30 s boundary | chunking an offline model without overlap/state | native streaming model, overlap, or cached states |
| First partial is late but RTF is fine | chunk + lookahead too large | shrink chunk; accept the WER cost, measure it |
| Audio gaps in TTS playback | jitter buffer too small / no sustained-rate guarantee | 200-500 ms buffer, prioritize sustained throughput |
| One stream degrades everyone | head-of-line blocking in the batched step | cap utterance length, bucket by cost, cap batch step time |
| Transcripts confidently wrong | sample-rate/format/channel mismatch | validate at edge; reject with a clear error |
| Cost per audio-hour much worse than expected | model running on silence | VAD gate before the model |
| Sessions killed on deploy | per-stream state with no drain | `preStop` + long grace, reject new streams, let actives finish ([Phase 8 lesson 4](../phase-8/04-kubernetes-for-gpu-serving.md)) |
| LLM billed for cancelled turns | no barge-in cancellation | propagate cancellation through all stages |

---

## What to read and run

- **Papers:** *Sequence Transduction with Recurrent Neural Networks* (Graves — RNN-T, the streaming workhorse); *Conformer* (Gulati et al.); *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper — the offline baseline everyone compares to).
- **Code:** `k2-fsa/sherpa` / `icefall` (production streaming RNN-T serving), `NVIDIA/NeMo` (streaming conformer + export paths), `ggerganov/whisper.cpp` (offline, CPU, quantized — pairs with [lesson 5](05-hardware-diversity.md)), `snakers4/silero-vad` (tiny VAD you should put in front of everything).
- **Run, on any laptop:** `whisper.cpp` with a quantized model over a 60 s clip; compute RTF. That single number, on your own hardware, anchors every capacity estimate in this lesson.

---

## Do this now (45 minutes)

1. **Measure RTF three ways** on the same audio: a tiny model, a small model, and a medium model (`whisper.cpp` or NeMo, CPU and GPU if available). Report RTF and WER-ish quality impressions. Then compute `streams_per_device ≈ 0.6 / RTF` for each.
2. **Write your streaming contract** as a table: chunk, lookahead, left context, partial policy, endpoint rule, max utterance, audio format. This is a real API design artifact; keep it for the capstone.
3. **Measure the VAD win.** Run 10 minutes of conversational audio through a VAD and report the fraction of frames that are speech. That percentage is the cost reduction available for free.
4. **Price a voice loop** with your own numbers: endpoint delay + ASR final + LLM TTFT (from your Phase 3/7 server) + TTS first audio + network. Compare to 800 ms and name the two stages you would attack first.

---

**Next:** [Hardware diversity: TPU, Inferentia, CPU →](05-hardware-diversity.md) — the same models on silicon that isn't an NVIDIA GPU, and how to decide by cost per unit of useful work instead of by default.
