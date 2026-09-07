# 1 — Tokens & Embeddings (Text → Numbers the Model Can Use)

> **You'll be able to say:** "The model never sees text. Text is chopped into tokens, each token becomes an ID, and each ID looks up a vector. The model only ever works with vectors of numbers."

---

## The problem: models do math, not language

A neural network multiplies matrices ([Phase 0 lesson 4](../phase-0/04-what-is-inference.md)). It has no idea what a letter is. So before anything, text must become numbers. This happens in two steps: **tokenization** (text → integer IDs) and **embedding** (integer IDs → vectors). After that, it's all math.

```
"Hello world"  ──tokenize──▶  [15496, 995]  ──embed──▶  [[0.1, -0.3, …],   ← vector for token 15496
                              (token IDs)                [0.7,  0.2, …]]   ← vector for token 995
                                                         (now the model can do matmuls)
```

---

## Step 1: Tokenization (text → token IDs)

A **token** is a chunk of text — often a word, a piece of a word ("sub-word"), or a single character. Modern LLMs use *sub-word* tokenization (the common scheme is called **BPE**, Byte-Pair Encoding). Why sub-words instead of whole words?

- **Whole-word** vocab would be enormous and still miss new/rare words ("antidisestablishmentarianism", typos, code, emoji).
- **Single-character** would make sequences painfully long (more tokens = more compute).
- **Sub-word** is the sweet spot: common words are one token, rare words split into a few reusable pieces. Nothing is ever "out of vocabulary."

Each unique token in the model's **vocabulary** has a fixed integer ID. GPT-2's vocabulary is 50,257 tokens. So "tokenize" = look up each chunk and emit its ID.

### See it

```python
# pip install tiktoken  (OpenAI's fast BPE tokenizer, what GPT-2/3/4 use)
import tiktoken
enc = tiktoken.get_encoding("gpt2")

ids = enc.encode("Hello world! Tokenization")
print(ids)                              # e.g. [15496, 995, 0, 29130, 1634]
print([enc.decode([i]) for i in ids])   # ['Hello', ' world', '!', ' Token', 'ization']
print("vocab size:", enc.n_vocab)        # 50257
```

Notice: `" world"` includes its leading space (spaces are part of tokens), and `"Tokenization"` split into `" Token"` + `"ization"` — a rare word made of common pieces. This is why "count the words" ≠ "count the tokens," and why you're billed per *token*, not per word.

### Why tokens matter to an inference engineer

- **Everything is measured in tokens:** context length limits, throughput (tokens/sec), cost ($/1M tokens), latency-per-token (TPOT). Tokens are the *unit* of this whole field.
- **More tokens = more compute and more KV-cache.** A prompt that tokenizes into 1,000 tokens costs more to process and caches more than one that tokenizes into 100. Tokenization efficiency is real money at scale.
- **Prompt length in tokens** decides prefill cost ([lesson 6](06-prefill-vs-decode.md)) and KV-cache size ([lesson 5](05-kv-cache.md)).

---

## Step 2: Embeddings (token IDs → vectors)

An integer ID like `15496` is just a name — it carries no meaning the math can use (ID 15496 isn't "bigger" or "closer to" 15497 in any meaningful way). So each ID is turned into a **vector**: a list of, say, 768 floating-point numbers that *does* carry meaning. Similar tokens end up with similar vectors.

How? A giant lookup table called the **embedding matrix**, of shape `(vocab_size, d_model)`:

- `vocab_size` = number of tokens (50,257 for GPT-2)
- `d_model` = the vector size / "width" of the model (768 for GPT-2 small; 4096 for a 7B model)

"Embedding a token" is literally **picking the row of this matrix at the token's ID.** No math — just a table lookup.

```python
import numpy as np
vocab_size, d_model = 50257, 768
embedding_matrix = np.random.randn(vocab_size, d_model).astype(np.float32)  # "trained" table

ids = np.array([15496, 995])          # our two tokens
vectors = embedding_matrix[ids]        # fancy indexing = row lookup
print(vectors.shape)                   # (2, 768)  → two token vectors
```

These vectors are *learned* during training so that meaning gets encoded geometrically (famous toy example: `king - man + woman ≈ queen`). For inference, you just look them up — the table is frozen ([Phase 0 lesson 4](../phase-0/04-what-is-inference.md)).

> **Memory note:** the embedding matrix is big — 50257 × 768 × 4 bytes ≈ 154 MB in FP32 for GPT-2 small. For big-vocab models it's a real chunk of the weights. Many models "tie" the input embedding and the output projection to the same matrix to save memory (you'll see this in the build, [lesson 9](09-build-gpt2-from-scratch.md)).

---

## Step 3: Positional information (where each token sits)

One more thing. Attention (next lesson) treats its inputs as a *set* — by itself it has no notion of order, so "dog bites man" and "man bites dog" would look identical. That's obviously wrong for language. So models add **positional information**: a signal encoding each token's position in the sequence.

- **GPT-2** adds a learned **position embedding**: another lookup table of shape `(max_positions, d_model)`; position 0 gets a vector, position 1 another, etc., and it's *added* to the token embedding.
- **Modern models** often use **RoPE** (Rotary Position Embedding), which rotates the Q/K vectors by an angle that depends on position — better at generalizing to longer sequences. You don't need the math now; just know position gets injected somehow.

```python
seq_len = 2
position_embedding = np.random.randn(1024, d_model).astype(np.float32)  # GPT-2: 1024 positions
positions = np.arange(seq_len)                    # [0, 1]
x = vectors + position_embedding[positions]        # token meaning + position
print(x.shape)                                     # (2, 768) — ready for the transformer
```

That final `x` — shape `(seq_len, d_model)` — is what actually enters the first transformer block. Everything from here is transformations of this matrix of vectors.

---

## The shape you'll carry through the whole phase

After tokenizing + embedding + adding position, the model's working data is a tensor of shape:

```
(batch, seq_len, d_model)
   │       │         │
   │       │         └── width of each token's vector (768 for GPT-2 small)
   │       └── how many tokens in the sequence
   └── how many independent sequences processed at once (1 for a single request)
```

Memorize this shape. Every tensor in Phase 1 is a variation of it. When you get lost in attention, come back to: *"it's still `(batch, seq_len, d_model)`, I'm just mixing information across the `seq_len` dimension."*

---

## Key takeaways

- The model never sees text — only **vectors of numbers**.
- **Tokenization** chops text into sub-word **tokens** (BPE) and maps each to an integer **ID** from a fixed **vocabulary**. Tokens are the unit of context limits, throughput, and cost.
- **Embedding** turns each ID into a `d_model`-sized vector via a frozen lookup table (`vocab_size × d_model`). Similar meanings → similar vectors.
- **Positional info** (learned position embeddings in GPT-2, RoPE in modern models) is added so the model knows token order.
- The tensor entering the transformer is **`(batch, seq_len, d_model)`** — the shape to keep in your head all phase.

**Next:** [The attention mechanism →](02-attention-mechanism.md) — the one operation that makes transformers work.
