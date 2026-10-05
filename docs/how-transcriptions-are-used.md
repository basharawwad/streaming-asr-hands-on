# How ASR transcriptions train a speech LLM

The question this answers: in a CTC or RNN-T system the transcription is
obviously the target — but a speech LLM has no time axis in its loss, so where
does the transcription go?

Answer: it goes in the **text** stream, as ordinary next-token targets. The
audio goes in as *embeddings that are never predicted*. There is no alignment
anywhere.

## The splice

Tokenise a prompt template with a run of placeholder tokens where the audio
belongs, embed the whole thing, then overwrite the placeholder rows with
projected audio frames.

```
 raw audio  ──► encoder ──► [T, D_enc] ──► stack k ──► [T/k, k·D_enc]
                 (frozen)                                   │
                                                            ▼
                                                        projector
                                                     (2-layer MLP, trained)
                                                            │
                                                            ▼
                                                        [T/k, D_lm]
                                                            │
 "<|audio|> x N  Transcribe.  ->  book a table"             │
        │                                                   │
        ▼                                                   │
   tokenizer ──► ids ──► embedding table ──► [S, D_lm] ◄────┘
                                                  overwrite the N
                                                  placeholder rows
                                                            │
                                                            ▼
                                                  LM forward on inputs_embeds
```

## The label tensor is where the transcription actually lives

```
 position     0      1      2      3      4      5      6      7      8
 input   [ <aud> <aud> <aud>  "Tra" "scribe"  ":"  "book"  "a"  "table" ]
 labels  [ -100  -100  -100   -100   -100   -100  "book"  "a"  "table" ]
            └──────────┬─────────┘   └────┬────┘   └────────┬────────┘
            audio rows: no loss      prompt: no loss    transcription:
            (nothing to predict)     (it is given)      THE ONLY TARGETS
```

`-100` is PyTorch's `ignore_index`. Every audio and prompt position
contributes exactly zero to the loss.

### The causal shift does the alignment work

Cross-entropy is computed between `logits[:-1]` and `labels[1:]`. So the
logits at the **last audio position** are scored against the **first
transcript token**. That single shift is the entire connection between the two
modalities:

```
 logits at position 5 ("<|audio|>" last row) ──scored against──► "book"
 logits at position 6 ("book")               ──scored against──► "a"
 logits at position 7 ("a")                  ──scored against──► "table"
```

The gradient of the loss on `"book"` flows back through the attention from
position 5 into every audio row, and from there through the projector into the
encoder. Nobody ever told the model which audio frame is "book". It is a
sequence-level loss, and the attention finds the correspondence itself.

That is why `test_the_audio_changes_the_text_loss_even_though_it_is_masked` in
`tests/test_continuous_adapter.py` is the load-bearing test of this module: if
perturbing the audio does not change the text loss, the splice is not wired up
and the projector has no gradient. (It is also why the toy LM in that module
needs real causal self-attention — without attention the audio rows are
unreachable from the text positions and the projector gradient is exactly
zero. That was a genuine bug in the first version of this repo.)

## Against the aligned losses

| | CTC | RNN-T | Speech LLM (continuous) |
|---|---|---|---|
| Loss is over | frame-level alignments | (t,u) lattice paths | text tokens only |
| Alignment | marginalised | marginalised | never represented |
| Target length vs input | T ≥ U + repeats | T ≥ 1 | unconstrained |
| What the transcription is | the label sequence to align to | the label sequence to align to | plain next-token targets |
| Timestamps | free, and precise | free | not available |
| Internal LM | none | prediction network | the whole LM |
| Streaming | natural | natural | needs extra machinery |
| Trainable part here | all of it | all of it | projector (often LoRA too) |

The trade is legible: the aligned losses give you timestamps, bounded latency
and a model that streams by construction. The speech LLM gives up all three
and gets a text decoder that already knows the language, the entities, the
formatting conventions and how to follow an instruction.

## Why frame stacking is not an optimisation detail

A 25 Hz encoder output over 30 seconds is 750 rows spliced into the prompt. At
k=5 that is 150. Since attention is quadratic and the KV cache is linear in
sequence length, the stacking factor sets the cost of every subsequent text
token too — the audio stays in the cache for the whole generation.

```
 encoder 25 Hz    k=1   25 tok/s    750 rows for 30 s
                  k=4  6.25 tok/s   188 rows
                  k=5     5 tok/s   150 rows
                  k=8   3.1 tok/s    94 rows
```

Stacking is a reshape, not a pooling: `[T, D] -> [T/k, k·D]`, so no
information is discarded. The projector's input dimension grows by k and it
learns which parts of the concatenated window matter. `FrameStacker` and
`tokens_per_second` in `streaming_asr/continuous_adapter.py`.

## The discrete alternative, in one block

```
 continuous:  audio ─► encoder ─► projector ─────────────► inputs_embeds
                                                           (no vocabulary)

 discrete:    audio ─► codec encoder ─► RVQ ─► token ids ─► embedding table
                                                           (shared vocabulary
                                                            with text)
```

Discrete tokens let one softmax generate both modalities, which is what makes
speech *output* straightforward, and they make audio cacheable and storable as
ids. The cost is quantisation loss and a token rate multiplied by the number
of codebooks, which is what the RQ-Transformer, delay patterns and
Thinker-Talker designs all exist to manage. `ToyVectorQuantiser` and
`residual_quantise` in the same module are there to make the residual
structure concrete, nothing more.

See `docs/ctc.md` for the aligned side of the comparison.
