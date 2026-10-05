# CTC in one page

## The problem

You have T audio frames and U labels, with T much larger than U, and **no
alignment**. Nobody told you which frames belong to which letter. Supervised
training needs a per-frame target, and you do not have one.

## The trick

Add one symbol to the vocabulary, **blank** (`_`), and let the network emit one
symbol per frame. Then define a many-to-one collapse from frame-level symbol
sequences to label sequences:

```
  1. collapse runs of the same symbol
  2. remove the blanks
```

Any frame-level sequence that collapses to the target is a valid alignment. The
loss is the total probability of all of them, so the alignment never has to be
supplied. It is marginalised away.

```
 frames      1     2     3     4     5     6     7     8
 network     C     C     _     A     A     A     T     _
             └──┬──┘     │     └─────┬────┘     │     │
 collapse       C        _           A          T     _
 strip blanks   C                    A          T
 result                        "C A T"
```

All of these are valid alignments of `CAT` and the loss sums over every one:

```
 C C _ A A A T _        C _ A _ T _ _ _        _ _ C A A T T T
 C C C A A T T T        _ C C _ _ A T _        C A T _ _ _ _ _
```

## Why blank exists at all

Not to pad. Blank is the **separator that makes genuine repeats expressible**.

```
 target "hello" — the double L must survive the collapse

 net A    h  e  l  _  l  o  o      collapse: h e l _ l o o -> h e l _ l o
                                   strip:    h e l l o              ✓ "hello"

 net B    h  e  l  l  o  o  _      collapse: h e l o _
                                   strip:    h e l o                ✗ "helo"
```

To emit a real double letter the network **must** place a blank between the two.
This is also the reason the decode order matters: collapse first, then strip. Do
it the other way round and every genuine repeat is destroyed. That bug is
`ctc_greedy_decode_buggy` in this repo, and it passes a smoke test on most
utterances, which is why it survives code review.

## The trellis

Pad the target with blanks between every label and at both ends. For a target
of U labels that gives **S = 2U + 1** states.

```
 target  C A T          extended: _ C _ A _ T _        S = 7
```

Each frame moves up the extended sequence by 0, 1 or 2 states. Three
transitions, and the third is restricted:

```
        s            stay          advance        skip a blank
                   s -> s          s -> s+1        s -> s+2
                  (dwell on        (move to        (only when the two
                   this symbol)     the next)       labels either side
                                                    are DIFFERENT)
```

```
 ext                t=1    t=2    t=3    t=4    t=5    t=6
  s6  _              ·      ·      ·      ·      ·      ●  ◄ valid end
                                                  ╲     ▲
  s5  T              ·      ·      ·      ·      ●─────►●  ◄ valid end
                                           ╲      ▲
  s4  _              ·      ·      ·      ●      ·      ·
                                    ╲      ▲
  s3  A              ·      ·      ●─────►●      ·      ·
                             ╲      ▲
  s2  _              ·      ●      ·      ·      ·      ·
                      ╲      ▲
  s1  C              ●─────►●      ·      ·      ·      ·
                      ▲
  s0  _              ●      ·      ·      ·      ·      ·
       ▲
   valid start:                    ─────►  stay on this symbol
   s0 or s1                            ╲   advance, or skip a blank
```

Every monotonic staircase from `{s0, s1}` to `{s5, s6}` is one alignment. The
forward algorithm sums them all in O(T·S).

### Why the skip is restricted

```
 target  L L          extended: _ L _ L _        S = 5

  s4  _         ·      ·      ·      ●
  s3  L         ·      ·      ●      ●
  s2  _         ·      ●      ●      ·          s1 ─╳─► s3  FORBIDDEN
  s1  L         ●      ●      ·      ·          ext[1] == ext[3] == L
  s0  _         ●      ·      ·      ·
                t1     t2     t3     t4
```

Skipping `s1 -> s3` would jump over the blank at `s2`, and that blank is the
only thing distinguishing `LL` from `L`. So the path is forced through it. When
the neighbouring labels differ, as with `_ C _ A _`, the blank carries no
information and may be skipped.

## The forward recursion

```
 alpha[t][s] = ( alpha[t-1][s]                      stay
               + alpha[t-1][s-1]                    advance
               + alpha[t-1][s-2] if skip allowed )  skip
               * p(ext[s] | frame t)

 loss = -log( alpha[T][S-1] + alpha[T][S-2] )
```

In log space, with `logsumexp` in place of the sum. That is
`ctc_loss_forward` in `streaming_asr/ctc.py`, checked against
`torch.nn.functional.ctc_loss`.

## Four consequences worth knowing

**No internal language model.** The frame-level distribution factorises:

```
 p(alignment | x) = product over t of p(a_t | x)
```

Each frame's output is conditionally independent of the others given the audio.
Nothing in the model knows that `q` is usually followed by `u`. This is the
substantive difference from a transducer, whose prediction network conditions
on label history, and it is why CTC systems lean on an external LM or on
contextual biasing to get entities right.

**Alignments go peaky.** A trained CTC model spends most frames on blank and
fires a label for one or two. Timestamps are therefore precise, but the
posteriors are poorly calibrated for a commit decision. `peaky_fraction` in
this repo measures it.

**It can fail outright on short audio.** You need at least one frame per
extended state that the path cannot skip, so

```
 T  >=  U + (number of adjacent repeated labels)
```

Below that no valid path exists, the probability is zero and the loss is
infinite. That is what PyTorch's `zero_infinity` flag is for, and in practice
it means a mislabelled short clip can poison a batch.

**Decoding is cheap and parallel.** Best-path decoding is one `argmax` per
frame plus the collapse. Prefix beam search does better by summing the
probability of all alignments that share a prefix, which is where an external
LM gets fused in.

## CTC against RNN-T, in one line

Both marginalise over alignments. The difference is what an alignment *is*.

| | CTC | RNN-T |
|---|---|---|
| Lattice axes | frame × extended-label state | frame × label |
| Per frame | exactly one symbol, possibly blank | any number of labels, then advance |
| So T must be | at least U plus the repeats | at least 1 |
| Output dependence | conditionally independent | conditioned on label history |
| Internal LM | none | yes, the prediction network |

The "any number of labels per frame" property is why a transducer can emit a
whole word on one frame and CTC cannot, and it is also why the transducer
lattice is the cleaner object: it has no extended label sequence and no
skip-transition special case.

See `streaming_asr/ctc.py` and `tests/test_ctc.py` for the runnable version,
and `docs/how-transcriptions-are-used.md` for how a speech LLM avoids all of
this by having no time axis in its loss at all.
