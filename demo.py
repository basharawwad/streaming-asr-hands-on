"""Run the whole thing end to end and narrate what each step proves.

    python demo.py

Nothing here is trained on real audio. The point is the mechanics: that the
streaming and offline paths agree, that the naive versions do not, and that the
four latencies are four different numbers.
"""

from __future__ import annotations

import torch

from streaming_asr.cache import kv_cache_bytes
from streaming_asr.causality import CausalDepthwiseConv1d, CentredDepthwiseConv1d
from streaming_asr.conformer import StreamingConformerBlock
from streaming_asr.ctc import ctc_greedy_decode, ctc_greedy_decode_buggy
from streaming_asr.endpointing import EndpointConfig, Endpointer
from streaming_asr.latency import (
    awed,
    effective_latency,
    frames_to_seconds,
    user_perceived_turn_latency,
)
from streaming_asr.masks import algorithmic_delay_seconds, chunk_mask, max_key_index
from streaming_asr.transducer import (
    num_alignments,
    rnnt_loss,
    rnnt_loss_bruteforce,
)

FRAME_MS = 40.0          # 10 ms features, subsampled by 4, as in a Conformer front end


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def demo_causality() -> None:
    rule("1. Causality is a property you can measure")
    x = torch.randn(1, 4, 20)
    future = x.clone()
    future[..., 11:] = torch.randn_like(future[..., 11:])

    with torch.no_grad():
        centred = CentredDepthwiseConv1d(4, 5).eval()
        causal = CausalDepthwiseConv1d(4, 5).eval()
        c_shift = (centred(x)[..., 10] - centred(future)[..., 10]).abs().max()
        k_shift = (causal(x)[..., 10] - causal(future)[..., 10]).abs().max()

    print("  replaced every frame after t=10, then read the output at t=10")
    print(f"  centred conv (padding=K//2)  output moved by {c_shift:.6f}  <- reads the future")
    print(f"  causal conv  (left padded)   output moved by {k_shift:.6f}")


def demo_masks() -> None:
    rule("2. The attention policy is one boolean matrix")
    T, C, L = 16, 4, 1
    m = chunk_mask(T, chunk_size=C, left_context_chunks=L)
    lookahead = (max_key_index(m) - torch.arange(T)).tolist()

    print(f"  chunk={C} frames, left context={L} chunk")
    print(f"  per-frame lookahead: {lookahead}")
    print("  the first frame of each chunk waits longest, and that is the")
    print("  number a paper reports as the model's latency:")
    print(f"  algorithmic delay = {algorithmic_delay_seconds(14, FRAME_MS)*1000:.0f} ms "
          f"for a 14-frame chunk at {FRAME_MS:.0f} ms/frame")


def demo_conformer() -> None:
    rule("3. Streaming a Conformer block equals running it offline")
    torch.manual_seed(0)
    block = StreamingConformerBlock(
        d_model=32, num_heads=4, conv_kernel=5, ff_expansion=2,
        chunk_size=4, left_context_chunks=2,
    ).eval()
    x = torch.randn(1, 23, 32)          # a partial final chunk on purpose

    with torch.no_grad():
        offline = block(x)
        streaming = block.stream(x)

    print(f"  input {tuple(x.shape)}, chunk=4, left context=2 chunks")
    print(f"  max absolute difference: {(offline - streaming).abs().max():.2e}")
    print("  same weights, same input, two completely different code paths")


def demo_transducer() -> None:
    rule("4. The transducer loss sums over every alignment")
    T, U, V = 4, 3, 5
    g = torch.Generator().manual_seed(0)
    lp = torch.randn(T, U + 1, V, generator=g).log_softmax(dim=-1)
    targets = torch.tensor([1, 2, 1])

    fast = float(rnnt_loss(lp.unsqueeze(0), targets.unsqueeze(0)))
    slow = float(rnnt_loss_bruteforce(lp, targets))

    print(f"  T={T} frames, U={U} labels -> {num_alignments(T, U)} monotonic paths")
    print(f"  forward recursion      {fast:.6f}")
    print(f"  brute-force over paths {slow:.6f}")
    print("  the loss weights a path only by its probability, so an early")
    print("  emission and a late one score the same. That indifference is")
    print("  why transducers emit late, and what a delay penalty removes:")

    uniform = torch.full((1, 6, 2, 3), -1.0986).log_softmax(dim=-1)
    one = torch.tensor([[1]])
    print(f"    delay_penalty=0.0 -> loss {float(rnnt_loss(uniform, one)):.4f}")
    print(f"    delay_penalty=0.3 -> loss {float(rnnt_loss(uniform, one, delay_penalty=0.3)):.4f}")


def demo_ctc() -> None:
    rule("5. CTC decode, in the right order and the wrong one")
    blank, h, e, l, o = 0, 1, 2, 3, 4
    names = {blank: "_", h: "h", e: "e", l: "l", o: "o"}
    path = [h, e, l, blank, l, o]

    lp = torch.full((len(path), 5), -20.0)
    for t, i in enumerate(path):
        lp[t, i] = 0.0
    lp = lp.log_softmax(dim=-1)

    show = lambda ids: "".join(names[i] for i in ids)
    print(f"  frames           {show(path)}")
    print(f"  collapse, strip  {show(ctc_greedy_decode(lp))}")
    print(f"  strip, collapse  {show(ctc_greedy_decode_buggy(lp))}   <- loses a letter")
    print("  the blank exists to separate genuine repeats, so it has to")
    print("  survive until after the collapse")


def demo_emission_and_latency() -> None:
    rule("6. Four latencies, and the one papers report")

    # A worked example rather than a decode: an untrained model's emission
    # times carry no information. These are the frames a decoder committed each
    # word on, against the frames where each word's audio actually ended.
    words = ["book", "a", "table", "for", "Wednesday"]
    emitted_frames = [14, 19, 27, 31, 48]
    reference_frames = [12, 18, 25, 30, 39]

    emitted_s = frames_to_seconds(emitted_frames, FRAME_MS)
    reference_s = frames_to_seconds(reference_frames, FRAME_MS)
    per_word = [(e - r) * 1000 for e, r in zip(emitted_s, reference_s)]

    print("  word emission delay, measured from each word's acoustic endpoint:")
    for w, d in zip(words, per_word):
        print(f"    {w:<10} {d:5.0f} ms{'   <- the tail is what users notice' if d > 200 else ''}")
    print(f"  median AWED {awed(emitted_s, reference_s)*1000:.0f} ms, "
          f"worst {max(per_word):.0f} ms")
    print("  report the median and a high percentile; a mean hides both")

    algorithmic = algorithmic_delay_seconds(14, FRAME_MS)
    parts = user_perceived_turn_latency(
        algorithmic_delay_s=algorithmic,
        emission_delay_s=0.30,
        rtfx=6.0,
        endpoint_silence_s=EndpointConfig.preset("balanced").max_turn_silence_ms / 1000.0,
        network_s=0.04,
    )
    print("\n  a turn-final response, broken down:")
    for name in ("algorithmic", "emission", "compute", "endpointing", "network"):
        bar = "#" * max(1, round(parts[name] * 40))
        print(f"    {name:<12} {parts[name]*1000:7.0f} ms  {bar}")
    print(f"    {'total':<12} {parts['total']*1000:7.0f} ms")
    print(f"\n  the structural delay alone is {algorithmic*1000:.0f} ms, "
          f"{algorithmic/parts['total']*100:.0f}% of what the user waits")
    print(f"  effective latency with compute: {effective_latency(algorithmic, 6.0)*1000:.0f} ms")


def demo_endpointing() -> None:
    rule("7. Endpointing under an asymmetric loss")
    for name in ("min_latency", "balanced", "max_accuracy"):
        cfg = EndpointConfig.preset(name)
        results = {}
        for complete in (True, False):
            ep = Endpointer(cfg, frame_ms=10.0)
            for _ in range(20):
                ep.update(0.9)
            fired = None
            for i in range(400):
                if ep.update(0.0, complete):
                    fired = (i + 1) * 10
                    break
            results[complete] = fired
        print(f"  {name:<13} sounds finished: {results[True]:>5} ms   "
              f"sounds unfinished: {results[False]:>5} ms")
    print("  cutting a user off costs far more than dead air, which is why the")
    print("  semantic signal only gets consulted after the minimum silence")


def demo_cache() -> None:
    rule("8. Why the encoder cache is bounded and the LLM cache is not")
    cfg = dict(num_layers=16, num_kv_heads=8, head_dim=64, dtype_bytes=2)
    window = 140                                  # left context, in frames
    for seconds in (10, 60, 600):
        frames = int(seconds * 1000 / FRAME_MS)
        enc = kv_cache_bytes(seq_len=min(frames, window), **cfg)
        llm = kv_cache_bytes(seq_len=frames, **cfg)
        print(f"  {seconds:>4}s of audio   encoder {enc/1e6:7.2f} MB   "
              f"decoder-only {llm/1e6:7.2f} MB")
    mha = kv_cache_bytes(32, 32, 128, 4096)
    gqa = kv_cache_bytes(32, 4, 128, 4096)
    print(f"\n  MHA 32 kv heads {mha/1e9:.2f} GB   GQA 4 kv heads {gqa/1e9:.2f} GB   "
          f"({mha/gqa:.0f}x)")


if __name__ == "__main__":
    print("streaming-asr-by-hand")
    demo_causality()
    demo_masks()
    demo_conformer()
    demo_transducer()
    demo_ctc()
    demo_emission_and_latency()
    demo_endpointing()
    demo_cache()
    print("\nRun `pytest -q` for the 92 assertions behind all of this.\n")
