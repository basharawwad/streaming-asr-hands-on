"""Chunked attention masks, and the latency they imply.

A streaming encoder's whole attention policy is one boolean matrix. Building it
by hand once is the fastest way to stop confusing chunk size with lookahead and
left context with history.

Convention throughout: `mask[i, j] == True` means query frame i is allowed to
attend to key frame j.
"""

from __future__ import annotations

import torch


def full_mask(num_frames: int, device=None) -> torch.Tensor:
    """Offline attention. Everything sees everything. Not causal."""
    return torch.ones(num_frames, num_frames, dtype=torch.bool, device=device)


def causal_mask(num_frames: int, device=None) -> torch.Tensor:
    """Strictly causal, one frame of algorithmic delay and no lookahead.

    Accurate but weak: the model never sees any future context at all, which
    costs more WER than a modest chunk would.
    """
    idx = torch.arange(num_frames, device=device)
    return idx.view(-1, 1) >= idx.view(1, -1)


def chunk_mask(
    num_frames: int,
    chunk_size: int,
    left_context_chunks: int | None = None,
    right_context_frames: int = 0,
    device=None,
) -> torch.Tensor:
    """The mask used by a cache-aware chunked-attention encoder.

    Frames are grouped into chunks of `chunk_size`. Every frame in chunk c may
    attend to:

      * all frames in its own chunk, including later ones, which is where the
        algorithmic delay comes from,
      * all frames in the previous `left_context_chunks` chunks, which is free
        in latency terms because they are already in the past,
      * `right_context_frames` frames past the end of its chunk, which is extra
        lookahead bought at extra delay.

    `left_context_chunks=None` means unlimited history, which is what you want
    for accuracy and can afford because the cost is memory rather than latency.
    """
    idx = torch.arange(num_frames, device=device)
    chunk_of = idx // chunk_size

    hi = (chunk_of + 1) * chunk_size - 1 + right_context_frames
    if left_context_chunks is None:
        lo = torch.zeros_like(idx)
    else:
        lo = ((chunk_of - left_context_chunks) * chunk_size).clamp_min(0)

    keys = idx.view(1, -1)
    return (keys >= lo.view(-1, 1)) & (keys <= hi.view(-1, 1))


def algorithmic_delay_frames(chunk_size: int, right_context_frames: int = 0) -> int:
    """Worst-case structural delay, in frames, for a chunked encoder.

    The first frame of a chunk is the unlucky one: it must wait for the rest of
    its own chunk to arrive, plus any right context. This is the number papers
    report as the model's latency, and it is only one of four contributions to
    what the user actually waits. See `latency.py`.
    """
    return chunk_size - 1 + right_context_frames


def algorithmic_delay_seconds(
    chunk_size: int, frame_ms: float, right_context_frames: int = 0
) -> float:
    """Same thing in seconds, given the frame rate after subsampling.

    A Conformer front end typically subsamples by 4, so 10 ms features become
    40 ms frames and a 14-frame chunk is 560 ms.
    """
    return algorithmic_delay_frames(chunk_size, right_context_frames) * frame_ms / 1000.0


def max_key_index(mask: torch.Tensor) -> torch.Tensor:
    """For each query, the furthest-future key it is allowed to see.

    Useful as an assertion in tests: a mask is causal with lookahead L when
    `max_key_index(mask) <= arange(T) + L` everywhere.
    """
    T = mask.shape[-1]
    idx = torch.arange(T, device=mask.device)
    return torch.where(mask, idx.view(1, -1), torch.full_like(idx.view(1, -1), -1)).max(
        dim=-1
    ).values
