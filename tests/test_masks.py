"""A streaming encoder's attention policy is one boolean matrix. Check it."""

import torch

from streaming_asr.masks import (
    algorithmic_delay_frames,
    algorithmic_delay_seconds,
    causal_mask,
    chunk_mask,
    full_mask,
    max_key_index,
)


def test_full_mask_is_not_causal():
    m = full_mask(6)
    assert m.all()
    assert bool((max_key_index(m) > torch.arange(6)).any())


def test_causal_mask_sees_nothing_ahead():
    m = causal_mask(8)
    torch.testing.assert_close(max_key_index(m), torch.arange(8))


def test_chunk_mask_lookahead_never_exceeds_the_chunk_boundary():
    """Every frame may see to the end of its own chunk, and no further."""
    T, C = 12, 4
    m = chunk_mask(T, chunk_size=C, left_context_chunks=None)
    idx = torch.arange(T)
    chunk_end = (idx // C + 1) * C - 1
    torch.testing.assert_close(max_key_index(m), chunk_end.clamp(max=T - 1))


def test_chunk_mask_right_context_extends_the_lookahead():
    T, C, R = 12, 4, 2
    m = chunk_mask(T, chunk_size=C, left_context_chunks=None, right_context_frames=R)
    idx = torch.arange(T)
    expected = ((idx // C + 1) * C - 1 + R).clamp(max=T - 1)
    torch.testing.assert_close(max_key_index(m), expected)


def test_left_context_bounds_history():
    """Finite left context is what makes per-stream memory constant."""
    T, C, L = 16, 4, 1
    m = chunk_mask(T, chunk_size=C, left_context_chunks=L)
    idx = torch.arange(T)
    lo = ((idx // C - L) * C).clamp_min(0)

    # No frame sees anything older than its window.
    first_visible = torch.where(
        m, idx.view(1, -1), torch.full_like(idx.view(1, -1), T)
    ).min(dim=-1).values
    torch.testing.assert_close(first_visible, lo)


def test_unlimited_left_context_reaches_frame_zero():
    m = chunk_mask(16, chunk_size=4, left_context_chunks=None)
    assert bool(m[:, 0].all())


def test_algorithmic_delay_matches_the_mask():
    """The delay figure a paper reports is a property of the mask.

    The unlucky frame is the first of a chunk: it waits for the rest of its own
    chunk plus any right context.
    """
    C, R = 8, 3
    T = 32
    m = chunk_mask(T, chunk_size=C, left_context_chunks=None, right_context_frames=R)
    worst = int((max_key_index(m) - torch.arange(T)).max().item())
    assert worst == algorithmic_delay_frames(C, R)


def test_algorithmic_delay_seconds_uses_the_post_subsampling_rate():
    """A 14-frame chunk at 40 ms per frame is 520 ms of structural delay."""
    assert abs(algorithmic_delay_seconds(14, frame_ms=40.0) - 0.52) < 1e-9
    # And with one chunk of right context on top:
    assert abs(algorithmic_delay_seconds(14, 40.0, right_context_frames=14) - 1.08) < 1e-9
