"""The headline test: streaming a Conformer block equals running it offline."""

import torch

from streaming_asr.cache import LeftContextKVCache, kv_cache_bytes
from streaming_asr.conformer import RelativePositionBias, StreamingConformerBlock

torch.manual_seed(0)


def _block(**kw) -> StreamingConformerBlock:
    defaults = dict(
        d_model=32, num_heads=4, conv_kernel=5, ff_expansion=2,
        chunk_size=4, left_context_chunks=2,
    )
    defaults.update(kw)
    return StreamingConformerBlock(**defaults).eval()


def test_streaming_equals_offline():
    """Same weights, same input, same output, two completely different paths."""
    block = _block()
    x = torch.randn(2, 24, 32)

    with torch.no_grad():
        offline = block(x)
        streaming = block.stream(x)

    torch.testing.assert_close(offline, streaming, atol=1e-5, rtol=1e-4)


def test_streaming_equals_offline_with_a_partial_final_chunk():
    """The case that catches a cache trimmed by capacity instead of chunk index."""
    block = _block(chunk_size=5, left_context_chunks=1)
    x = torch.randn(1, 23, 32)          # 4 full chunks and a stub of 3

    with torch.no_grad():
        torch.testing.assert_close(
            block(x), block.stream(x), atol=1e-5, rtol=1e-4
        )


def test_streaming_equals_offline_with_unbounded_left_context():
    block = _block(left_context_chunks=10_000)
    x = torch.randn(1, 20, 32)
    with torch.no_grad():
        torch.testing.assert_close(block(x), block.stream(x), atol=1e-5, rtol=1e-4)


def test_block_output_does_not_depend_on_later_chunks():
    """Causality at the block level, not just inside the convolution."""
    block = _block(chunk_size=4, left_context_chunks=2)
    x = torch.randn(1, 24, 32)
    y = torch.cat([x[:, :12, :], torch.randn(1, 12, 32)], dim=1)

    with torch.no_grad():
        a, b = block(x), block(y)

    torch.testing.assert_close(a[:, :12], b[:, :12], atol=1e-5, rtol=1e-4)


def test_left_context_actually_changes_the_output():
    """Guards against a mask so permissive the test above passes for free."""
    x = torch.randn(1, 24, 32)
    narrow, wide = _block(left_context_chunks=0), _block(left_context_chunks=5)
    wide.load_state_dict(narrow.state_dict())

    with torch.no_grad():
        assert not torch.allclose(narrow(x), wide(x), atol=1e-4)


# --------------------------------------------------------------------------
# Why relative position encoding, specifically
# --------------------------------------------------------------------------


def test_relative_bias_is_shift_invariant():
    """The property that lets a chunk be scored the same wherever it starts.

    An absolute encoding would give the same acoustic frame a different score
    depending on where the window began, and the streaming and offline paths
    would not agree.
    """
    bias = RelativePositionBias(num_heads=4, max_distance=32)
    q = torch.arange(10, 14)
    k = torch.arange(4, 14)

    with torch.no_grad():
        here = bias(q, k)
        later = bias(q + 100, k + 100)

    torch.testing.assert_close(here, later)


def test_relative_bias_saturates_beyond_max_distance():
    """Distances past the table's edge share a bucket, which is how it extrapolates."""
    bias = RelativePositionBias(num_heads=2, max_distance=4)
    q = torch.tensor([20])
    with torch.no_grad():
        far = bias(q, torch.tensor([0]))      # distance 20, clamped to 4
        edge = bias(q, torch.tensor([16]))    # distance 4
    torch.testing.assert_close(far, edge)


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------


def test_cache_keeps_only_the_recent_window():
    cache = LeftContextKVCache(capacity=6)
    for _ in range(5):
        k = torch.randn(1, 2, 3, 4)
        cache.append(k, k)
    assert len(cache) == 6


def test_cache_window_contents_are_the_most_recent_frames():
    cache = LeftContextKVCache(capacity=4)
    frames = [torch.full((1, 1, 1, 2), float(i)) for i in range(6)]
    for f in frames:
        k, _ = cache.append(f, f)
    torch.testing.assert_close(k, torch.cat(frames[-4:], dim=-2))


def test_kv_cache_bytes_formula():
    """2 * layers * kv_heads * head_dim * seq * dtype_bytes."""
    args = dict(num_layers=32, head_dim=128, seq_len=1000, dtype_bytes=2)
    mha = kv_cache_bytes(num_kv_heads=32, **args)
    gqa8 = kv_cache_bytes(num_kv_heads=4, **args)

    assert mha == 2 * 32 * 32 * 128 * 1000 * 2
    assert mha == 8 * gqa8          # the whole argument for grouped-query attention


def test_encoder_cache_is_bounded_but_decoder_cache_is_not():
    """Why a streaming encoder serves thousands of streams and an LLM does not."""
    bounded = [
        kv_cache_bytes(16, 8, 64, seq_len=min(t, 140), dtype_bytes=2)
        for t in (100, 1_000, 100_000)
    ]
    assert bounded[0] < bounded[1] == bounded[2]

    growing = [
        kv_cache_bytes(16, 8, 64, seq_len=t, dtype_bytes=2)
        for t in (100, 1_000, 100_000)
    ]
    assert growing[0] < growing[1] < growing[2]
