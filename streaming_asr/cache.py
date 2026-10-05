"""Encoder-side key/value cache, and the arithmetic behind its size.

A cache-aware streaming encoder carries exactly three things across a chunk
boundary: the attention keys and values for its left context, the convolution
state (see `causality.py`), and any normalisation running statistics. That is
the whole of the model's memory.

The important property is that the left context is *bounded*, so memory per
stream is constant rather than growing with the length of the call. That is
what makes it cheap to serve thousands of concurrent streams, and it is the
main way an encoder cache differs from a decoder-only LLM's KV cache.
"""

from __future__ import annotations

import torch


class LeftContextKVCache:
    """A fixed-capacity rolling window of attention keys and values.

    Holds the most recent `capacity` frames. Older frames fall off the back,
    which is the behaviour a chunked mask with a finite left context specifies.
    """

    def __init__(self, capacity: int | None = None):
        assert capacity is None or capacity >= 0
        self.capacity = capacity
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    def __len__(self) -> int:
        return 0 if self.k is None else self.k.shape[-2]

    def append(
        self, k_new: torch.Tensor, v_new: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add a chunk of keys and values, then return the whole visible window.

        k_new, v_new: (B, H, T_chunk, D)
        returns (k, v) each (B, H, T_visible, D) with
        T_visible <= capacity.
        """
        if self.k is None:
            self.k, self.v = k_new, v_new
        else:
            self.k = torch.cat([self.k, k_new], dim=-2)
            self.v = torch.cat([self.v, v_new], dim=-2)

        if self.capacity is not None:
            self.keep_last(self.capacity)

        return self.k, self.v

    def keep_last(self, num_frames: int) -> None:
        """Drop everything but the most recent `num_frames`.

        The block calls this with a window derived from the chunk index rather
        than from a fixed capacity, so that a partial final chunk produces the
        same visible window as the offline mask does. Getting this wrong is the
        usual reason a streaming equality test fails only on the last chunk.
        """
        if self.k is None:
            return
        if self.k.shape[-2] > num_frames:
            self.k = self.k[..., -num_frames:, :]
            self.v = self.v[..., -num_frames:, :]

    def start_index(self, frames_seen: int) -> int:
        """Global index of the oldest frame still visible.

        `frames_seen` is the total number of frames pushed so far, including
        the chunk just appended. Needed to compute relative position offsets
        that agree with the offline path.
        """
        return max(0, frames_seen - len(self))

    def reset(self) -> None:
        self.k = None
        self.v = None


def kv_cache_bytes(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    dtype_bytes: int = 2,
) -> int:
    """Size of a transformer KV cache.

        2 * layers * kv_heads * head_dim * seq_len * dtype_bytes

    The leading 2 is keys and values. The only term you can cut without
    touching the model's width or depth is `num_kv_heads`: multi-head attention
    sets it equal to the number of query heads, grouped-query attention shares
    one KV head across a group of query heads, and multi-query attention
    collapses it to one. Eight-way GQA is an eightfold reduction, which is why
    essentially every deployed model uses it.

    The other lever is `seq_len`, and for a speech model that is set by the
    tokeniser frame rate rather than by the architecture. Halving the frame
    rate halves the cache.
    """
    return 2 * num_layers * num_kv_heads * head_dim * seq_len * dtype_bytes


def frames_per_second(frame_ms: float) -> float:
    """Frames per second of audio, the multiplier on every cache cost."""
    return 1000.0 / frame_ms
