"""A Conformer block you can run offline or stream, with identical output.

The block is the standard four submodules, feed-forward, self-attention,
convolution, feed-forward, with the Macaron half-step residuals. Three things
make it streamable, and they are the three things to be able to list:

  1. attention is chunked with a bounded left context (`masks.py`, `cache.py`),
  2. the depthwise convolution is left-padded and carries a state
     (`causality.py`),
  3. normalisation is LayerNorm, which touches no other frame.

Relative position bias rather than absolute encoding is not a detail. The bias
depends only on the distance between two frames, so a chunk produces the same
scores wherever it happens to start. An absolute encoding would make the same
acoustic frame look different depending on where the window began, and the
streaming and offline paths would diverge.

`test_conformer.py` asserts the two paths agree to floating point tolerance.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .cache import LeftContextKVCache
from .causality import CausalDepthwiseConv1d
from .masks import chunk_mask


class RelativePositionBias(nn.Module):
    """Learned scalar bias per head, indexed by clamped relative distance.

    Distances are clamped into [-max_distance, max_distance], so the table is
    finite and the model extrapolates to sequences longer than it was trained
    on by reusing the end buckets.
    """

    def __init__(self, num_heads: int, max_distance: int = 64):
        super().__init__()
        self.max_distance = max_distance
        self.table = nn.Parameter(torch.zeros(2 * max_distance + 1, num_heads))
        nn.init.normal_(self.table, std=0.02)

    def forward(self, q_pos: torch.Tensor, k_pos: torch.Tensor) -> torch.Tensor:
        """q_pos: (Tq,), k_pos: (Tk,) global frame indices.

        returns (1, H, Tq, Tk)
        """
        rel = q_pos.view(-1, 1) - k_pos.view(1, -1)
        rel = rel.clamp(-self.max_distance, self.max_distance) + self.max_distance
        bias = self.table[rel]              # (Tq, Tk, H)
        return bias.permute(2, 0, 1).unsqueeze(0)


class ChunkedSelfAttention(nn.Module):
    """Multi-head self-attention with an explicit mask or an explicit cache."""

    def __init__(self, d_model: int, num_heads: int, max_rel_distance: int = 64):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rel_bias = RelativePositionBias(num_heads, max_rel_distance)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_pos: torch.Tensor,
        k_pos: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + self.rel_bias(q_pos, k_pos)
        if mask is not None:
            # Use the dtype's own minimum rather than -1e9, which overflows to
            # -inf in fp16 and produces NaN after the softmax.
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attn = scores.softmax(dim=-1)
        return attn @ v

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Offline path. x: (B, T, D), mask: (T, T) boolean, True means visible."""
        B, T, _ = x.shape
        h = self.norm(x)
        q, k, v = self._split(self.q_proj(h)), self._split(self.k_proj(h)), self._split(self.v_proj(h))
        pos = torch.arange(T, device=x.device)
        out = self._attend(q, k, v, pos, pos, mask.view(1, 1, T, T))
        out = out.transpose(1, 2).reshape(B, T, self.d_model)
        return self.out_proj(out)

    def step(
        self,
        x_chunk: torch.Tensor,
        kv: LeftContextKVCache,
        frames_before: int,
        keep_frames: int,
    ) -> torch.Tensor:
        """Streaming path for one chunk.

        `frames_before` is how many frames were consumed before this chunk, so
        that positions are global. `keep_frames` is the size of the visible
        window after this chunk is appended, derived from the chunk index so it
        matches the offline mask exactly.

        No mask is needed: everything left in the cache is by definition
        visible, and every query in the chunk sees the same window.
        """
        B, Tc, _ = x_chunk.shape
        h = self.norm(x_chunk)
        q = self._split(self.q_proj(h))
        k_new, v_new = self._split(self.k_proj(h)), self._split(self.v_proj(h))

        kv.append(k_new, v_new)
        kv.keep_last(keep_frames)
        k, v = kv.k, kv.v

        frames_seen = frames_before + Tc
        k_start = frames_seen - k.shape[-2]
        q_pos = torch.arange(frames_before, frames_seen, device=x_chunk.device)
        k_pos = torch.arange(k_start, frames_seen, device=x_chunk.device)

        out = self._attend(q, k, v, q_pos, k_pos, mask=None)
        out = out.transpose(1, 2).reshape(B, Tc, self.d_model)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Macaron feed-forward. The caller applies the half-step residual."""

    def __init__(self, d_model: int, expansion: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.SiLU(),
            nn.Linear(d_model * expansion, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x))


class StreamingConvModule(nn.Module):
    """The Conformer convolution module, made causal.

    Order follows the paper: pointwise expansion by 2, GLU, depthwise
    convolution, normalisation, Swish, pointwise back down. Two deviations from
    the original, both required for streaming: the depthwise convolution is
    left-padded, and BatchNorm is replaced by LayerNorm.
    """

    def __init__(self, d_model: int, kernel_size: int = 15):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.pointwise_in = nn.Conv1d(d_model, 2 * d_model, 1)
        self.depthwise = CausalDepthwiseConv1d(d_model, kernel_size)
        self.post_norm = nn.LayerNorm(d_model)
        self.activation = nn.SiLU()
        self.pointwise_out = nn.Conv1d(d_model, d_model, 1)

    def _head(self, x: torch.Tensor) -> torch.Tensor:
        """Everything before the depthwise convolution. Pointwise, so no state."""
        h = self.norm(x).transpose(1, 2)              # (B, D, T)
        h = self.pointwise_in(h)
        return nn.functional.glu(h, dim=1)            # (B, D, T)

    def _tail(self, h: torch.Tensor) -> torch.Tensor:
        """Everything after it. Also pointwise."""
        h = self.post_norm(h.transpose(1, 2)).transpose(1, 2)
        h = self.activation(h)
        return self.pointwise_out(h).transpose(1, 2)  # (B, T, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._tail(self.depthwise(self._head(x)))

    def init_state(self, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        return self.depthwise.init_state(batch_size, device=device, dtype=dtype)

    def step(self, x_chunk: torch.Tensor, state: torch.Tensor):
        h, new_state = self.depthwise.step(self._head(x_chunk), state)
        return self._tail(h), new_state


class StreamingConformerBlock(nn.Module):
    """One Conformer block with matching offline and streaming paths.

    Args:
        chunk_size: frames per chunk. Sets the algorithmic delay.
        left_context_chunks: how many previous chunks stay visible. Costs
            memory, not latency, which is why it is the first dial to turn when
            accuracy is short.
    """

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        conv_kernel: int = 9,
        ff_expansion: int = 4,
        chunk_size: int = 8,
        left_context_chunks: int = 2,
        max_rel_distance: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.left_context_chunks = left_context_chunks

        self.ff1 = FeedForward(d_model, ff_expansion)
        self.attn = ChunkedSelfAttention(d_model, num_heads, max_rel_distance)
        self.conv = StreamingConvModule(d_model, conv_kernel)
        self.ff2 = FeedForward(d_model, ff_expansion)
        self.final_norm = nn.LayerNorm(d_model)

    def default_mask(self, num_frames: int, device=None) -> torch.Tensor:
        return chunk_mask(
            num_frames,
            self.chunk_size,
            self.left_context_chunks,
            right_context_frames=0,
            device=device,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Offline path. x: (B, T, D)"""
        if mask is None:
            mask = self.default_mask(x.shape[1], x.device)
        x = x + 0.5 * self.ff1(x)
        x = x + self.attn(x, mask)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.final_norm(x)

    def init_state(self, batch_size: int, device=None, dtype=None) -> dict:
        return {
            "kv": LeftContextKVCache(capacity=None),
            "conv": self.conv.init_state(batch_size, device=device, dtype=dtype),
            "frames_seen": 0,
            "chunks_seen": 0,
        }

    def step(self, x_chunk: torch.Tensor, state: dict):
        """Streaming path for one chunk of at most `chunk_size` frames."""
        assert x_chunk.shape[1] <= self.chunk_size, "chunk longer than chunk_size"

        frames_before = state["frames_seen"]
        chunk_idx = state["chunks_seen"]
        frames_after = frames_before + x_chunk.shape[1]

        # Visible window, derived from the chunk index so that a short final
        # chunk sees exactly what the offline mask allows and no more.
        lo = max(0, (chunk_idx - self.left_context_chunks) * self.chunk_size)
        keep = frames_after - lo

        x = x_chunk + 0.5 * self.ff1(x_chunk)
        x = x + self.attn.step(x, state["kv"], frames_before, keep)
        conv_out, conv_state = self.conv.step(x, state["conv"])
        x = x + conv_out
        x = x + 0.5 * self.ff2(x)
        x = self.final_norm(x)

        new_state = {
            "kv": state["kv"],
            "conv": conv_state,
            "frames_seen": frames_after,
            "chunks_seen": chunk_idx + 1,
        }
        return x, new_state

    @torch.no_grad()
    def stream(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: run the whole utterance chunk by chunk and concatenate."""
        state = self.init_state(x.shape[0], device=x.device, dtype=x.dtype)
        outputs = []
        for start in range(0, x.shape[1], self.chunk_size):
            chunk = x[:, start : start + self.chunk_size, :]
            y, state = self.step(chunk, state)
            outputs.append(y)
        return torch.cat(outputs, dim=1)
