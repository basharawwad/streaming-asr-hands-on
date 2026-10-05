"""Causal convolution and streaming-safe normalisation.

Two of the three changes that turn a Conformer into a streaming Conformer live
in this file. The third is the attention mask, in `masks.py`.

Every class here comes in two forms: the one that works in a streaming system
and the one that quietly does not. The tests perturb a future frame and assert
that the output at time t did or did not change, which is what "causal"
actually means.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Depthwise convolution
# --------------------------------------------------------------------------


class CentredDepthwiseConv1d(nn.Module):
    """The default you get if you are not thinking about streaming.

    Symmetric padding means the kernel centred on frame t reads (K-1)//2 frames
    into the future. This is the single most common way a Conformer stops being
    causal, because `padding=kernel_size // 2` looks like the obvious thing to
    write and is what most non-streaming reference code does.
    """

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        assert kernel_size % 2 == 1, "centred padding assumes an odd kernel"
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            channels, channels, kernel_size,
            padding=kernel_size // 2, groups=channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) -> (B, C, T)"""
        return self.conv(x)


class CausalDepthwiseConv1d(nn.Module):
    """Left-padded depthwise convolution with an explicit streaming state.

    The receptive field at frame t is [t - (K-1), t]. Nothing in the future.
    `step` consumes one chunk at a time and carries the last K-1 frames forward
    as state, which is the only thing the convolution needs to remember.
    """

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.left_pad = kernel_size - 1
        self.conv = nn.Conv1d(
            channels, channels, kernel_size, padding=0, groups=channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Offline path. x: (B, C, T) -> (B, C, T)"""
        return self.conv(F.pad(x, (self.left_pad, 0)))

    def init_state(self, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        """Zeros standing in for the frames before the utterance started."""
        return torch.zeros(
            batch_size, self.channels, self.left_pad, device=device, dtype=dtype
        )

    def step(
        self, x_chunk: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Streaming path.

        x_chunk: (B, C, T_chunk), state: (B, C, K-1)
        returns (y_chunk, new_state) with y_chunk the same shape as x_chunk.
        """
        padded = torch.cat([state, x_chunk], dim=-1)
        y = self.conv(padded)
        new_state = padded[..., -self.left_pad:] if self.left_pad > 0 else state
        return y, new_state


# --------------------------------------------------------------------------
# Normalisation over time
# --------------------------------------------------------------------------


def utterance_cmvn(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Classic per-utterance mean and variance normalisation. NOT causal.

    Statistics are computed over the whole utterance, so frame 0 is normalised
    using audio that has not arrived yet. This is the violation people forget,
    because it lives in the feature pipeline rather than in the model.

    x: (B, T, C) -> (B, T, C)
    """
    mean = x.mean(dim=1, keepdim=True)
    var = x.var(dim=1, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + eps)


class RunningCMVN(nn.Module):
    """Cumulative mean and variance normalisation. Causal by construction.

    Frame t is normalised with the statistics of frames 0..t only, so the
    offline and streaming paths agree exactly. The cost is that early frames
    are normalised with a poor estimate, which is why production systems often
    prefer a fixed global estimate collected offline from the training set.
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Offline path, vectorised over the cumulative statistics.

        x: (B, T, C) -> (B, T, C)
        """
        counts = torch.arange(
            1, x.shape[1] + 1, device=x.device, dtype=x.dtype
        ).view(1, -1, 1)
        csum = x.cumsum(dim=1)
        csumsq = (x * x).cumsum(dim=1)
        mean = csum / counts
        var = (csumsq / counts) - mean * mean
        return (x - mean) / torch.sqrt(var.clamp_min(0) + self.eps)

    def init_state(self, batch_size: int, channels: int, device=None, dtype=None):
        zeros = torch.zeros(batch_size, 1, channels, device=device, dtype=dtype)
        return {"n": 0, "sum": zeros.clone(), "sumsq": zeros.clone()}

    def step(self, x_chunk: torch.Tensor, state: dict):
        """Streaming path. x_chunk: (B, T_chunk, C)"""
        out = torch.empty_like(x_chunk)
        n, s, sq = state["n"], state["sum"], state["sumsq"]
        for i in range(x_chunk.shape[1]):
            frame = x_chunk[:, i : i + 1, :]
            n = n + 1
            s = s + frame
            sq = sq + frame * frame
            mean = s / n
            var = (sq / n) - mean * mean
            out[:, i : i + 1, :] = (frame - mean) / torch.sqrt(
                var.clamp_min(0) + self.eps
            )
        return out, {"n": n, "sum": s, "sumsq": sq}


def batchnorm_is_unsafe_note() -> str:
    """Why the Conformer convolution module swaps BatchNorm for LayerNorm.

    BatchNorm normalises across the batch and time axes together, so in
    training the statistics for frame t depend on every other frame in the
    utterance and on the other utterances in the batch. At inference it uses
    frozen running statistics and is technically causal, but the train and
    inference behaviour then differ in a way that hurts streaming models
    specifically, because the chunked inference distribution is not the
    distribution the running statistics were collected under.

    LayerNorm normalises each frame over its own channels. No time axis is
    involved, so there is nothing to leak and nothing to mismatch.
    """
    return batchnorm_is_unsafe_note.__doc__ or ""
