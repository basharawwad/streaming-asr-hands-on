"""Causality is testable: perturb a future frame and see whether t moved."""

import torch

from streaming_asr.causality import (
    CausalDepthwiseConv1d,
    CentredDepthwiseConv1d,
    RunningCMVN,
    utterance_cmvn,
)

torch.manual_seed(0)


def _perturb_future(x: torch.Tensor, t: int, dim: int = -1) -> torch.Tensor:
    """Return a copy of x with everything strictly after frame t replaced."""
    y = x.clone()
    if dim == -1:
        y[..., t + 1 :] = torch.randn_like(y[..., t + 1 :])
    else:
        y[:, t + 1 :, :] = torch.randn_like(y[:, t + 1 :, :])
    return y


# --------------------------------------------------------------------------
# Convolution
# --------------------------------------------------------------------------


def test_centred_conv_leaks_the_future():
    """The failure this repo exists to make visible.

    padding=kernel_size//2 is what non-streaming reference code writes, and it
    makes the output at frame t depend on frames after t.
    """
    conv = CentredDepthwiseConv1d(channels=4, kernel_size=5).eval()
    x = torch.randn(1, 4, 20)
    t = 10

    with torch.no_grad():
        a = conv(x)
        b = conv(_perturb_future(x, t))

    assert not torch.allclose(a[..., t], b[..., t], atol=1e-6), (
        "a centred convolution should read the future; if this passes, the "
        "test is wrong, not the code"
    )


def test_causal_conv_does_not_leak_the_future():
    conv = CausalDepthwiseConv1d(channels=4, kernel_size=5).eval()
    x = torch.randn(1, 4, 20)
    t = 10

    with torch.no_grad():
        a = conv(x)
        b = conv(_perturb_future(x, t))

    torch.testing.assert_close(a[..., : t + 1], b[..., : t + 1])


def test_causal_conv_streaming_equals_offline():
    """Chunk-by-chunk with a state must equal one pass over the utterance."""
    conv = CausalDepthwiseConv1d(channels=6, kernel_size=7).eval()
    x = torch.randn(2, 6, 37)          # deliberately not a multiple of the chunk

    with torch.no_grad():
        offline = conv(x)

        state = conv.init_state(2)
        pieces = []
        for start in range(0, x.shape[-1], 8):
            y, state = conv.step(x[..., start : start + 8], state)
            pieces.append(y)
        streaming = torch.cat(pieces, dim=-1)

    torch.testing.assert_close(offline, streaming)


def test_causal_conv_receptive_field_is_exactly_k_minus_1():
    """Frame t depends on t-(K-1)..t and nothing older."""
    k = 5
    conv = CausalDepthwiseConv1d(channels=2, kernel_size=k).eval()
    x = torch.randn(1, 2, 16)
    t = 12

    y = conv(x)
    perturbed = x.clone()
    perturbed[..., t - k] = torch.randn_like(perturbed[..., t - k])  # just outside
    with torch.no_grad():
        torch.testing.assert_close(y[..., t], conv(perturbed)[..., t])


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------


def test_utterance_cmvn_leaks_the_future():
    """The violation that lives in the feature pipeline, not the model."""
    x = torch.randn(1, 30, 8)
    t = 5
    a = utterance_cmvn(x)
    b = utterance_cmvn(_perturb_future(x, t, dim=1))
    assert not torch.allclose(a[:, t], b[:, t], atol=1e-6)


def test_running_cmvn_is_causal():
    norm = RunningCMVN()
    x = torch.randn(1, 30, 8)
    t = 5
    a = norm(x)
    b = norm(_perturb_future(x, t, dim=1))
    torch.testing.assert_close(a[:, : t + 1], b[:, : t + 1])


def test_running_cmvn_streaming_equals_offline():
    norm = RunningCMVN()
    x = torch.randn(3, 25, 4)

    offline = norm(x)
    state = norm.init_state(3, 4)
    pieces = []
    for start in range(0, x.shape[1], 6):
        y, state = norm.step(x[:, start : start + 6, :], state)
        pieces.append(y)
    streaming = torch.cat(pieces, dim=1)

    torch.testing.assert_close(offline, streaming, atol=1e-5, rtol=1e-4)
