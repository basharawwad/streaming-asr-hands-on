"""CTC decode ordering, and the forward algorithm against PyTorch's."""

import torch

from streaming_asr.ctc import (
    ctc_greedy_decode,
    ctc_greedy_decode_buggy,
    ctc_loss_forward,
    peaky_fraction,
)

torch.manual_seed(0)

BLANK = 0
H, E, L, O = 1, 2, 3, 4


def _one_hot_path(ids: list[int], vocab: int = 5) -> torch.Tensor:
    """A deterministic frame sequence, as log probs, for decode tests."""
    lp = torch.full((len(ids), vocab), -20.0)
    for t, i in enumerate(ids):
        lp[t, i] = 0.0
    return lp.log_softmax(dim=-1)


def test_blank_separates_genuine_repeated_labels():
    """h e l <blank> l o decodes as "hello", not "helo".

    This is the reason to collapse repeats before stripping blanks. The blank
    exists to say "these two are different letters".
    """
    frames = _one_hot_path([H, E, L, BLANK, L, O])
    assert ctc_greedy_decode(frames, blank=BLANK) == [H, E, L, L, O]


def test_the_wrong_order_silently_loses_a_letter():
    frames = _one_hot_path([H, E, L, BLANK, L, O])
    assert ctc_greedy_decode_buggy(frames, blank=BLANK) == [H, E, L, O]


def test_repeated_frames_of_one_label_collapse_to_one():
    frames = _one_hot_path([H, H, H, E, E, L, L, L, BLANK, L, O, O])
    assert ctc_greedy_decode(frames, blank=BLANK) == [H, E, L, L, O]


def test_leading_and_trailing_blanks_are_dropped():
    frames = _one_hot_path([BLANK, BLANK, H, E, BLANK, BLANK])
    assert ctc_greedy_decode(frames, blank=BLANK) == [H, E]


def test_all_blank_input_decodes_to_nothing():
    assert ctc_greedy_decode(_one_hot_path([BLANK] * 5), blank=BLANK) == []


def test_forward_algorithm_matches_torch_ctc_loss():
    """Check the hand-written recursion against the reference implementation."""
    T, V, U = 12, 6, 4
    lp = torch.randn(T, V).log_softmax(dim=-1)
    targets = torch.tensor([1, 2, 2, 3])

    mine = ctc_loss_forward(lp, targets, blank=BLANK)
    theirs = torch.nn.functional.ctc_loss(
        lp.unsqueeze(1),                       # (T, N, V)
        targets.unsqueeze(0),                  # (N, U)
        torch.tensor([T]),
        torch.tensor([U]),
        blank=BLANK,
        reduction="none",
        zero_infinity=False,
    )
    torch.testing.assert_close(mine, theirs.squeeze(0), atol=1e-4, rtol=1e-4)


def test_forward_algorithm_matches_torch_without_repeats():
    T, V = 8, 5
    lp = torch.randn(T, V).log_softmax(dim=-1)
    targets = torch.tensor([1, 2, 3])
    mine = ctc_loss_forward(lp, targets, blank=BLANK)
    theirs = torch.nn.functional.ctc_loss(
        lp.unsqueeze(1), targets.unsqueeze(0),
        torch.tensor([T]), torch.tensor([3]),
        blank=BLANK, reduction="none",
    )
    torch.testing.assert_close(mine, theirs.squeeze(0), atol=1e-4, rtol=1e-4)


def test_peaky_fraction_reports_blank_dominance():
    frames = _one_hot_path([BLANK, BLANK, H, BLANK, BLANK, E, BLANK, BLANK])
    assert abs(peaky_fraction(frames, blank=BLANK) - 0.75) < 1e-6
