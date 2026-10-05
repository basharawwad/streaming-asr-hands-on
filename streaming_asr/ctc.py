"""CTC: the forward algorithm, and the decode bug everyone writes once.

CTC is the cheaper streaming decoder. It is frame-synchronous, trivially
parallel, and gives good timestamps. What it does not have is a prediction
network, so outputs are conditionally independent given the audio and the model
carries no internal language model. That is the substantive difference from a
transducer, and the reason CTC systems lean on an external LM or on contextual
biasing to get entities right.

See `docs/ctc.md` for the trellis drawn out.
"""

from __future__ import annotations

import itertools

import torch


def ctc_greedy_decode(log_probs: torch.Tensor, blank: int = 0) -> list[int]:
    """Best-path decode, done in the right order.

    Collapse runs of repeated labels FIRST, then remove blanks. The blank
    symbol exists precisely to separate genuine repeated labels, so a blank
    between two identical frames means "two of these", not "one".

    log_probs: (T, V) for a single utterance.
    """
    ids = log_probs.argmax(dim=-1).tolist()
    collapsed = [k for k, _ in itertools.groupby(ids)]
    return [i for i in collapsed if i != blank]


def ctc_greedy_decode_buggy(log_probs: torch.Tensor, blank: int = 0) -> list[int]:
    """The same function with the two steps the wrong way round.

    Strips blanks before collapsing, so any genuinely repeated label collapses
    into one. "hello" comes out as "helo". It passes a smoke test on most
    utterances, which is why it survives code review.
    """
    ids = log_probs.argmax(dim=-1).tolist()
    stripped = [i for i in ids if i != blank]
    return [k for k, _ in itertools.groupby(stripped)]


def ctc_loss_forward(
    log_probs: torch.Tensor, targets: torch.Tensor, blank: int = 0
) -> torch.Tensor:
    """Forward algorithm over the extended label sequence, for one utterance.

    The target is padded with blanks between every label and at both ends,
    giving 2U+1 states. Transitions: stay, advance one, or skip a blank when
    the two labels either side of it differ.

    log_probs: (T, V). targets: (U,).
    """
    T = log_probs.shape[0]
    U = targets.shape[0]
    S = 2 * U + 1

    ext = torch.full((S,), blank, dtype=torch.long, device=targets.device)
    ext[1::2] = targets

    alpha = log_probs.new_full((T, S), -1e30)
    alpha[0, 0] = log_probs[0, ext[0]]
    if S > 1:
        alpha[0, 1] = log_probs[0, ext[1]]

    for t in range(1, T):
        for s in range(S):
            candidates = [alpha[t - 1, s]]
            if s > 0:
                candidates.append(alpha[t - 1, s - 1])
            # A skip is only allowed past a blank separating two different labels.
            if s > 1 and ext[s] != blank and ext[s] != ext[s - 2]:
                candidates.append(alpha[t - 1, s - 2])
            alpha[t, s] = torch.logsumexp(torch.stack(candidates), dim=0) + log_probs[t, ext[s]]

    last = [alpha[T - 1, S - 1]]
    if S > 1:
        last.append(alpha[T - 1, S - 2])
    return -torch.logsumexp(torch.stack(last), dim=0)


def peaky_fraction(log_probs: torch.Tensor, blank: int = 0) -> float:
    """Fraction of frames whose best path symbol is blank.

    CTC alignments are famously peaky: a trained model spends most frames
    predicting blank and fires a label for one or two frames. Handy as a
    diagnostic, and the reason CTC timestamps are precise but its posteriors
    are poorly calibrated for a commit decision.
    """
    ids = log_probs.argmax(dim=-1)
    return float((ids == blank).float().mean().item())
