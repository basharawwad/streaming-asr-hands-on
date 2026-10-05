"""RNN-T: the lattice, the forward algorithm, greedy decoding, and delay.

The transducer is the default streaming decoder, and the thing worth being able
to reconstruct is the lattice. Nodes are (t, u): t frames consumed, u labels
emitted. From each node there are two moves:

    blank  -> (t+1, u)     advance time, emit nothing
    label  -> (t, u+1)     emit the next target label, stay on this frame

Every monotonic staircase from (0, 0) to (T-1, U) followed by a final blank is
a valid alignment of the same transcript. The loss is the negative log of the
sum over all of them, computed by the forward recursion here and checked in the
tests against brute-force enumeration of every path.

That marginalisation explains the emission-delay problem exactly. The loss
weights a path only by its probability, so turning upward early and turning
upward late score identically. Nothing prefers committing as soon as the
evidence arrives, and blank is always locally safe. `delay_penalty` below is
the standard fix: break the tie by scoring label transitions lower the later
they happen.
"""

from __future__ import annotations

import itertools

import torch
import torch.nn as nn

NEG_INF = -1e30


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------


def rnnt_loss(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    blank: int = 0,
    delay_penalty: float = 0.0,
) -> torch.Tensor:
    """Forward algorithm over the transducer lattice, in log space.

    Args:
        log_probs: (B, T, U+1, V) joiner output, already log-softmaxed.
        targets: (B, U) target label ids, none equal to `blank`.
        blank: index of the blank symbol.
        delay_penalty: subtract `delay_penalty * t` from the score of a label
            transition taken at frame t. Zero reproduces the standard loss.
            Positive values bias the model towards emitting earlier, which is
            the k2 formulation of the idea behind FastEmit.

    Returns:
        (B,) negative log likelihood.
    """
    B, T, Up1, _ = log_probs.shape
    U = Up1 - 1
    assert targets.shape == (B, U), f"expected targets {(B, U)}, got {tuple(targets.shape)}"

    blank_lp = log_probs[..., blank]                                   # (B, T, U+1)
    if U > 0:
        idx = targets.view(B, 1, U, 1).expand(B, T, U, 1)
        label_lp = log_probs[:, :, :U, :].gather(-1, idx).squeeze(-1)  # (B, T, U)
        if delay_penalty != 0.0:
            t_idx = torch.arange(T, device=log_probs.device, dtype=log_probs.dtype)
            label_lp = label_lp - delay_penalty * t_idx.view(1, T, 1)
    else:
        label_lp = log_probs.new_zeros(B, T, 0)

    alpha = log_probs.new_full((B, T, U + 1), NEG_INF)
    alpha[:, 0, 0] = 0.0

    for t in range(T):
        for u in range(U + 1):
            if t == 0 and u == 0:
                continue
            candidates = []
            if t > 0:
                candidates.append(alpha[:, t - 1, u] + blank_lp[:, t - 1, u])
            if u > 0:
                candidates.append(alpha[:, t, u - 1] + label_lp[:, t, u - 1])
            alpha[:, t, u] = torch.logsumexp(torch.stack(candidates, dim=0), dim=0)

    return -(alpha[:, T - 1, U] + blank_lp[:, T - 1, U])


def rnnt_loss_bruteforce(
    log_probs: torch.Tensor, targets: torch.Tensor, blank: int = 0
) -> torch.Tensor:
    """Enumerate every monotonic path and sum. Only tractable for tiny lattices.

    This exists so the forward recursion above can be checked against the
    definition rather than against another implementation of itself. For T
    frames and U labels there are C(T-1+U, U) paths.

    log_probs: (T, U+1, V) for a single example. targets: (U,).
    """
    T = log_probs.shape[0]
    U = targets.shape[0]
    total = torch.tensor(float("-inf"), dtype=log_probs.dtype, device=log_probs.device)

    num_moves = T - 1 + U
    for label_steps in itertools.combinations(range(num_moves), U):
        label_at = set(label_steps)
        t = u = 0
        score = log_probs.new_zeros(())
        for step in range(num_moves):
            if step in label_at:
                score = score + log_probs[t, u, targets[u]]
                u += 1
            else:
                score = score + log_probs[t, u, blank]
                t += 1
        score = score + log_probs[t, u, blank]        # the final blank
        total = torch.logaddexp(total, score)

    return -total


def num_alignments(num_frames: int, num_labels: int) -> int:
    """How many distinct monotonic alignments the loss sums over."""
    from math import comb

    return comb(num_frames - 1 + num_labels, num_labels)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


class StatelessPredictor(nn.Module):
    """Prediction network with no recurrence, just the last N labels.

    Replacing the LSTM predictor with an embedding of the last one or two
    labels costs almost no accuracy and is now the default in the k2 recipes.
    For a streaming system the win is structural: there is no hidden state to
    carry across chunks, reset between utterances, or keep consistent when
    batching streams that started at different times. The only state left
    anywhere in the model is the encoder cache.
    """

    def __init__(self, vocab_size: int, dim: int, context: int = 2, blank: int = 0):
        super().__init__()
        self.context = context
        self.blank = blank
        self.embed = nn.Embedding(vocab_size, dim)
        self.proj = nn.Linear(dim * context, dim)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """history: (B, U+1, context) label ids, padded with blank at the start."""
        B, Up1, C = history.shape
        e = self.embed(history).reshape(B, Up1, C * self.embed.embedding_dim)
        return self.proj(e)

    def build_history(self, targets: torch.Tensor) -> torch.Tensor:
        """Turn (B, U) targets into the (B, U+1, context) predictor input.

        Position u sees the `context` labels immediately before it, so position
        0 sees only blanks. This is the teacher-forced arrangement used in
        training; `greedy_decode` builds the same thing incrementally.
        """
        B, U = targets.shape
        pad = targets.new_full((B, self.context), self.blank)
        padded = torch.cat([pad, targets], dim=1)               # (B, context+U)
        windows = [padded[:, u : u + self.context] for u in range(U + 1)]
        return torch.stack(windows, dim=1)                      # (B, U+1, context)


class Joiner(nn.Module):
    """Additive joiner: project both sides, add, squash, classify.

    The output for a full lattice is (B, T, U+1, V), which is the tensor that
    dominates transducer training memory. Pruned RNN-T exists because of this
    shape: evaluate a cheap linear joiner first, use it to find the (t, u)
    cells that actually carry the loss, then run the real joiner only there.
    """

    def __init__(self, enc_dim: int, pred_dim: int, hidden: int, vocab_size: int):
        super().__init__()
        self.enc_proj = nn.Linear(enc_dim, hidden)
        self.pred_proj = nn.Linear(pred_dim, hidden)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, enc: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """enc: (B, T, E), pred: (B, U+1, P) -> (B, T, U+1, V) log probs."""
        e = self.enc_proj(enc).unsqueeze(2)      # (B, T, 1, H)
        p = self.pred_proj(pred).unsqueeze(1)    # (B, 1, U+1, H)
        return self.out(torch.tanh(e + p)).log_softmax(dim=-1)

    def step(self, enc_t: torch.Tensor, pred_u: torch.Tensor) -> torch.Tensor:
        """One lattice cell. enc_t: (B, E), pred_u: (B, P) -> (B, V) log probs."""
        h = torch.tanh(self.enc_proj(enc_t) + self.pred_proj(pred_u))
        return self.out(h).log_softmax(dim=-1)


class TinyTransducer(nn.Module):
    """Encoder stub, stateless predictor, additive joiner.

    The encoder here is a linear layer so the tests stay fast; in a real system
    it is the streaming Conformer stack in `conformer.py`, and nothing else
    about the transducer changes.
    """

    def __init__(
        self,
        input_dim: int = 16,
        enc_dim: int = 32,
        pred_dim: int = 32,
        joiner_dim: int = 32,
        vocab_size: int = 8,
        context: int = 2,
        blank: int = 0,
    ):
        super().__init__()
        self.blank = blank
        self.vocab_size = vocab_size
        self.encoder = nn.Linear(input_dim, enc_dim)
        self.predictor = StatelessPredictor(vocab_size, pred_dim, context, blank)
        self.joiner = Joiner(enc_dim, pred_dim, joiner_dim, vocab_size)

    def forward(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Full lattice log probs. features: (B, T, F), targets: (B, U)."""
        enc = self.encoder(features)
        pred = self.predictor(self.predictor.build_history(targets))
        return self.joiner(enc, pred)

    @torch.no_grad()
    def greedy_decode(
        self, features: torch.Tensor, max_symbols_per_frame: int = 3
    ) -> tuple[list[int], list[int]]:
        """Decode one utterance and report when each label was emitted.

        features: (1, T, F)

        Returns (labels, emission_frames), where emission_frames[i] is the
        frame index at which labels[i] was committed. Those indices are what
        `latency.py` turns into an emission delay against a reference
        alignment.

        `max_symbols_per_frame` guards the loop: without it a model that always
        prefers a label over blank never advances time.
        """
        assert features.shape[0] == 1, "greedy_decode handles one utterance"
        enc = self.encoder(features)
        T = enc.shape[1]
        ctx = self.predictor.context

        labels: list[int] = []
        frames: list[int] = []
        history = torch.full((1, ctx), self.blank, dtype=torch.long, device=features.device)

        for t in range(T):
            emitted_here = 0
            while emitted_here < max_symbols_per_frame:
                pred = self.predictor(history.unsqueeze(1)).squeeze(1)   # (1, P)
                logp = self.joiner.step(enc[:, t, :], pred)              # (1, V)
                best = int(logp.argmax(dim=-1).item())
                if best == self.blank:
                    break
                labels.append(best)
                frames.append(t)
                history = torch.cat(
                    [history[:, 1:], torch.tensor([[best]], device=features.device)],
                    dim=1,
                )
                emitted_here += 1

        return labels, frames
