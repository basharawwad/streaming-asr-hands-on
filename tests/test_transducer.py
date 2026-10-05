"""The forward algorithm checked against the definition it implements."""

import torch

from streaming_asr.transducer import (
    StatelessPredictor,
    TinyTransducer,
    num_alignments,
    rnnt_loss,
    rnnt_loss_bruteforce,
)

torch.manual_seed(0)


def _log_probs(T: int, U: int, V: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(T, U + 1, V, generator=g).log_softmax(dim=-1)


def test_forward_algorithm_matches_brute_force_enumeration():
    """The test that proves the recursion, rather than comparing it to itself.

    For T=4 and U=3 there are 20 monotonic paths. The loss is the negative log
    of their summed probability, and the forward recursion computes exactly
    that in O(T*U) instead of O(number of paths).
    """
    T, U, V = 4, 3, 5
    lp = _log_probs(T, U, V)
    targets = torch.tensor([1, 2, 1])

    assert num_alignments(T, U) == 20

    fast = rnnt_loss(lp.unsqueeze(0), targets.unsqueeze(0), blank=0)
    slow = rnnt_loss_bruteforce(lp, targets, blank=0)

    torch.testing.assert_close(fast.squeeze(0), slow, atol=1e-5, rtol=1e-5)


def test_forward_matches_brute_force_across_shapes():
    for T, U in [(1, 0), (2, 1), (3, 1), (5, 4), (6, 2)]:
        vocab = U + 2                      # blank at 0, plus one id per label
        lp = _log_probs(T, U, vocab, seed=T * 10 + U)
        targets = torch.arange(1, U + 1)
        fast = rnnt_loss(lp.unsqueeze(0), targets.unsqueeze(0))
        slow = rnnt_loss_bruteforce(lp, targets)
        torch.testing.assert_close(
            fast.squeeze(0), slow, atol=1e-5, rtol=1e-5,
            msg=f"mismatch at T={T}, U={U}",
        )


def test_loss_is_a_proper_negative_log_likelihood():
    """Non-negative, and finite whenever at least one path exists."""
    lp = _log_probs(5, 3, 6)
    loss = rnnt_loss(lp.unsqueeze(0), torch.tensor([[1, 2, 3]]))
    assert torch.isfinite(loss).all()
    assert float(loss) >= 0.0


def test_loss_is_differentiable():
    lp = _log_probs(4, 2, 5).requires_grad_(True)
    rnnt_loss(lp.unsqueeze(0), torch.tensor([[1, 2]])).sum().backward()
    assert lp.grad is not None and torch.isfinite(lp.grad).all()
    assert float(lp.grad.abs().sum()) > 0


def test_batching_matches_per_example_losses():
    a, b = _log_probs(4, 2, 5, seed=1), _log_probs(4, 2, 5, seed=2)
    tg = torch.tensor([[1, 2], [3, 4]])
    batched = rnnt_loss(torch.stack([a, b]), tg)
    single = torch.stack([
        rnnt_loss(a.unsqueeze(0), tg[:1]).squeeze(0),
        rnnt_loss(b.unsqueeze(0), tg[1:]).squeeze(0),
    ])
    torch.testing.assert_close(batched, single)


# --------------------------------------------------------------------------
# Delay
# --------------------------------------------------------------------------


def test_delay_penalty_reweights_towards_early_emission():
    """The loss is indifferent to when a label is emitted. This breaks the tie.

    Build a lattice where the label is equally likely at every frame, then
    check that the delay penalty makes late alignments cost more. This is the
    mechanism behind FastEmit and the k2 delay penalty.
    """
    T, U, V = 6, 1, 3
    lp = torch.full((1, T, U + 1, V), -1.0986).log_softmax(dim=-1)   # uniform
    targets = torch.tensor([[1]])

    plain = rnnt_loss(lp, targets, delay_penalty=0.0)
    penalised = rnnt_loss(lp, targets, delay_penalty=0.3)

    # Penalising late label transitions can only reduce total path mass.
    assert float(penalised) > float(plain)


def test_delay_penalty_is_a_no_op_at_zero():
    lp = _log_probs(4, 2, 5).unsqueeze(0)
    tg = torch.tensor([[1, 2]])
    torch.testing.assert_close(
        rnnt_loss(lp, tg, delay_penalty=0.0), rnnt_loss(lp, tg)
    )


# --------------------------------------------------------------------------
# Model pieces
# --------------------------------------------------------------------------


def test_stateless_predictor_history_is_causal_and_padded():
    pred = StatelessPredictor(vocab_size=6, dim=8, context=2, blank=0)
    targets = torch.tensor([[3, 4, 5]])
    hist = pred.build_history(targets)

    assert hist.shape == (1, 4, 2)
    torch.testing.assert_close(hist[0, 0], torch.tensor([0, 0]))   # nothing yet
    torch.testing.assert_close(hist[0, 1], torch.tensor([0, 3]))
    torch.testing.assert_close(hist[0, 2], torch.tensor([3, 4]))
    torch.testing.assert_close(hist[0, 3], torch.tensor([4, 5]))


def test_stateless_predictor_output_depends_only_on_recent_labels():
    """The whole point: no hidden state, so context beyond N is invisible."""
    pred = StatelessPredictor(vocab_size=8, dim=8, context=2).eval()
    a = pred.build_history(torch.tensor([[1, 2, 3, 4]]))
    b = pred.build_history(torch.tensor([[7, 6, 3, 4]]))   # differs only early
    with torch.no_grad():
        torch.testing.assert_close(pred(a)[:, -1], pred(b)[:, -1])


def test_lattice_shape_is_the_memory_problem():
    """(B, T, U+1, V) is the tensor pruned RNN-T exists to avoid."""
    model = TinyTransducer(input_dim=8, vocab_size=7).eval()
    feats = torch.randn(2, 9, 8)
    targets = torch.tensor([[1, 2, 3], [4, 5, 6]])
    with torch.no_grad():
        lattice = model(feats, targets)
    assert lattice.shape == (2, 9, 4, 7)
    torch.testing.assert_close(
        lattice.exp().sum(-1), torch.ones(2, 9, 4), atol=1e-5, rtol=1e-5
    )


def test_greedy_decode_is_monotonic_and_bounded():
    model = TinyTransducer(input_dim=8, vocab_size=7).eval()
    feats = torch.randn(1, 12, 8)
    labels, frames = model.greedy_decode(feats, max_symbols_per_frame=2)

    assert len(labels) == len(frames)
    assert all(f2 >= f1 for f1, f2 in zip(frames, frames[1:])), "emission must advance"
    assert all(0 <= f < 12 for f in frames)
    assert len(labels) <= 12 * 2, "the per-frame guard must bound the loop"
    assert model.blank not in labels


def test_loss_decreases_when_the_model_is_trained_on_one_example():
    """End-to-end sanity: the pieces fit together and gradients flow."""
    torch.manual_seed(7)
    model = TinyTransducer(input_dim=8, enc_dim=16, pred_dim=16, joiner_dim=16, vocab_size=6)
    feats = torch.randn(1, 8, 8)
    targets = torch.tensor([[1, 2, 3]])
    opt = torch.optim.Adam(model.parameters(), lr=0.05)

    with torch.no_grad():
        first = float(rnnt_loss(model(feats, targets), targets))
    for _ in range(40):
        opt.zero_grad()
        loss = rnnt_loss(model(feats, targets), targets).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        last = float(rnnt_loss(model(feats, targets), targets))

    assert last < first * 0.5, f"loss went {first:.3f} -> {last:.3f}"
