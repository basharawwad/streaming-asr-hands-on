"""Continuous embeddings into an LLM, and the three ways it goes wrong."""

import pytest
import torch

from streaming_asr.continuous_adapter import (
    IGNORE_INDEX,
    AudioProjector,
    FrameStacker,
    PlaceholderCountMismatch,
    TinyLM,
    ToyVectorQuantiser,
    build_labels,
    mean_row_norm,
    reconstruction_error,
    residual_quantise,
    splice_audio_embeddings,
    tokens_per_second,
)

torch.manual_seed(0)

AUDIO_ID = 5          # the id reserved for audio positions
D_MODEL = 32
VOCAB = 64


def _prompt(num_frames: int, batch: int = 1) -> torch.Tensor:
    """[BOS] <audio> * n [text] [text] - one placeholder per audio frame."""
    row = [1] + [AUDIO_ID] * num_frames + [7, 8, 9]
    return torch.tensor([row] * batch)


# --------------------------------------------------------------------------
# The premise
# --------------------------------------------------------------------------


def test_there_is_no_continuous_token_you_just_skip_the_lookup():
    """Feeding ids and feeding their embeddings are the same computation.

    This is the whole trick. The embedding table is a lookup producing vectors,
    so handing the model vectors directly is not a special mode.
    """
    lm = TinyLM(VOCAB, D_MODEL).eval()
    ids = torch.randint(0, VOCAB, (2, 9))

    with torch.no_grad():
        from_ids, _ = lm(input_ids=ids)
        from_embeds, _ = lm(inputs_embeds=lm.get_input_embeddings()(ids))

    torch.testing.assert_close(from_ids, from_embeds)


# --------------------------------------------------------------------------
# The splice
# --------------------------------------------------------------------------


def test_splice_replaces_the_placeholder_rows_and_nothing_else():
    lm = TinyLM(VOCAB, D_MODEL).eval()
    ids = _prompt(num_frames=4)
    text_embeds = lm.get_input_embeddings()(ids)
    audio = torch.randn(1, 4, D_MODEL)

    spliced = splice_audio_embeddings(text_embeds, ids, audio, AUDIO_ID)

    torch.testing.assert_close(spliced[0, 1:5], audio[0])      # audio landed
    torch.testing.assert_close(spliced[0, 0], text_embeds[0, 0])   # BOS intact
    torch.testing.assert_close(spliced[0, 5:], text_embeds[0, 5:])  # tail intact


def test_splice_does_not_mutate_its_input():
    lm = TinyLM(VOCAB, D_MODEL).eval()
    ids = _prompt(3)
    text_embeds = lm.get_input_embeddings()(ids)
    before = text_embeds.clone()
    splice_audio_embeddings(text_embeds, ids, torch.randn(1, 3, D_MODEL), AUDIO_ID)
    torch.testing.assert_close(text_embeds, before)


def test_splice_works_across_a_batch():
    lm = TinyLM(VOCAB, D_MODEL).eval()
    ids = _prompt(num_frames=4, batch=3)
    audio = torch.randn(3, 4, D_MODEL)
    spliced = splice_audio_embeddings(
        lm.get_input_embeddings()(ids), ids, audio, AUDIO_ID
    )
    for b in range(3):
        torch.testing.assert_close(spliced[b, 1:5], audio[b])


# --------------------------------------------------------------------------
# Failure mode 1: placeholder misalignment
# --------------------------------------------------------------------------


def test_one_placeholder_too_few_is_caught():
    """The off-by-one that a frame stacker's dropped remainder causes."""
    lm = TinyLM(VOCAB, D_MODEL).eval()
    ids = _prompt(num_frames=4)
    audio = torch.randn(1, 5, D_MODEL)          # 5 frames, 4 slots

    with pytest.raises(PlaceholderCountMismatch, match="4 placeholder tokens but 5"):
        splice_audio_embeddings(lm.get_input_embeddings()(ids), ids, audio, AUDIO_ID)


def test_one_placeholder_too_many_is_caught():
    lm = TinyLM(VOCAB, D_MODEL).eval()
    ids = _prompt(num_frames=6)
    audio = torch.randn(1, 5, D_MODEL)

    with pytest.raises(PlaceholderCountMismatch):
        splice_audio_embeddings(lm.get_input_embeddings()(ids), ids, audio, AUDIO_ID)


def test_dimension_mismatch_is_reported_as_a_projector_problem():
    lm = TinyLM(VOCAB, D_MODEL).eval()
    ids = _prompt(4)
    with pytest.raises(ValueError, match="projector is misconfigured"):
        splice_audio_embeddings(
            lm.get_input_embeddings()(ids), ids, torch.randn(1, 4, D_MODEL // 2), AUDIO_ID
        )


def test_a_realistic_stacker_and_prompt_agree():
    """The count that must line up: frames after downsampling, not before."""
    stacker = FrameStacker(k=4)
    frames = stacker(torch.randn(1, 23, 8))       # 23 // 4 = 5, three dropped
    assert frames.shape[1] == 5

    lm = TinyLM(VOCAB, D_MODEL).eval()
    ids = _prompt(num_frames=5)
    audio = AudioProjector(8 * 4, D_MODEL)(frames)
    splice_audio_embeddings(lm.get_input_embeddings()(ids), ids, audio, AUDIO_ID)


# --------------------------------------------------------------------------
# Failure mode 2: loss leakage
# --------------------------------------------------------------------------


def test_labels_mask_the_audio_positions():
    ids = _prompt(num_frames=4)
    labels = build_labels(ids, AUDIO_ID)

    assert (labels[0, 1:5] == IGNORE_INDEX).all()
    torch.testing.assert_close(labels[0, 5:], ids[0, 5:])
    assert labels[0, 0] == ids[0, 0]


def test_forgetting_to_mask_changes_the_objective():
    """Unmasked, the model is trained to predict the placeholder id on audio.

    It converges, slowly, while spending capacity on a meaningless target. The
    symptom is a training curve that looks merely disappointing rather than
    broken, which is why this one survives review.
    """
    lm = TinyLM(VOCAB, D_MODEL).eval()
    ids = _prompt(num_frames=6)
    embeds = splice_audio_embeddings(
        lm.get_input_embeddings()(ids), ids, torch.randn(1, 6, D_MODEL), AUDIO_ID
    )

    with torch.no_grad():
        _, masked = lm(inputs_embeds=embeds, labels=build_labels(ids, AUDIO_ID))
        _, unmasked = lm(inputs_embeds=embeds, labels=ids)

    assert not torch.isclose(masked, unmasked), (
        "if these agree the mask is doing nothing; check the ignore index"
    )


def test_the_audio_changes_the_text_loss_even_though_it_is_masked():
    """Masked out of the loss is not the same as ignored.

    Audio positions contribute no target, but the text positions attend to them
    and their predictions change when the audio changes. That attention path is
    the only thing connecting the two, and it is what carries gradient back to
    the projector. If this test passes but the projector will not learn, the
    attention mask is wrong.
    """
    lm = TinyLM(VOCAB, D_MODEL).eval()
    ids = _prompt(num_frames=5)
    labels = build_labels(ids, AUDIO_ID)
    table = lm.get_input_embeddings()(ids)

    with torch.no_grad():
        losses = []
        for seed in (1, 2):
            g = torch.Generator().manual_seed(seed)
            audio = torch.randn(1, 5, D_MODEL, generator=g)
            embeds = splice_audio_embeddings(table, ids, audio, AUDIO_ID)
            losses.append(lm(inputs_embeds=embeds, labels=labels)[1])

    assert not torch.isclose(losses[0], losses[1]), (
        "the text loss must depend on the audio, or nothing is being learned"
    )


def test_the_first_transcript_token_is_predicted_at_the_last_audio_position():
    """Where the shift puts the burden.

    With the standard causal shift, the prediction scored against the first
    transcript token is made at the final audio position. That position is the
    one that must have understood the audio, which is why the adapter works at
    all.
    """
    ids = _prompt(num_frames=4)          # [BOS] a a a a t t t
    labels = build_labels(ids, AUDIO_ID)

    # Position 4 is the last audio frame; its target is labels[5], the first
    # text token after the audio.
    assert ids[0, 4] == AUDIO_ID
    assert labels[0, 5] == ids[0, 5] != IGNORE_INDEX


# --------------------------------------------------------------------------
# Failure mode 3: scale mismatch
# --------------------------------------------------------------------------


def test_calibration_matches_the_embedding_table_scale():
    lm = TinyLM(VOCAB, D_MODEL)
    frames = torch.randn(2, 10, 40)
    proj = AudioProjector(40, D_MODEL, normalise=True)

    proj.calibrate_scale(frames, lm.get_input_embeddings())

    target = mean_row_norm(lm.get_input_embeddings().weight)
    with torch.no_grad():
        got = mean_row_norm(proj(frames))
    assert abs(got - target) / target < 0.05, f"{got:.3f} against {target:.3f}"


def test_an_uncalibrated_projector_can_sit_at_the_wrong_magnitude():
    """Why calibration is worth one line of setup code.

    The LLM has never seen embeddings at this scale. In practice the audio is
    either ignored or it swamps the text, and neither looks like a bug.
    """
    lm = TinyLM(VOCAB, D_MODEL)
    frames = torch.randn(2, 10, 40)
    proj = AudioProjector(40, D_MODEL, normalise=False)
    with torch.no_grad():
        proj.linear.weight.mul_(50.0)

    target = mean_row_norm(lm.get_input_embeddings().weight)
    with torch.no_grad():
        got = mean_row_norm(proj(frames))
    assert got / target > 10.0, f"expected a large mismatch, got {got/target:.2f}x"


def test_calibration_reports_the_scale_it_chose():
    """A returned value far from 1 is the diagnostic worth logging."""
    lm = TinyLM(VOCAB, D_MODEL)
    proj = AudioProjector(40, D_MODEL)
    chosen = proj.calibrate_scale(torch.randn(2, 10, 40), lm.get_input_embeddings())
    assert chosen > 0
    assert abs(float(proj.scale.detach()) - chosen) < 1e-6


# --------------------------------------------------------------------------
# Streaming
# --------------------------------------------------------------------------


def test_frame_stacker_shape_math():
    assert FrameStacker(1)(torch.randn(1, 10, 4)).shape == (1, 10, 4)
    assert FrameStacker(5)(torch.randn(1, 50, 8)).shape == (1, 10, 40)


def test_frame_stacker_drops_the_trailing_remainder():
    out = FrameStacker(4)(torch.randn(2, 23, 6))
    assert out.shape == (2, 5, 24)          # 5 groups, 3 frames dropped


def test_frame_stacker_streaming_equals_offline_on_ragged_chunks():
    """Chunk boundaries rarely land on a multiple of k, hence the buffer."""
    stacker = FrameStacker(k=4)
    x = torch.randn(2, 30, 6)

    offline = stacker(x)

    state = stacker.init_state(2, 6)
    pieces = []
    pos = 0
    for size in [3, 7, 2, 5, 9, 4]:            # sums to 30, none a multiple of 4
        chunk = x[:, pos : pos + size, :]
        pos += size
        out, state = stacker.step(chunk, state)
        if out.shape[1]:
            pieces.append(out)
    streaming = torch.cat(pieces, dim=1)

    torch.testing.assert_close(offline, streaming)


def test_projector_is_pointwise_so_it_streams_for_free():
    """No state, no cache, no equality test to worry about."""
    proj = AudioProjector(24, D_MODEL).eval()
    x = torch.randn(1, 17, 24)
    with torch.no_grad():
        offline = proj(x)
        streaming = torch.cat([proj(x[:, i : i + 5, :]) for i in range(0, 17, 5)], dim=1)
    torch.testing.assert_close(offline, streaming)


# --------------------------------------------------------------------------
# The discrete alternative
# --------------------------------------------------------------------------


def test_quantisation_loses_information_and_continuous_does_not():
    """The cost you pay before the LLM sees anything."""
    x = torch.randn(1, 40, 16)
    _, quantised = ToyVectorQuantiser(dim=16, codebook_size=64)(x)

    assert reconstruction_error(x, quantised) > 0.0
    assert reconstruction_error(x, x) == 0.0


def test_a_bigger_codebook_reduces_the_error():
    x = torch.randn(1, 200, 8)
    errors = [
        reconstruction_error(x, ToyVectorQuantiser(8, k, seed=1)(x)[1])
        for k in (4, 32, 256)
    ]
    assert errors[0] > errors[1] > errors[2]


def test_residual_stages_reduce_the_error_monotonically():
    """Why a codec emits several tokens per frame rather than one."""
    x = torch.randn(1, 100, 8)
    stages = [ToyVectorQuantiser(8, 16, seed=s) for s in range(4)]

    errors = []
    for n in (1, 2, 3, 4):
        _, recon = residual_quantise(x, stages[:n])
        errors.append(reconstruction_error(x, recon))

    assert errors == sorted(errors, reverse=True), errors


def test_residual_quantise_returns_one_id_stream_per_stage():
    x = torch.randn(2, 25, 8)
    ids, _ = residual_quantise(x, [ToyVectorQuantiser(8, 16, seed=s) for s in range(3)])
    assert len(ids) == 3
    assert all(i.shape == (2, 25) for i in ids)


def test_flattening_codebooks_multiplies_the_sequence_length():
    """The arithmetic behind factorising the codebook axis instead.

    Mimi runs at 12.5 Hz with 8 quantisers. Flattened that is 100 positions per
    second of audio; factorised with a transformer over the codebook axis it
    stays at 12.5.
    """
    assert tokens_per_second(12.5, num_codebooks=8, flatten_codebooks=True) == 100.0
    assert tokens_per_second(12.5, num_codebooks=8, flatten_codebooks=False) == 12.5


def test_continuous_frame_rate_is_a_free_parameter():
    """Downsampling is yours to choose, which the codec's frame rate is not."""
    whisper_rate = 50.0
    assert tokens_per_second(whisper_rate / 5) == 10.0
    assert tokens_per_second(whisper_rate / 10) == 5.0


# --------------------------------------------------------------------------
# The SLAM-ASR recipe, end to end
# --------------------------------------------------------------------------


def test_only_the_projector_receives_gradient_when_both_ends_are_frozen():
    """Freeze the encoder and the LLM, train one linear layer.

    The recipe that reaches competitive LibriSpeech numbers on 4 GPUs in 4
    hours. The test checks the plumbing rather than the accuracy: gradient must
    reach the projector and nothing else.
    """
    torch.manual_seed(4)
    lm = TinyLM(VOCAB, D_MODEL)
    for p in lm.parameters():
        p.requires_grad_(False)

    encoder = torch.nn.Linear(13, 16)
    for p in encoder.parameters():
        p.requires_grad_(False)

    stacker = FrameStacker(k=2)
    proj = AudioProjector(16 * 2, D_MODEL)

    feats = torch.randn(1, 12, 13)
    audio = proj(stacker(encoder(feats)))               # 12 // 2 = 6 frames
    ids = _prompt(num_frames=6)
    embeds = splice_audio_embeddings(
        lm.get_input_embeddings()(ids), ids, audio, AUDIO_ID
    )
    _, loss = lm(inputs_embeds=embeds, labels=build_labels(ids, AUDIO_ID))
    loss.backward()

    assert proj.linear.weight.grad is not None
    assert float(proj.linear.weight.grad.abs().sum()) > 0
    assert all(p.grad is None for p in lm.parameters())
    assert all(p.grad is None for p in encoder.parameters())


def test_the_loss_falls_when_the_projector_alone_is_trained():
    torch.manual_seed(5)
    lm = TinyLM(VOCAB, D_MODEL)
    for p in lm.parameters():
        p.requires_grad_(False)

    stacker = FrameStacker(k=2)
    proj = AudioProjector(16, D_MODEL)
    proj.calibrate_scale(torch.randn(1, 6, 16), lm.get_input_embeddings())

    feats = torch.randn(1, 12, 8)
    ids = _prompt(num_frames=6)
    labels = build_labels(ids, AUDIO_ID)
    table = lm.get_input_embeddings()(ids)
    opt = torch.optim.Adam(proj.parameters(), lr=0.05)

    def step(train: bool) -> float:
        audio = proj(stacker(feats))
        embeds = splice_audio_embeddings(table, ids, audio, AUDIO_ID)
        _, loss = lm(inputs_embeds=embeds, labels=labels)
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        return float(loss.detach())

    first = step(train=False)
    for _ in range(60):
        step(train=True)
    last = step(train=False)

    assert last < first, f"loss went {first:.4f} -> {last:.4f}"
