"""Keeping the four latencies apart, and the endpointer's asymmetric loss."""

import pytest

from streaming_asr.endpointing import EndpointConfig, Endpointer, TurnState
from streaming_asr.latency import (
    awed,
    effective_latency,
    emission_delays,
    frames_to_seconds,
    user_perceived_turn_latency,
)
from streaming_asr.masks import algorithmic_delay_seconds


# --------------------------------------------------------------------------
# Emission delay
# --------------------------------------------------------------------------


def test_emission_delay_is_measured_from_the_acoustic_endpoint():
    """Not from the start of the utterance, which is what makes it comparable."""
    emitted = [1.10, 2.35, 3.05]
    word_ends = [1.00, 2.00, 3.00]
    assert emission_delays(emitted, word_ends) == pytest.approx([0.10, 0.35, 0.05])


def test_awed_uses_the_median_because_the_tail_is_the_problem():
    emitted = [1.05, 2.05, 3.05, 4.05, 5.90]     # one bad word
    ends = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert awed(emitted, ends) == pytest.approx(0.05)


def test_negative_delay_is_legal():
    """A model may commit before a word finishes when the prefix determines it."""
    assert emission_delays([0.9], [1.0]) == pytest.approx([-0.1])


def test_mismatched_lengths_are_rejected_rather_than_zipped():
    with pytest.raises(ValueError):
        emission_delays([1.0, 2.0], [1.0])


def test_structural_delay_can_understate_what_the_user_waits():
    """The finding worth remembering: nominal delay and measured delay diverge.

    A model configured with 80 ms of structural delay was measured with a
    median word emission delay near 0.77 s, because the loss gives it no reason
    to commit as soon as the evidence arrives.
    """
    structural = 0.080
    measured = awed([0.85, 0.77, 0.74], [0.0, 0.0, 0.0])
    assert measured > 9 * structural


# --------------------------------------------------------------------------
# Compute and totals
# --------------------------------------------------------------------------


def test_effective_latency_adds_the_catch_up_term():
    """delay * (1 + 1/RTFx). At RTFx 6 and 0.56 s of delay, about 0.65 s."""
    assert effective_latency(0.56, rtfx=6.0) == pytest.approx(0.6533, abs=1e-4)


def test_effective_latency_doubles_at_real_time():
    assert effective_latency(0.5, rtfx=1.0) == pytest.approx(1.0)


def test_effective_latency_rejects_a_stalled_pipeline():
    with pytest.raises(ValueError):
        effective_latency(0.5, rtfx=0.0)


def test_turn_latency_breakdown_sums_and_identifies_the_dominant_term():
    parts = user_perceived_turn_latency(
        algorithmic_delay_s=algorithmic_delay_seconds(14, frame_ms=40.0),
        emission_delay_s=0.30,
        rtfx=6.0,
        endpoint_silence_s=0.64,
    )
    assert parts["total"] == pytest.approx(
        parts["algorithmic"] + parts["emission"] + parts["compute"]
        + parts["endpointing"] + parts["network"]
    )
    named = {k: v for k, v in parts.items() if k != "total"}
    assert max(named, key=named.get) == "endpointing"


def test_frames_to_seconds_uses_the_post_subsampling_rate():
    assert frames_to_seconds([0, 10, 25], frame_ms=40.0) == pytest.approx([0.0, 0.4, 1.0])


# --------------------------------------------------------------------------
# Endpointing
# --------------------------------------------------------------------------


def _run(ep: Endpointer, pattern: list[tuple[float, bool]]) -> int | None:
    """Feed frames and return the index where the turn ended, or None."""
    for i, (prob, complete) in enumerate(pattern):
        if ep.update(prob, complete):
            return i
    return None


def test_silence_alone_ends_the_turn_at_the_hard_cutoff():
    ep = Endpointer(EndpointConfig(min_turn_silence_ms=128, max_turn_silence_ms=640),
                    frame_ms=10.0)
    pattern = [(0.9, False)] * 20 + [(0.0, False)] * 100
    assert _run(ep, pattern) == 20 + 64 - 1


def test_a_complete_sounding_phrase_ends_the_turn_much_sooner():
    """The semantic signal is what buys back the latency silence alone costs."""
    ep = Endpointer(EndpointConfig(128, 640), frame_ms=10.0)
    pattern = [(0.9, False)] * 20 + [(0.0, True)] * 100
    assert _run(ep, pattern) == 20 + 13 - 1      # 130 ms of silence, not 640


def test_semantic_signal_is_ignored_below_the_minimum_silence():
    """Otherwise a mid-sentence pause after a complete clause cuts the user off."""
    ep = Endpointer(EndpointConfig(min_turn_silence_ms=500, max_turn_silence_ms=2000),
                    frame_ms=10.0)
    pattern = [(0.9, False)] * 10 + [(0.0, True)] * 20 + [(0.9, False)] * 10
    assert _run(ep, pattern) is None
    assert ep.state is TurnState.LISTENING


def test_resumed_speech_resets_the_silence_accumulator():
    ep = Endpointer(EndpointConfig(128, 640), frame_ms=10.0)
    for _ in range(10):
        ep.update(0.9)
    for _ in range(30):
        ep.update(0.0)
    ep.update(0.9)
    assert ep.silence_ms == 0.0


def test_silence_before_any_speech_does_not_end_a_turn():
    ep = Endpointer(EndpointConfig(128, 640), frame_ms=10.0)
    assert _run(ep, [(0.0, True)] * 200) is None
    assert ep.state is TurnState.IDLE


def test_presets_are_ordered_from_impatient_to_careful():
    fast = EndpointConfig.preset("min_latency")
    mid = EndpointConfig.preset("balanced")
    slow = EndpointConfig.preset("max_accuracy")
    assert (
        fast.max_turn_silence_ms
        < mid.max_turn_silence_ms
        < slow.max_turn_silence_ms
    )
    assert fast.min_turn_silence_ms <= slow.min_turn_silence_ms


def test_raising_patience_mid_call_protects_a_spoken_phone_number():
    """Switching mode by call stage is the standard fix for digit strings.

    The same pause that should end a conversational turn should not end one in
    the middle of someone reciting a number. Note `looks_complete=False`: a
    punctuation model does not see a finished sentence halfway through a phone
    number, so the semantic signal offers no protection here and the hard
    cutoff is doing all the work. That is exactly why the stage needs a larger
    `max_turn_silence` rather than a smarter endpointer.
    """
    pause = [(0.0, False)] * 70                  # 700 ms of silence mid-number

    impatient = Endpointer(EndpointConfig.preset("min_latency"), frame_ms=10.0)
    for _ in range(10):
        impatient.update(0.9)
    assert _run(impatient, pause) is not None

    careful = Endpointer(EndpointConfig.preset("max_accuracy"), frame_ms=10.0)
    for _ in range(10):
        careful.update(0.9)
    assert _run(careful, pause) is None


def test_force_endpoint_skips_the_wait_entirely():
    ep = Endpointer(EndpointConfig.preset("balanced"), frame_ms=10.0)
    ep.update(0.9)
    assert ep.force_endpoint() is True
    assert ep.state is TurnState.ENDED


def test_force_endpoint_does_nothing_before_any_speech():
    ep = Endpointer(EndpointConfig.preset("balanced"), frame_ms=10.0)
    assert ep.force_endpoint() is False
