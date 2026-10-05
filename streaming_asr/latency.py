"""The four latencies, kept apart.

Papers report one number and it is almost always the first of these. A user
waits for all four.

  1. algorithmic delay   chunk size plus right context, fixed by architecture
  2. emission delay      how long after the acoustic evidence the model commits
  3. compute latency     real time factor, batching, queueing
  4. endpointing delay   only for turn-final output, but it dominates how
                         responsive a voice agent feels

Aligned Word Emission Delay measures (1) and (2) together, from each word's
acoustic endpoint rather than from the start of the input. The gap between AWED
and the advertised structural delay can be an order of magnitude, because the
transducer loss gives the model no reason to commit early (see
`transducer.py`).
"""

from __future__ import annotations

from statistics import median


def emission_delays(
    emission_times_s: list[float], reference_end_times_s: list[float]
) -> list[float]:
    """Per-word delay from acoustic endpoint to the moment the word was emitted.

    Both lists are in seconds and must describe the same words in order. A
    negative value means the model committed before the word finished, which is
    possible and often correct when the word is predictable from its prefix.
    """
    if len(emission_times_s) != len(reference_end_times_s):
        raise ValueError(
            f"{len(emission_times_s)} emissions against "
            f"{len(reference_end_times_s)} reference words; align them first"
        )
    return [e - r for e, r in zip(emission_times_s, reference_end_times_s)]


def awed(emission_times_s: list[float], reference_end_times_s: list[float]) -> float:
    """Median aligned word emission delay, in seconds.

    The median rather than the mean, because the distribution has a tail and
    the tail is what users notice. Report a high percentile alongside it.
    """
    delays = emission_delays(emission_times_s, reference_end_times_s)
    if not delays:
        raise ValueError("no words to measure")
    return median(delays)


def frames_to_seconds(frame_indices: list[int], frame_ms: float) -> list[float]:
    """Convert decoder frame indices to seconds at the post-subsampling rate."""
    return [i * frame_ms / 1000.0 for i in frame_indices]


def effective_latency(algorithmic_delay_s: float, rtfx: float) -> float:
    """Structural delay plus the compute needed to catch up with it.

        delay * (1 + 1/RTFx)

    RTFx is how many times faster than real time the model runs. At RTFx just
    above 1 the compute term nearly doubles the delay; by RTFx 6 it adds about
    a sixth. This is the term that disappears from a paper's latency figure and
    reappears in production under load.
    """
    if rtfx <= 0:
        raise ValueError("rtfx must be positive")
    return algorithmic_delay_s * (1.0 + 1.0 / rtfx)


def user_perceived_turn_latency(
    algorithmic_delay_s: float,
    emission_delay_s: float,
    rtfx: float,
    endpoint_silence_s: float,
    network_s: float = 0.0,
) -> dict[str, float]:
    """Add up what actually sits between a speaker stopping and a reply starting.

    Returns the breakdown as well as the total, because the useful output of
    this calculation is which term dominates, not the number itself.
    """
    compute = algorithmic_delay_s / rtfx
    total = algorithmic_delay_s + emission_delay_s + compute + endpoint_silence_s + network_s
    return {
        "algorithmic": algorithmic_delay_s,
        "emission": emission_delay_s,
        "compute": compute,
        "endpointing": endpoint_silence_s,
        "network": network_s,
        "total": total,
    }
