"""VAD, endpointing and turn detection are three different things.

    VAD              is there speech in this frame?
    endpointing      has the speaker finished?
    turn detection   should the system take the floor now?

Only the first is an acoustic question. Deciding that a speaker has finished
needs linguistic evidence, because a pause in the middle of a thought and a
pause at the end of one sound the same. Production systems read the punctuation
their own recogniser predicts, which is also why the punctuation stage has to be
causal.

The asymmetry is the design principle: cutting a user off mid-sentence costs far
more than half a second of dead air, so the threshold is a decision under an
asymmetric loss rather than a latency optimisation. The asymmetry softens in an
agent that can be interrupted cheaply, because a wrong early turn-take is
recoverable when barge-in works.

Preset values follow the ranges published for commercial streaming endpointers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TurnState(Enum):
    IDLE = "idle"           # nothing heard yet
    LISTENING = "listening"  # user is or was recently speaking
    ENDED = "ended"          # turn is over, the system may speak


@dataclass(frozen=True)
class EndpointConfig:
    """Silence thresholds in milliseconds.

    min_turn_silence: how long to wait before even asking whether the turn is
        over. Below this the semantic signal is not consulted.
    max_turn_silence: hard cutoff. The turn ends on silence alone, however
        unfinished the sentence sounds.
    vad_threshold: speech confidence above which a frame counts as speech.
    """

    min_turn_silence_ms: float
    max_turn_silence_ms: float
    vad_threshold: float = 0.2

    @staticmethod
    def preset(name: str) -> "EndpointConfig":
        presets = {
            # Rapid exchanges and menu-style prompts. Interrupts sometimes.
            "min_latency": EndpointConfig(128, 640),
            # The usual default for conversational agents.
            "balanced": EndpointConfig(128, 1280),
            # Noisy audio, unfamiliar accents, or collecting a phone number.
            "max_accuracy": EndpointConfig(512, 2560),
        }
        if name not in presets:
            raise ValueError(f"unknown preset {name!r}; try {sorted(presets)}")
        return presets[name]


class Endpointer:
    """Silence accumulator plus a semantic gate.

    Feed it one frame at a time. `update` returns True on the frame where the
    turn ends.

    Switching config mid-call is a real technique rather than a curiosity: an
    agent collecting an address or a card number should raise its patience to
    `max_accuracy` for that stretch and drop back afterwards, because the cost
    of interrupting someone reciting digits is much higher than usual.
    """

    def __init__(self, config: EndpointConfig, frame_ms: float = 10.0):
        self.config = config
        self.frame_ms = frame_ms
        self.reset()

    def reset(self) -> None:
        self.state = TurnState.IDLE
        self.silence_ms = 0.0
        self.speech_ms = 0.0

    def set_config(self, config: EndpointConfig) -> None:
        """Change patience without losing the current silence accumulation."""
        self.config = config

    def update(self, speech_prob: float, looks_complete: bool = False) -> bool:
        """Advance one frame.

        Args:
            speech_prob: VAD confidence for this frame.
            looks_complete: the semantic signal, in practice derived from the
                punctuation the recogniser predicted for the text so far.

        Returns:
            True on the frame where the turn ends.
        """
        if self.state is TurnState.ENDED:
            return False

        if speech_prob >= self.config.vad_threshold:
            self.state = TurnState.LISTENING
            self.speech_ms += self.frame_ms
            self.silence_ms = 0.0
            return False

        if self.state is TurnState.IDLE:
            return False

        self.silence_ms += self.frame_ms

        if self.silence_ms >= self.config.max_turn_silence_ms:
            self.state = TurnState.ENDED
            return True

        if looks_complete and self.silence_ms >= self.config.min_turn_silence_ms:
            self.state = TurnState.ENDED
            return True

        return False

    def force_endpoint(self) -> bool:
        """End the turn now, without waiting for silence.

        For when the application knows the turn is over and the audio does not:
        the caller pressed a key, or the form field is full.
        """
        if self.state is TurnState.IDLE:
            return False
        self.state = TurnState.ENDED
        return True
