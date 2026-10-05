"""Streaming ASR mechanics, built from scratch and tested for causality.

The organising idea: streaming correctness is a property you can test. For each
component there is an offline path and a streaming path, and a test asserting
they produce the same output. Where a naive implementation silently reads the
future, there is a second test that perturbs a future frame and proves it.
"""

from .cache import LeftContextKVCache, kv_cache_bytes
from .causality import (
    CausalDepthwiseConv1d,
    CentredDepthwiseConv1d,
    RunningCMVN,
    utterance_cmvn,
)
from .conformer import StreamingConformerBlock
from .continuous_adapter import (
    AudioProjector,
    FrameStacker,
    PlaceholderCountMismatch,
    TinyLM,
    ToyVectorQuantiser,
    build_labels,
    residual_quantise,
    splice_audio_embeddings,
    tokens_per_second,
)
from .ctc import ctc_greedy_decode, ctc_greedy_decode_buggy, ctc_loss_forward
from .endpointing import EndpointConfig, Endpointer, TurnState
from .latency import awed, effective_latency, emission_delays, user_perceived_turn_latency
from .masks import algorithmic_delay_frames, causal_mask, chunk_mask, full_mask
from .transducer import (
    Joiner,
    StatelessPredictor,
    TinyTransducer,
    num_alignments,
    rnnt_loss,
    rnnt_loss_bruteforce,
)

__all__ = [
    "CausalDepthwiseConv1d",
    "CentredDepthwiseConv1d",
    "RunningCMVN",
    "utterance_cmvn",
    "chunk_mask",
    "causal_mask",
    "full_mask",
    "algorithmic_delay_frames",
    "LeftContextKVCache",
    "kv_cache_bytes",
    "StreamingConformerBlock",
    "FrameStacker",
    "AudioProjector",
    "splice_audio_embeddings",
    "build_labels",
    "PlaceholderCountMismatch",
    "TinyLM",
    "ToyVectorQuantiser",
    "residual_quantise",
    "tokens_per_second",
    "rnnt_loss",
    "rnnt_loss_bruteforce",
    "num_alignments",
    "StatelessPredictor",
    "Joiner",
    "TinyTransducer",
    "ctc_greedy_decode",
    "ctc_greedy_decode_buggy",
    "ctc_loss_forward",
    "Endpointer",
    "EndpointConfig",
    "TurnState",
    "emission_delays",
    "awed",
    "effective_latency",
    "user_perceived_turn_latency",
]
