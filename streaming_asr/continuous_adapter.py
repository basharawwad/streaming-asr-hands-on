"""Getting audio into an LLM: continuous embeddings versus discrete tokens.

There is no such thing as a continuous *token*. An LLM's embedding table is a
lookup that turns an integer into a vector, so to use continuous audio you skip
the lookup and hand the model vectors directly. Every framework exposes this as
`inputs_embeds` in place of `input_ids`.

The choice between the two representations is forced by the *output* side.

    continuous  no quantisation loss, one linear layer to train, and no codec.
                But you cannot sample the next audio step from a real-valued
                vector with a softmax, so the model can only emit text.

    discrete    a categorical distribution you can sample autoregressively,
                which is the whole reason codecs exist. You pay quantisation
                error before the LLM sees anything, and a codec is its own
                training stage.

Hence the standard asymmetry in systems that both understand and speak:
continuous in, discrete out.

Everything awkward about codecs follows from the sampling requirement. A single
vector quantiser at a usable bitrate reconstructs badly, so you stack residual
stages; stacking stages multiplies the tokens per frame; and multiplying tokens
per frame forces either a longer sequence or a second transformer over the
codebook axis. `residual_quantise` and `tokens_per_second` below make that
chain measurable.

See `docs/how-transcriptions-are-used.md` for how the transcript becomes a
training target once the audio is spliced in.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = -100


# --------------------------------------------------------------------------
# Downsampling
# --------------------------------------------------------------------------


class FrameStacker(nn.Module):
    """Concatenate `k` adjacent encoder frames into one.

    The cheapest downsampler that works, and the one most LLM-based ASR systems
    use. A Whisper-style encoder runs at 50 Hz, so thirty seconds of audio is
    1500 frames and would eat most of a short context window. Stacking five
    frames turns that into 300 at 10 Hz, with no information discarded: the
    channels grow by the same factor the time axis shrinks.

    The trailing frames that do not fill a group are dropped, which is why the
    streaming path needs a buffer: a chunk boundary rarely lands on a multiple
    of `k`.
    """

    def __init__(self, k: int):
        super().__init__()
        assert k >= 1
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T // k, D * k)"""
        B, T, D = x.shape
        usable = (T // self.k) * self.k
        return x[:, :usable, :].reshape(B, usable // self.k, D * self.k)

    def init_state(self, batch_size: int, dim: int, device=None, dtype=None):
        return torch.zeros(batch_size, 0, dim, device=device, dtype=dtype)

    def step(self, x_chunk: torch.Tensor, state: torch.Tensor):
        """Stack across chunk boundaries by carrying the remainder forward.

        Returns (stacked, new_state). `stacked` may be empty when the chunk did
        not complete a group, which is normal and the caller must handle it.
        """
        buffered = torch.cat([state, x_chunk], dim=1)
        B, T, D = buffered.shape
        usable = (T // self.k) * self.k
        out = buffered[:, :usable, :].reshape(B, usable // self.k, D * self.k)
        return out, buffered[:, usable:, :]


# --------------------------------------------------------------------------
# Projection into the LLM's embedding space
# --------------------------------------------------------------------------


class AudioProjector(nn.Module):
    """Map encoder frames into the language model's embedding space.

    SLAM-ASR's result is that a single linear layer is enough, with the encoder
    and the LLM both frozen. The elaborate adapters were not buying much.

    The `LayerNorm` and the learned `scale` exist for a failure mode that is
    easy to miss: the LLM's embedding table has a characteristic row norm, and
    if the projector's output sits at a very different magnitude the model
    either ignores the audio or training destabilises. Calibrating the scale
    once against the table costs nothing and removes the problem.
    """

    def __init__(self, in_dim: int, out_dim: int, normalise: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim) if normalise else nn.Identity()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(x)) * self.scale

    @torch.no_grad()
    def calibrate_scale(
        self, sample: torch.Tensor, embedding: nn.Embedding
    ) -> float:
        """Set `scale` so the mean output row norm matches the embedding table.

        Call once with a representative batch of encoder frames before
        training. Returns the scale it chose, which is a useful thing to log:
        a value far from 1 tells you the encoder and the LLM were living at
        very different magnitudes.
        """
        target = embedding.weight.norm(dim=-1).mean()
        self.scale.fill_(1.0)
        current = self(sample).reshape(-1, self.linear.out_features).norm(dim=-1).mean()
        chosen = float(target / current.clamp_min(1e-9))
        self.scale.fill_(chosen)
        return chosen


def mean_row_norm(x: torch.Tensor) -> float:
    """Mean L2 norm over the last axis. The quantity to match across a splice."""
    return float(x.detach().reshape(-1, x.shape[-1]).norm(dim=-1).mean())


# --------------------------------------------------------------------------
# The splice
# --------------------------------------------------------------------------


class PlaceholderCountMismatch(ValueError):
    """Raised when the prompt's audio placeholders do not match the frames.

    Worth its own exception type because it is the single most common bug in
    this pattern and it is otherwise reported as an opaque shape error from a
    masked assignment, or worse, silently misaligns when the counts happen to
    be compatible.
    """


def splice_audio_embeddings(
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    audio_embeds: torch.Tensor,
    placeholder_id: int,
) -> torch.Tensor:
    """Replace the placeholder rows of a text embedding sequence with audio.

    Args:
        inputs_embeds: (B, L, D) from the LLM's own embedding table.
        input_ids: (B, L) the tokenised prompt, containing `placeholder_id`
            once per audio frame.
        audio_embeds: (B, T, D) projected encoder output.
        placeholder_id: the id reserved for audio positions.

    Returns a new (B, L, D) tensor. The text rows are untouched, so the LLM
    sees one ordinary sequence and neither the attention mask nor the position
    ids need special handling.
    """
    if inputs_embeds.shape[-1] != audio_embeds.shape[-1]:
        raise ValueError(
            f"audio dim {audio_embeds.shape[-1]} does not match model dim "
            f"{inputs_embeds.shape[-1]}; the projector is misconfigured"
        )

    mask = input_ids == placeholder_id
    slots = int(mask.sum().item())
    frames = audio_embeds.shape[0] * audio_embeds.shape[1]
    if slots != frames:
        raise PlaceholderCountMismatch(
            f"{slots} placeholder tokens but {frames} audio frames. "
            "The prompt must carry exactly one placeholder per frame after "
            "downsampling; check the frame stacker's k and its dropped "
            "remainder."
        )

    out = inputs_embeds.clone()
    out[mask] = audio_embeds.reshape(-1, audio_embeds.shape[-1]).to(out.dtype)
    return out


def build_labels(
    input_ids: torch.Tensor, placeholder_id: int, ignore_index: int = IGNORE_INDEX
) -> torch.Tensor:
    """Mask the audio positions out of the loss.

    Without this the model is trained to predict the placeholder id at audio
    positions, which wastes capacity on a meaningless target and presents as
    slow convergence rather than as a bug.
    """
    labels = input_ids.clone()
    labels[labels == placeholder_id] = ignore_index
    return labels


# --------------------------------------------------------------------------
# A stand-in language model, so the tests need no download
# --------------------------------------------------------------------------


class TinyLM(nn.Module):
    """Minimal causal LM with the interface the splice pattern relies on.

    Four properties matter and this has all of them: an embedding table you can
    read the scale off, a forward that accepts `inputs_embeds` in place of
    `input_ids`, **causal self-attention**, and a shifted loss that honours an
    ignore index.

    The attention is not decoration. It is the only path by which a text
    position can see the audio, so without it the gradient to the projector is
    exactly zero and the whole pattern is inert. If you are debugging a speech
    adapter that will not learn, check that the text positions actually attend
    to the audio positions before you look anywhere else.

    The loss is shifted the way every causal LM shifts it: the prediction made
    at position i is scored against the token at position i+1. So the first
    transcript token is predicted at the *last audio position*, which is
    precisely where the audio has to have been understood.
    """

    def __init__(self, vocab_size: int = 64, d_model: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("pass input_ids or inputs_embeds")
            inputs_embeds = self.embed(input_ids)

        B, L, D = inputs_embeds.shape
        h = self.norm(inputs_embeds)
        q, k, v = self.q_proj(h), self.k_proj(h), self.v_proj(h)

        scores = (q @ k.transpose(-2, -1)) / (D ** 0.5)
        idx = torch.arange(L, device=inputs_embeds.device)
        causal = idx.view(-1, 1) >= idx.view(1, -1)
        scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
        attended = scores.softmax(dim=-1) @ v

        x = inputs_embeds + self.o_proj(attended)
        logits = self.head(x)

        loss = None
        if labels is not None:
            # The standard shift: position i predicts the token at i+1.
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, self.vocab_size),
                labels[:, 1:].reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
        return logits, loss


# --------------------------------------------------------------------------
# The discrete alternative, for comparison
# --------------------------------------------------------------------------


class ToyVectorQuantiser(nn.Module):
    """Nearest-neighbour vector quantiser. Lossy by construction.

    Stands in for a codec's quantiser so the information cost of going discrete
    is measurable rather than asserted. Real codecs train the codebook jointly
    with an encoder and a decoder and distil a self-supervised model into the
    first stage to make it semantic; none of that removes the quantisation
    error, it only spends it better.
    """

    def __init__(self, dim: int, codebook_size: int, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer(
            "codebook", torch.randn(codebook_size, dim, generator=g)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, D) -> (ids (B, T), quantised (B, T, D))"""
        dist = (x.unsqueeze(-2) - self.codebook).pow(2).sum(-1)   # (B, T, K)
        ids = dist.argmin(-1)
        return ids, self.codebook[ids]


def residual_quantise(
    x: torch.Tensor, stages: list[ToyVectorQuantiser]
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Quantise the residual left by each previous stage.

    This is why a codec emits several tokens per frame. One stage at a
    practical codebook size reconstructs poorly; each extra stage codes what
    the previous ones missed, and the reconstruction error falls monotonically.
    The cost is one more token per frame, every frame.
    """
    residual = x
    reconstruction = torch.zeros_like(x)
    all_ids: list[torch.Tensor] = []
    for q in stages:
        ids, quantised = q(residual)
        all_ids.append(ids)
        reconstruction = reconstruction + quantised
        residual = residual - quantised
    return all_ids, reconstruction


def reconstruction_error(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """Mean squared error. Zero for continuous embeddings, by definition."""
    return float((x - x_hat).pow(2).mean())


def tokens_per_second(
    frame_rate_hz: float,
    num_codebooks: int = 1,
    flatten_codebooks: bool = True,
) -> float:
    """Sequence positions per second of audio, which sets the KV cache cost.

    A continuous path calls this with `num_codebooks=1`: one position per
    frame, and the frame rate is yours to choose in the downsampler.

    A discrete path multiplies by the codebooks if it flattens them into the
    time axis. Factorising them instead, with a small transformer over the
    codebook axis at each timestep, keeps the sequence at the frame rate and is
    the reason that design exists.
    """
    return frame_rate_hz * (num_codebooks if flatten_codebooks else 1)
