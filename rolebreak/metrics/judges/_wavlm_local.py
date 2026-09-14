"""s3prl-free WavLM-large upstream for the speaker-verification judge.

seed-tts-eval's ECAPA-TDNN head extracts features from WavLM-large via
``torch.hub.load('s3prl/s3prl', 'wavlm_large')`` — which pulls in ``s3prl`` and
``fairseq`` just to download and wrap the upstream.  But the fine-tuned
checkpoint (``model.pth``) already ships the *complete* WavLM-large weights
under ``feature_extract.model.*``, and this repo vendors Microsoft's own WavLM
implementation (``…/UniSpeech/WavLM/WavLM.py``), which depends only on
``torch``/``numpy``.

This module rebuilds that upstream from the vendored code and reproduces the
exact tensor s3prl's wrapper returns: a ``"hidden_states"`` list of 25 layer
representations (input to each of the 24 transformer layers, then the encoder
output), which the ECAPA head soft-weights with its learned ``feature_weight``.
Reproducing that list — same contents, same order — is what makes the loaded
``feature_weight`` (and therefore the SIM score) identical to seed-tts-eval's.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

# Vendored seed-tts-eval tree.
_SV_DIR = (
    Path(__file__).resolve().parents[3] / "external/seed-tts-eval/thirdparty/UniSpeech/downstreams/speaker_verification"
)
_WAVLM_DIR = Path(__file__).resolve().parents[3] / "external/seed-tts-eval/thirdparty/UniSpeech/WavLM"

# WavLM-large architecture (the cfg s3prl would have loaded from the upstream
# checkpoint).  No weights depend on these *except* the relative-position
# bucketing, so ``max_distance`` matters for numeric parity — WavLM-large uses
# 800, not the WavLMConfig default of 1280.
_WAVLM_LARGE_CFG = {
    "extractor_mode": "layer_norm",
    "encoder_layers": 24,
    "encoder_embed_dim": 1024,
    "encoder_ffn_embed_dim": 4096,
    "encoder_attention_heads": 16,
    "activation_fn": "gelu",
    "layer_norm_first": True,
    "conv_feature_layers": "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
    "conv_bias": False,
    "feature_grad_mult": 0.0,  # inference-only; forward runs under no_grad anyway
    "normalize": True,
    "dropout": 0.0,
    "attention_dropout": 0.0,
    "activation_dropout": 0.0,
    "encoder_layerdrop": 0.0,
    "dropout_input": 0.0,
    "dropout_features": 0.0,
    "conv_pos": 128,
    "conv_pos_groups": 16,
    "relative_position_embedding": True,
    "num_buckets": 320,
    "max_distance": 800,
    "gru_rel_pos": True,
}


def _import_wavlm() -> tuple[type, type]:
    """Import the vendored ``WavLM`` / ``WavLMConfig`` (adds its dir to sys.path)."""
    if str(_WAVLM_DIR) not in sys.path:
        sys.path.insert(0, str(_WAVLM_DIR))  # WavLM.py does a bare ``from modules import ...``
    from WavLM import WavLM, WavLMConfig  # type: ignore[import-not-found]

    return WavLM, WavLMConfig


class WavLMLargeUpstream(nn.Module):
    """Drop-in replacement for s3prl's ``wavlm_large`` upstream.

    ``forward(wavs)`` takes a list of 1-D waveform tensors and returns
    ``{"hidden_states": [25 × (B, T, 1024)]}`` — matching what the ECAPA head's
    ``get_feat`` expects from ``feature_selection="hidden_states"``.
    """

    def __init__(self) -> None:
        super().__init__()
        WavLM, WavLMConfig = _import_wavlm()
        cfg = WavLMConfig(_WAVLM_LARGE_CFG)
        self.cfg = cfg
        self.model = WavLM(cfg)

        # Reproduce s3prl's UpstreamBase hooks: capture the input to every
        # transformer layer, then the encoder's output.  Fire order is
        # layer0…layer23 (during the encoder loop) then the encoder itself,
        # giving the 25 hidden states in the order the head was trained on.
        self._hidden: list[torch.Tensor] = []
        for layer in self.model.encoder.layers:
            layer.register_forward_hook(self._layer_hook)
        self.model.encoder.register_forward_hook(self._encoder_hook)

    def _layer_hook(self, _module: nn.Module, inp: tuple, _out: Any) -> None:
        # Layer input x is (T, B, C); s3prl stores it transposed to (B, T, C).
        self._hidden.append(inp[0].transpose(0, 1))

    def _encoder_hook(self, _module: nn.Module, _inp: tuple, out: Any) -> None:
        # Encoder returns (x, layer_results); x is already (B, T, C).
        self._hidden.append(out[0])

    def forward(self, wavs: list[torch.Tensor]) -> dict[str, list[torch.Tensor]]:
        if self.cfg.normalize:
            wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]
        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(int(wav_lengths.max())).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        self._hidden = []
        self.model.extract_features(padded_wav, padding_mask=wav_padding_mask, mask=False)
        hidden = self._hidden
        self._hidden = []
        return {"hidden_states": hidden, "default": hidden[-1]}


def build_wavlm_sv_model(checkpoint: str | Path, device: str = "cpu") -> nn.Module:
    """Build the seed-tts-eval ECAPA-TDNN SV model without s3prl/fairseq.

    The ECAPA constructor would normally call ``torch.hub.load('s3prl/s3prl',
    'wavlm_large')`` to create its feature extractor; we temporarily patch that
    to return :class:`WavLMLargeUpstream` instead, then load the full fine-tuned
    state dict (head + WavLM).  We load ``strict=True`` after dropping the
    training-only ``loss_calculator`` head — so a mismatch in any *inference*
    weight (a wrong cfg, a renamed layer) is caught loudly rather than silently
    skipped the way the upstream's ``strict=False`` would.
    """
    if str(_SV_DIR) not in sys.path:
        sys.path.insert(0, str(_SV_DIR))
    from models.ecapa_tdnn import ECAPA_TDNN_SMALL  # type: ignore[import-not-found]

    orig_hub_load = torch.hub.load
    torch.hub.load = lambda *a, **k: WavLMLargeUpstream()  # type: ignore[assignment]
    try:
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
    finally:
        torch.hub.load = orig_hub_load  # type: ignore[assignment]

    state = {
        k: v
        for k, v in torch.load(checkpoint, map_location="cpu")["model"].items()
        if not k.startswith("loss_calculator.")
    }
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)
