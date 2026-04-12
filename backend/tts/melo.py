import torch
import numpy as np
from melo.api import TTS

_model = None


def _get_model(lang: str = "en"):
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = TTS(language=lang, device=device)
    return _model


def synthesize(text: str) -> np.ndarray:
    """
    Returns PCM16 mono @ 16 kHz
    """
    model = _get_model()

    audio = model.tts_to_numpy(text)

    # Melo returns float32 [-1, 1]
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767).astype(np.int16)