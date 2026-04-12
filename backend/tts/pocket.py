import torch
import numpy as np
from pocket_tts.api import PocketTTS  # adjust if path differs

_model = None


def _get_model():
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = PocketTTS(device=device)
    return _model


def synthesize(text: str) -> np.ndarray:
    """
    Returns PCM16 mono @ 16 kHz
    """
    model = _get_model()

    audio = model.synthesize(text)  # expected float32 [-1, 1]

    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767).astype(np.int16)