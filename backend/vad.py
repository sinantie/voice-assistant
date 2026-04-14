import torch
import numpy as np

_model = None


def load_vad():
    global _model
    if _model is None:
        _model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
    return _model


def is_speech(samples: np.ndarray, sr: int = 24000) -> bool:
    """
    samples: float32 mono audio
    """
    model = load_vad()

    with torch.no_grad():
        speech_prob = model(
            torch.from_numpy(samples),
            sr,
        ).item()

    return speech_prob > 0.5