import torch
import numpy as np
from melo.api import TTS

_tts_model = None


def get_tts_model(lang: str = "en"):
    global _tts_model

    if _tts_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        _tts_model = TTS(
            language=lang,
            device=device,
        )

    return _tts_model


def synthesize(text: str, speaker_id: int = 0):
    """
    Returns PCM16 mono audio at 16 kHz
    """
    model = get_tts_model()

    audio = model.tts_to_numpy(
        text,
        speaker_id=speaker_id,
    )

    # Melo returns float32 in [-1, 1]
    pcm16 = np.int16(audio * 32767)
    return pcm16``