import os

import logging
import numpy as np
from pocket_tts import TTSModel, export_model_state  # adjust if path differs

_model = None
_voice_state = None

logger = logging.getLogger("voice-assistant")

def _get_model():
    global _model, _voice_state
    # if _model is None:
    #     device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = TTSModel.load_model()
    if os.path.exists("./alba.safetensors"):
        logger.info("Loading voice state from cache")
        _voice_state = _model.get_state_for_audio_prompt("./alba.safetensors")
    else:
        _voice_state = _model.get_state_for_audio_prompt(
            "alba" 
        )
        export_model_state(_voice_state, "./alba.safetensors")
    return _model


def synthesize(text: str) -> tuple[np.ndarray, int]:
    """
    Returns PCM16 mono @ 24 kHz
    """
    model = _get_model()

    audio = model.generate_audio(_voice_state, text)  # expected float32 [-1, 1]
    
    audio = audio.numpy()
    audio = np.clip(audio, -1.0, 1.0)
    return audio, 24000  # Pocket TTS outputs at 24 kHz, resampling might be needed in audio.py