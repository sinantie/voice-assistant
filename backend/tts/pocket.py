import torch
import numpy as np
from pocket_tts import TTSModel  # adjust if path differs

_model = None
_voice_state = None


def _get_model():
    global _model, _voice_state
    # if _model is None:
    #     device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = TTSModel.load_model()
    _voice_state = _model.get_state_for_audio_prompt(
        "alba"  # One of the pre-made voices, see above
        # You can also use any voice file you have locally or from Hugging Face:
        # "./some_audio.wav"
        # or "hf://kyutai/tts-voices/expresso/ex01-ex02_default_001_channel2_198s.wav"
    )
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