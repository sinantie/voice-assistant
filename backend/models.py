import torch
from faster_whisper import WhisperModel

_stt_model = None


def select_device():
    """
    Device selection logic:
    - CUDA if available (ZeroGPU, desktop GPU)
    - CPU fallback (including Apple Silicon)
    """
    if torch.cuda.is_available():
        return "cuda", "float16"

    # MPS detected but not used by faster-whisper
    if torch.backends.mps.is_available():
        return "cpu", "int8"

    return "cpu", "int8"


def get_stt_model():
    global _stt_model

    if _stt_model is None:
        device, compute_type = select_device()

        _stt_model = WhisperModel(
            "tiny.en",
            device=device,
            compute_type=compute_type,
        )

    return _stt_model
