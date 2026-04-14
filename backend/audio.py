import io
import os
import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
from fastapi import HTTPException


EXPECTED_SR = int(os.getenv("SAMPLE_RATE", 24000))


def load_wav(file_bytes: bytes, enforce_sr: bool = False) -> np.ndarray:
    """
    Load a WAV file from bytes.
    Enforces: mono, float32.
    """
    try:
        audio, sr = sf.read(io.BytesIO(file_bytes), dtype="float32")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid WAV file: {e}")

    if sr != EXPECTED_SR:
        if enforce_sr:
            raise HTTPException(
                status_code=400,
                detail=f"Sample rate must be {EXPECTED_SR} Hz",
            )

    if audio.ndim != 1:
        raise HTTPException(
            status_code=400,
            detail="Audio must be mono",
        )

    return audio


def write_wav(
    samples: np.ndarray,
    sample_rate: int = EXPECTED_SR,
) -> bytes:
    """
    Encode mono PCM16 WAV and return bytes.
    """
    buf = io.BytesIO()

    # resample if necessary
    samples = resample_audio(samples, sample_rate, EXPECTED_SR)

    samples = (samples * 32767.0).astype(np.int16)


    sf.write(
        buf,
        samples,
        samplerate=EXPECTED_SR,
        subtype="PCM_16",
        format="WAV",
    )
    buf.seek(0)
    return buf.read()

def resample_audio(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return samples

    print(f"Resampling from {orig_sr} Hz to {target_sr} Hz")
    resampled = resample_poly(samples, target_sr, orig_sr)
    return np.clip(resampled, -1.0, 1.0)