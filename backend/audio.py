import io
import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
from fastapi import HTTPException


EXPECTED_SR = 16000


def load_wav(file_bytes: bytes) -> np.ndarray:
    """
    Load a WAV file from bytes.
    Enforces: mono, 16 kHz, float32.
    """
    try:
        audio, sr = sf.read(io.BytesIO(file_bytes), dtype="float32")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid WAV file: {e}")

    if sr != EXPECTED_SR:
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
    if sample_rate != EXPECTED_SR:
        print(f"Resampling from {sample_rate} Hz to {EXPECTED_SR} Hz")

        print(
            EXPECTED_SR, sample_rate
        )

        samples = resample_poly(
            samples,
            EXPECTED_SR,
            sample_rate,
        )
        samples = np.clip(samples, -1.0, 1.0)
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