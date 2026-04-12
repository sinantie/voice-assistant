from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
import numpy as np
import math

from audio import load_wav, write_wav
from models import get_stt_model
from tts import synthesize

app = FastAPI(title="ZeroGPU Voice Assistant (v0)")


def generate_beep(
    duration_s: float = 0.5,
    freq_hz: float = 440.0,
    sample_rate: int = 16000,
):
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    waveform = 0.3 * np.sin(2 * math.pi * freq_hz * t)
    return np.int16(waveform * 32767)


@app.post("/speech_to_speech")
async def speech_to_speech(
    audio: UploadFile = File(...),
    lang: str | None = "en",
):
    # MIME types from curl / ESP32 are unreliable → trust bytes
    audio_bytes = await audio.read()
    samples = load_wav(audio_bytes)

    model = get_stt_model()

    segments, info = model.transcribe(
        samples,
        language=lang,
        beam_size=5,
    )

    text = "".join(segment.text for segment in segments).strip()

    # --- Real TTS ---    
    if not text:
            text = "I did not hear anything."
    tts_audio, tts_sample_rate = synthesize(text)
    wav_bytes = write_wav(tts_audio, sample_rate=tts_sample_rate)

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Transcript": text,
            "X-Language": info.language,
        },
    )