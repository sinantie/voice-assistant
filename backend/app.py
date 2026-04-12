import logging
import time
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
import numpy as np
import math

from audio import load_wav, write_wav
from models import get_stt_model
from tts import synthesize

app = FastAPI(title="ZeroGPU Voice Assistant (v0)")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("voice-assistant")

class Timer:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start

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
    req_start = time.perf_counter()       
    # MIME types from curl / ESP32 are unreliable → trust bytes
    audio_bytes = await audio.read()
    samples = load_wav(audio_bytes)
    logger.info(
            f"Request received | input_samples={len(samples)} | lang={lang}"
        )

    # --- STT ---
    with Timer("stt") as t_stt:
        model = get_stt_model()

        segments, info = model.transcribe(
            samples,
            language=lang,
            beam_size=5,
        )

    text = "".join(segment.text for segment in segments).strip()
    logger.info(
            f"STT done | time={t_stt.elapsed:.3f}s | text_len={len(text)}"
        )    

    # --- TTS ---    
    if not text:
        text = "I did not hear anything."
    
    with Timer("tts") as t_tts:
        tts_audio, tts_sample_rate = synthesize(text)
    duration = len(tts_audio) / tts_sample_rate

    logger.info(
            f"TTS done | time={t_tts.elapsed:.3f}s | "
            f"sr={tts_sample_rate} | duration={duration:.2f}s"
        )

    wav_bytes = write_wav(tts_audio, sample_rate=tts_sample_rate)

    total_time = time.perf_counter() - req_start
    logger.info(
        f"Request complete | total_time={total_time:.3f}s"
    )


    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Transcript": text,
            "X-Language": info.language,
        },
    )