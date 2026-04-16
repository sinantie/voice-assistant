import os
import logging
import numpy as np
from fastapi import WebSocket
from audio import resample_audio, write_wav
from tts.pocket import synthesize
from vad import is_speech  # expects 16 kHz float32
import soundfile as sf

from collections import deque
import json

logger = logging.getLogger("voice-assistant")

# ---------- audio constants ----------

STREAM_SR = int(os.getenv("SAMPLE_RATE", 24000))          # incoming audio
VAD_SR = 16000             # silero requirement

FRAME_MS = 10              # ESP32 / client frame size
FRAME_SAMPLES = STREAM_SR * FRAME_MS // 1000


# Silero constraints

VAD_FRAME_SAMPLES = 512            # REQUIRED by Silero
# Hysteresis
SPEECH_FRAMES_START = 2            # ~64 ms
SPEECH_FRAMES_END = 6              # ~192 ms


# ---------- websocket handler ----------

async def handle_utterance(ws, utterance: np.ndarray, connected: bool):
    if not connected:
        logger.info("Skipping utterance send (socket closed)")
        return

    duration = len(utterance) / STREAM_SR
    logger.info(f"Utterance finalized: {duration:.2f}s")

    # ---- STT (stub for now) ----
    text = "The capital of Greece is Athens."

    await ws.send_json({
        "type": "transcript",
        "text": text,
    })

    # ---- TTS ----
    audio, sr = synthesize(text)  # sr should be 24000
    pcm16 = (audio * 32767.0).astype(np.int16) # convert to PCM16
    FRAME_SAMPLES = 240  # 10 ms @ 24kHz
    idx = 0
    while idx + FRAME_SAMPLES <= len(pcm16):
        frame = pcm16[idx : idx + FRAME_SAMPLES]
        await ws.send_bytes(frame.tobytes())
        idx += FRAME_SAMPLES


async def websocket_handler(ws: WebSocket):
    connected = True
    await ws.accept()
    logger.info("WebSocket connected")

    vad_buffer = np.zeros(0, dtype=np.float32)
    speech_buffer = []
    PRE_SPEECH_FRAMES = 5  # 5 × 10 ms = 50 ms pre-roll
    pre_buffer = deque(maxlen=PRE_SPEECH_FRAMES)

    speech_count = 0
    silence_count = 0
    in_speech = False

    try:
        while True:
            msg = await ws.receive()

            if msg["type"] == "websocket.disconnect":
                connected = False
                break

            # ---- explicit end-of-stream control message ----
            if msg["type"] == "websocket.receive" and msg.get("text"):
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue

                if data.get("type") == "end_of_stream":
                    logger.info("Received end_of_stream from client")
                    connected = False
                    break

            if msg["type"] == "websocket.receive" and msg.get("bytes"):
                # ---- decode 24 kHz PCM ----
                pcm16 = np.frombuffer(msg["bytes"], dtype=np.int16)
                audio = pcm16.astype(np.float32) / 32768.0

                # ---- pre-roll buffer for VAD ----
                pre_buffer.append(audio)

                
                # ---- VAD path ----
                audio_16k = resample_audio(audio, STREAM_SR, VAD_SR)
                vad_buffer = np.concatenate([vad_buffer, audio_16k])

                # ---- process exact 512-sample frames ----
                while len(vad_buffer) >= VAD_FRAME_SAMPLES:
                    frame = vad_buffer[:VAD_FRAME_SAMPLES]
                    vad_buffer = vad_buffer[VAD_FRAME_SAMPLES:]

                    voiced = is_speech(frame, sr=VAD_SR)

                    if voiced:
                        speech_count += 1
                        silence_count = 0
                    else:
                        silence_count += 1
                        speech_count = 0

                    # ---- speech start ----
                    if speech_count >= SPEECH_FRAMES_START and not in_speech:
                        in_speech = True
                        speech_buffer.clear()

                        # inject pre-roll BEFORE first speech frame
                        speech_buffer.extend(pre_buffer)
                        pre_buffer.clear()

                        await ws.send_json({"type": "vad_start"})
                        logger.info("VAD start")

                    # ---- collect speech audio ----
                    if in_speech:
                        speech_buffer.append(audio)

                    # ---- speech end ----
                    if silence_count >= SPEECH_FRAMES_END and in_speech:
                        in_speech = False

                        if connected:
                            await ws.send_json({"type": "vad_end"})

                        logger.info("VAD end")

                        utterance = np.concatenate(speech_buffer)
                        await handle_utterance(ws, utterance, connected)

                        speech_buffer.clear()

    except Exception as e:
        logger.exception(e)

    finally:        
        
        if in_speech and speech_buffer:
            logger.info("End-of-stream flush")

            utterance = np.concatenate(speech_buffer)

            if connected:
                await ws.send_json({"type": "vad_end"})
                await handle_utterance(ws, utterance, connected)
            else:
                logger.info("Socket closed; skipping WS send on flush")

            speech_buffer.clear()
        logger.info("WebSocket handler finished")

