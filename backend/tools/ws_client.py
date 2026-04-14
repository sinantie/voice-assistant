import asyncio
import websockets
import numpy as np
import soundfile as sf
import json
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError
from pathlib import Path

# ---------------- config ----------------

WS_URL = "ws://127.0.0.1:8000/ws"

INPUT_WAV = Path("samples/capital_of_greece.wav")     # must be mono PCM16 24kHz
OUTPUT_WAV = Path("reply.wav")

SAMPLE_RATE = 24000
FRAME_MS = 10
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
FRAME_BYTES = FRAME_SAMPLES * 2  # int16

# ---------------------------------------

response_done = False

async def stream_audio(ws, samples: np.ndarray):
    """
    Stream PCM16 samples in real time.
    """
    print("▶ streaming audio")

    idx = 0
    total = len(samples)

    while idx + FRAME_SAMPLES <= total:
        frame = samples[idx : idx + FRAME_SAMPLES]
        await ws.send(frame.tobytes())

        idx += FRAME_SAMPLES
        await asyncio.sleep(FRAME_MS / 1000.0)

    print("■ audio stream finished")


async def receive_messages(ws, tts_buffer):
    global response_done

    try:
        async for msg in ws:
            if isinstance(msg, bytes):
                pcm = np.frombuffer(msg, dtype=np.int16)
                tts_buffer.append(pcm)
                print(f"🔊 received audio chunk ({len(pcm)} samples)")

            else:
                data = json.loads(msg)
                evt = data.get("type")

                if evt == "vad_start":
                    print("🟢 VAD start")

                elif evt == "vad_end":
                    print("🔴 VAD end")
                    response_done = True

                elif evt == "transcript":
                    print(f"📝 transcript: {data['text']}")

                else:
                    print(f"ℹ️ event: {data}")

    except ConnectionClosedOK:
        print("✅ WebSocket closed normally")

    except ConnectionClosedError:
        print("ℹ️ WebSocket closed without close frame (normal)")


async def main():
    # -------- load input wav --------
    samples, sr = sf.read(INPUT_WAV, dtype="int16")

    assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE} Hz, got {sr}"
    assert samples.ndim == 1, "Input WAV must be mono"

    print(f"Loaded {INPUT_WAV} ({len(samples)/sr:.2f}s)")

    # -------- connect --------
    async with websockets.connect(WS_URL, max_size=None) as ws:
        print("✅ WebSocket connected")

        tts_chunks = []

        receiver = asyncio.create_task(
            receive_messages(ws, tts_chunks)
        )

        await stream_audio(ws, samples)

        # tell backend we are done sending audio
        await ws.send(json.dumps({"type": "end_of_stream"}))

        # # give backend time to respond
        # await asyncio.sleep(2.0)

        # wait until backend finishes responding
        while not response_done:
            await asyncio.sleep(0.05)

        print("response received, closing socket")
        await ws.close()

        await receiver

    # -------- save TTS --------
    if tts_chunks:
        tts = np.concatenate(tts_chunks)
        sf.write(
            OUTPUT_WAV,
            tts,
            SAMPLE_RATE,
            subtype="PCM_16",
        )
        print(f"✅ saved TTS to {OUTPUT_WAV}")
    else:
        print("⚠️ no TTS audio received")


if __name__ == "__main__":
    asyncio.run(main())