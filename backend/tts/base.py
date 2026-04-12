import os

_BACKEND = os.getenv("TTS_BACKEND", "melo").lower()

if _BACKEND == "melo":
    from .melo import synthesize
elif _BACKEND == "pocket":
    from .pocket import synthesize
else:
    raise RuntimeError(f"Unknown TTS_BACKEND: {_BACKEND}")