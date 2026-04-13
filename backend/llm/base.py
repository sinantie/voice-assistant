import os

_BACKEND = os.getenv("LLM_BACKEND", "llama_cpp").lower()

if _BACKEND == "llama_cpp":
    from .llama_cpp import run_llm
elif _BACKEND == "local_model":
    from .local_model import run_llm
else:
    raise RuntimeError(f"Unknown LLM_BACKEND: {_BACKEND}")