import os
import requests
import logging

logger = logging.getLogger("voice-assistant")

MODEL_NAME = os.getenv("LLM_MODEL", "Qwen/Qwen2-1.5B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
}


ROUTER_URL = "https://router.huggingface.com/v1/chat/completions"


def _call_hf(model: str, user_text: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a voice assistant. "
                    "Reply in one short sentence. "
                    "Do not explain unless asked."
                ),
            },
            {
                "role": "user",
                "content": user_text,
            },
        ],
        "max_tokens": 64,
        "temperature": 0.2,
    }

    r = requests.post(
        ROUTER_URL,
        headers=HEADERS,
        json=payload,
        timeout=30,
    )

    r.raise_for_status()
    data = r.json()

    return data["choices"][0]["message"]["content"].strip()


def run_llm(user_text: str) -> str:
    try:
        return _call_hf(MODEL_NAME, user_text)
    except requests.HTTPError as e:
        logger.warning(f"LLM failed ({e}), falling back")
        return _call_hf("Qwen/Qwen2-1.5B-Instruct", user_text)
