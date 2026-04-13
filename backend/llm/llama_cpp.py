import requests

LLAMA_URL = "http://127.0.0.1:8080/v1/chat/completions"

# SYSTEM_PROMPT = (
#     "You are a voice assistant. "
#     "Reply in one short sentence. "
#     "Do not explain unless asked."
# )

SYSTEM_PROMPT = (
    "You are a voice assistant. "
    "Reply with only the final answer. "
    "Do not show reasoning. "
    "Do not explain unless asked."
)

def run_llm(user_text: str) -> str:
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        "max_tokens": 64,
        "temperature": 0.2,
        "stop": ["Thought:", "Reasoning:", "<thinking>", "Let's think"],
    }

    r = requests.post(LLAMA_URL, json=payload, timeout=10)
    r.raise_for_status()
    data = r.json()

    return data["choices"][0]["message"]["content"].strip()