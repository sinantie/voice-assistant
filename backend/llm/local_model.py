import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = os.getenv("LLM_MODEL", "Qwen/Qwen2-1.5B-Instruct")  # works on ZeroGPU

_tokenizer = None
_model = None

SYSTEM_PROMPT = (
    "You are a voice assistant. "
    "Reply in one short sentence. "
    "Do not explain unless asked."
)

def _load():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
        )

def run_llm(user_text: str) -> str:
    _load()

    prompt = f"{SYSTEM_PROMPT}\nUser: {user_text}\nAssistant:"
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)

    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.2,
            do_sample=True,
        )

    text = _tokenizer.decode(out[0], skip_special_tokens=True)
    return text.split("Assistant:")[-1].strip()