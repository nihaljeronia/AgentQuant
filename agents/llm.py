"""
Shared OpenAI access for all agents.

- Clients are created lazily, so importing the project never crashes when the key
  is missing (the dashboard can start and explain what to do).
- Every call asks for JSON, strips code fences, and retries on API or parse errors.
"""
import json
import os
import sys
import time
from functools import lru_cache

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import has_openai_key  # noqa: E402


class LLMError(RuntimeError):
    """Raised when the model cannot be reached or keeps returning invalid JSON."""


@lru_cache(maxsize=None)
def get_llm(model: str, temperature: float):
    if not has_openai_key():
        raise LLMError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        model_kwargs={"response_format": {"type": "json_object"}},
        timeout=60,
        max_retries=2,
    )


def _parse_json(text: str) -> dict:
    text = str(text).strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Model returned JSON that is not an object")
    return data


def invoke_json(model: str, temperature: float, prompt: str, attempts: int = 3) -> dict:
    """Send one prompt and return the parsed JSON object, retrying on failure."""
    from langchain_core.messages import HumanMessage

    llm = get_llm(model, temperature)
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            return _parse_json(response.content)
        except Exception as e:  # API error, timeout or bad JSON
            last_error = e
            if attempt < attempts - 1:
                time.sleep(1.5 * (attempt + 1))
    raise LLMError(f"{model} failed after {attempts} attempts: {last_error}")
