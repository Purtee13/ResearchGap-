"""Thin wrapper around Gemma 4 on the Gemini API that returns validated Pydantic objects.

Gemma models on the Gemini API do not take system instructions or a response
schema, so the schema goes into the prompt and the reply is parsed, validated,
and retried with the validation error if it does not fit.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .config import Settings

T = TypeVar("T", bound=BaseModel)


@dataclass
class ImagePart:
    label: str  # e.g. "Paper A, page 4 (Figure 2: accuracy by codebase)"
    data: bytes
    mime_type: str = "image/jpeg"


class LLMError(RuntimeError):
    pass


def extract_json(text: str) -> str:
    """Pull the JSON object out of a reply that may include fences or prose."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found in model reply")
    return text[start : end + 1]


class GemmaClient:
    def __init__(self, settings: Settings):
        if not settings.api_key:
            raise LLMError("GEMINI_API_KEY is not set. Add it to .env or paste it in the sidebar.")
        from google import genai

        self.settings = settings
        self.client = genai.Client(api_key=settings.api_key)
        self.calls = 0

    def _generate(self, prompt: str, images: list[ImagePart]) -> str:
        from google.genai import types

        parts = [types.Part.from_text(text=prompt)]
        for img in images:
            parts.append(types.Part.from_text(text=f"[Image] {img.label}"))
            parts.append(types.Part.from_bytes(data=img.data, mime_type=img.mime_type))
        config = types.GenerateContentConfig(
            temperature=self.settings.temperature,
            max_output_tokens=self.settings.max_output_tokens,
        )
        delay = 4.0
        for attempt in range(5):
            try:
                self.calls += 1
                started = time.monotonic()
                resp = self.client.models.generate_content(
                    model=self.settings.model,
                    contents=[types.Content(role="user", parts=parts)],
                    config=config,
                )
                print(f"[researchgap] {self.settings.model} call took {time.monotonic() - started:.0f}s "
                      f"({len(images)} image(s))", flush=True)
                return resp.text or ""
            except Exception as exc:  # rate limits and transient server errors
                msg = str(exc)
                retryable = any(code in msg for code in ("429", "500", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE"))
                if not retryable or attempt == 4:
                    raise LLMError(f"Gemini API call failed: {msg}") from exc
                print(f"[researchgap] retrying in {delay:.0f}s after: {msg[:200]}", flush=True)
                time.sleep(delay)
                delay *= 2
        raise LLMError("unreachable")

    def structured(self, prompt: str, model_cls: type[T], images: list[ImagePart] | None = None,
                   max_repairs: int = 2) -> T:
        schema = json.dumps(model_cls.model_json_schema(), separators=(",", ":"))
        full_prompt = (
            f"{prompt}\n\n"
            "Respond with ONE JSON object only, no prose and no markdown, that validates against this JSON Schema:\n"
            f"{schema}"
        )
        reply = self._generate(full_prompt, images or [])
        for attempt in range(max_repairs + 1):
            try:
                return model_cls.model_validate_json(extract_json(reply))
            except (ValueError, ValidationError) as err:
                if attempt == max_repairs:
                    raise LLMError(f"{model_cls.__name__}: model output did not validate: {err}") from err
                repair = (
                    f"{full_prompt}\n\nYour previous reply was:\n{reply[:6000]}\n\n"
                    f"It failed validation with this error:\n{str(err)[:1500]}\n"
                    "Return a corrected JSON object only."
                )
                # images are omitted on repair: the content is already in the previous reply
                reply = self._generate(repair, [])
        raise LLMError("unreachable")
