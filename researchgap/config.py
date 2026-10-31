from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

GEMMA_MODELS = ["gemma-4-31b-it", "gemma-4-26b-a4b-it"]


@dataclass
class Settings:
    api_key: str = ""
    model: str = GEMMA_MODELS[0]
    temperature: float = 0.2
    max_output_tokens: int = 8192
    max_images_per_paper: int = 6
    render_dpi: int = 110
    run_text_only_baseline: bool = True
    max_papers: int = 3

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            api_key=os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", ""),
            model=os.getenv("GEMMA_MODEL", GEMMA_MODELS[0]),
            max_images_per_paper=int(os.getenv("MAX_IMAGES_PER_PAPER", "6")),
            render_dpi=int(os.getenv("RENDER_DPI", "110")),
        )
