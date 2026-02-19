from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class LLMConfig:
    llm_type: str = "qwen3_vl_8b_thinking"
    
    model_path: Optional[Path] = None
    
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9
    
    qwen3_vl_8b_thinking_config: dict = field(default_factory=lambda: {
        "model_path": None,
    })
    
    qwen3_vl_2b_thinking_config: dict = field(default_factory=lambda: {
        "model_path": None,
    })
    
    openai_config: dict = field(default_factory=lambda: {
        "api_key": None,
        "model": "gpt-4-vision-preview",
    })
    
    biomedclip_config: dict = field(default_factory=lambda: {
        "model_path": None,
    })
    
    clip_config: dict = field(default_factory=lambda: {
        "model_path": None,
    })
    
    dino_config: dict = field(default_factory=lambda: {
        "model_path": None,
    })
