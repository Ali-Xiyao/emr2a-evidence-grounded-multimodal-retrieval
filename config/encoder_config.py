from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List


@dataclass
class EncoderConfig:
    encoder_type: str = "qwen3_vl_8b"
    
    model_path: Optional[Path] = None
    model_name: Optional[str] = None
    
    sample_n: int = 4
    sample_mode: str = "even"
    variance_size: int = 64
    
    text_mode: str = "hybrid"
    
    target_layers: List[int] = field(default_factory=lambda: [-3, -2, -1])
    
    qwen3_vl_8b_config: dict = field(default_factory=lambda: {
        "model_path": None,
    })
    
    qwen3_vl_2b_config: dict = field(default_factory=lambda: {
        "model_path": None,
    })
    
    vit_config: dict = field(default_factory=lambda: {
        "model_name": "vit_base_patch16_224",
        "model_path": None,
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
