from typing import Optional
from pathlib import Path

from .base import BaseEncoder
from .qwen3_vl_8b_thinking_encoder import Qwen3VL8BThinkingEncoder
from .qwen3_vl_2b_thinking_encoder import Qwen3VL2BThinkingEncoder
from .vit_encoder import ViTEncoder
from .biomedclip_encoder import BioMedCLIPLEncoder
from .clip_encoder import CLIPEncoder
from .dino_encoder import DINOEncoder


def create_encoder(
    encoder_type: str,
    device: str = "cuda",
    model_path: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs
) -> BaseEncoder:
    encoder_map = {
        "qwen3_vl": Qwen3VL8BThinkingEncoder,
        "qwen3": Qwen3VL8BThinkingEncoder,
        "qwen3_vl_8b": Qwen3VL8BThinkingEncoder,
        "qwen3_vl_8b_thinking": Qwen3VL8BThinkingEncoder,
        "qwen3_vl_2b": Qwen3VL2BThinkingEncoder,
        "qwen3_vl_2b_thinking": Qwen3VL2BThinkingEncoder,
        "vit": ViTEncoder,
        "biomedclip": BioMedCLIPLEncoder,
        "clip": CLIPEncoder,
        "clip_vit_large_patch14_336": CLIPEncoder,
        "dino": DINOEncoder,
        "dinov3": DINOEncoder,
        "dinov3_vitl16": DINOEncoder,
    }
    
    encoder_type_lower = encoder_type.lower()
    if encoder_type_lower not in encoder_map:
        raise ValueError(
            f"Unsupported encoder type: {encoder_type}. "
            f"Supported types: {list(encoder_map.keys())}"
        )
    
    encoder_class = encoder_map[encoder_type_lower]
    
    if encoder_type_lower in ["qwen3_vl", "qwen3", "qwen3_vl_8b", "qwen3_vl_8b_thinking"]:
        default_model_path = kwargs.get("qwen3_vl_8b_config", {}).get("model_path")
        
        return encoder_class(
            model_path=model_path or default_model_path,
            device=device,
            target_layers=kwargs.get("target_layers", [-3, -2, -1]),
            dtype=kwargs.get("dtype", "bfloat16"),
        )
    elif encoder_type_lower in ["qwen3_vl_2b", "qwen3_vl_2b_thinking"]:
        default_model_path = kwargs.get("qwen3_vl_2b_config", {}).get("model_path")
        
        return encoder_class(
            model_path=model_path or default_model_path,
            device=device,
            target_layers=kwargs.get("target_layers", [-3, -2, -1]),
            dtype=kwargs.get("dtype", "bfloat16"),
        )
    elif encoder_type_lower == "vit":
        return encoder_class(
            model_name=model_name or kwargs.get("vit_config", {}).get("model_name", "vit_base_patch16_224"),
            model_path=model_path or kwargs.get("vit_config", {}).get("model_path"),
            device=device,
        )
    elif encoder_type_lower == "biomedclip":
        return encoder_class(
            model_path=model_path or kwargs.get("biomedclip_config", {}).get("model_path"),
            device=device,
        )
    elif encoder_type_lower in ["clip", "clip_vit_large_patch14_336"]:
        return encoder_class(
            model_path=model_path or kwargs.get("clip_config", {}).get("model_path"),
            device=device,
        )
    elif encoder_type_lower in ["dino", "dinov3", "dinov3_vitl16"]:
        return encoder_class(
            model_path=model_path or kwargs.get("dino_config", {}).get("model_path"),
            device=device,
        )
    else:
        raise ValueError(f"Unsupported encoder type: {encoder_type}")


__all__ = [
    'create_encoder', 
    'BaseEncoder', 
    'Qwen3VL8BThinkingEncoder', 
    'Qwen3VL2BThinkingEncoder', 
    'ViTEncoder', 
    'BioMedCLIPLEncoder',
    'CLIPEncoder',
    'DINOEncoder'
]
