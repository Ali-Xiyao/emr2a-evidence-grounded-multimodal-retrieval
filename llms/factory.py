from typing import Optional

from .base import BaseLLM
from .qwen3_llm import Qwen3LLM
from .openai_llm import OpenAILLM


def create_llm(
    llm_type: str,
    device: str = "cuda",
    model_path: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs
) -> BaseLLM:
    llm_map = {
        "qwen3": Qwen3LLM,
        "qwen3_vl": Qwen3LLM,
        "qwen3_vl_8b_thinking": Qwen3LLM,
        "qwen3_vl_2b_thinking": Qwen3LLM,
        "openai": OpenAILLM,
        "gpt4": OpenAILLM,
    }
    
    llm_type_lower = llm_type.lower()
    if llm_type_lower not in llm_map:
        raise ValueError(
            f"Unsupported LLM type: {llm_type}. "
            f"Supported types: {list(llm_map.keys())}"
        )
    
    llm_class = llm_map[llm_type_lower]
    
    if llm_type_lower in ["qwen3", "qwen3_vl", "qwen3_vl_8b_thinking", "qwen3_vl_2b_thinking"]:
        config_key = llm_type_lower if llm_type_lower in kwargs else None
        if config_key:
            default_model_path = kwargs.get(config_key, {}).get("model_path")
        elif llm_type_lower == "qwen3_vl_8b_thinking":
            default_model_path = kwargs.get("qwen3_vl_8b_thinking_config", {}).get("model_path")
        elif llm_type_lower == "qwen3_vl_2b_thinking":
            default_model_path = kwargs.get("qwen3_vl_2b_thinking_config", {}).get("model_path")
        else:
            default_model_path = kwargs.get("qwen3_config", {}).get("model_path")
        
        return llm_class(
            model_path=model_path or default_model_path,
            device=device,
            dtype=kwargs.get("dtype", "bfloat16"),
        )
    elif llm_type_lower in ["openai", "gpt4"]:
        return llm_class(
            api_key=api_key or kwargs.get("openai_config", {}).get("api_key"),
            model=model_name or kwargs.get("openai_config", {}).get("model", "gpt-4-vision-preview"),
            device=device,
        )
    else:
        raise ValueError(f"Unsupported LLM type: {llm_type}")


__all__ = ['create_llm', 'BaseLLM', 'Qwen3LLM', 'OpenAILLM']
