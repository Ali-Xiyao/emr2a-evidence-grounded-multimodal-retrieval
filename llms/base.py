from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union
from PIL import Image


class BaseLLM(ABC):
    def __init__(self, device: str = "cuda"):
        self.device = device
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        images: Optional[List[Image.Image]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        pass
    
    @abstractmethod
    def generate_batch(
        self,
        prompts: List[str],
        images_list: Optional[List[List[Image.Image]]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        **kwargs
    ) -> List[str]:
        pass
    
    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, Union[str, List[Dict]]]],
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        pass
    
    def to(self, device: str):
        self.device = device
        return self
