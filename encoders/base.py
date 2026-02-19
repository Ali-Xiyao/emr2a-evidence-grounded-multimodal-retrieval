from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from PIL import Image
from pathlib import Path
import numpy as np


class BaseEncoder(ABC):
    def __init__(self, device: str = "cuda"):
        self.device = device
    
    @abstractmethod
    def encode_image(self, image: Image.Image) -> Optional[np.ndarray]:
        pass
    
    @abstractmethod
    def encode_text(self, text: str) -> Optional[np.ndarray]:
        pass
    
    @abstractmethod
    def encode_batch_images(self, images: List[Image.Image]) -> List[Optional[np.ndarray]]:
        pass
    
    @abstractmethod
    def encode_batch_texts(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        pass
    
    def encode_images(self, image_paths: List[Path]) -> np.ndarray:
        images = []
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
            except Exception:
                images.append(None)
        
        embeddings = self.encode_batch_images(images)
        
        valid_embeddings = [emb for emb in embeddings if emb is not None]
        if valid_embeddings:
            return np.array(valid_embeddings)
        return np.array([])
    
    def to(self, device: str):
        self.device = device
        return self
