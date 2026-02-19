import torch
import numpy as np
from typing import List, Optional
from PIL import Image

from .base import BaseEncoder


class DINOEncoder(BaseEncoder):
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
    ):
        super().__init__(device)
        self.model_path = model_path
        self._load_model()
    
    def _load_model(self):
        try:
            from transformers import AutoImageProcessor, AutoModel
            from pathlib import Path
            import os
            
            print(f"Loading DINO model from: {self.model_path}")
            
            if self.model_path and os.path.isdir(self.model_path):
                print(f"Using local DINO model from {self.model_path}")
                model_path = self.model_path
            else:
                print(f"DINO model path not found, using HuggingFace hub")
                model_path = "facebook/dinov3-vitl16"
            
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
            self.processor = AutoImageProcessor.from_pretrained(model_path)
            
            self.model.to(self.device)
            self.model.eval()
            
            print(f"DINO model loaded successfully!")
        except ImportError:
            raise ImportError(
                "transformers is required for DINO. "
                "Install with: pip install transformers"
            )
    
    def encode_image(self, image: Image.Image) -> Optional[np.ndarray]:
        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                image_features = outputs.last_hidden_state
                image_features = image_features.mean(dim=1)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.squeeze(0).cpu().float().numpy()
        except Exception as e:
            print(f"Error encoding image: {e}")
            return None
    
    def encode_text(self, text: str) -> Optional[np.ndarray]:
        print("Warning: DINO does not support text encoding. Returning None.")
        return None
    
    def encode_batch_images(self, images: List[Image.Image]) -> List[Optional[np.ndarray]]:
        try:
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                image_features = outputs.last_hidden_state
                image_features = image_features.mean(dim=1)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            image_features = image_features.cpu().float().numpy()
            return [image_features[i] for i in range(len(images))]
        except Exception as e:
            print(f"Error encoding batch images: {e}")
            return [None for _ in images]
    
    def encode_batch_texts(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        print("Warning: DINO does not support text encoding. Returning None for all texts.")
        return [None for _ in texts]
