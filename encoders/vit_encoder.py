import torch
import timm
import numpy as np
from typing import List, Optional
from PIL import Image
from pathlib import Path

from timm.data import create_transform, resolve_data_config

from .base import BaseEncoder


class ViTEncoder(BaseEncoder):
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        model_path: Optional[str] = None,
        device: str = "cuda",
    ):
        super().__init__(device)
        self.model_name = model_name
        self.model_path = model_path
        
        self._load_model()
        self._setup_transform()
    
    def _load_model(self):
        self.model = timm.create_model(
            self.model_name,
            pretrained=False,
            num_classes=0,
            global_pool="avg"
        )
        
        if self.model_path:
            state = torch.load(self.model_path, map_location="cpu")
            if isinstance(state, dict):
                for key in ("state_dict", "model", "model_state", "model_state_dict"):
                    if key in state:
                        state = state[key]
                        break
            
            if isinstance(state, dict):
                cleaned = {}
                for key, value in state.items():
                    cleaned[key.replace("module.", "")] = value
                self.model.load_state_dict(cleaned, strict=False)
        
        self.model.to(self.device)
        self.model.eval()
    
    def _setup_transform(self):
        data_config = resolve_data_config({}, model=self.model)
        self.transform = create_transform(**data_config, is_training=False)
    
    def encode_image(self, image: Image.Image) -> Optional[np.ndarray]:
        try:
            img = image.convert("RGB")
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                feats = self.model(tensor)
            
            if isinstance(feats, (list, tuple)):
                feats = feats[0]
            
            if feats.ndim > 1:
                feats = feats.squeeze(0)
            
            return feats.cpu().float().numpy()
        except Exception as e:
            print(f"Error encoding image: {e}")
            return None
    
    def encode_text(self, text: str) -> Optional[np.ndarray]:
        raise NotImplementedError("ViT encoder does not support text encoding")
    
    def encode_batch_images(self, images: List[Image.Image]) -> List[Optional[np.ndarray]]:
        return [self.encode_image(img) for img in images]
    
    def encode_batch_texts(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        raise NotImplementedError("ViT encoder does not support text encoding")
