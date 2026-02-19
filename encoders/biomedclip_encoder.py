import torch
import numpy as np
from typing import List, Optional
from PIL import Image

from .base import BaseEncoder


class BioMedCLIPLEncoder(BaseEncoder):
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
            from open_clip import create_model_and_transforms, get_tokenizer
            import torch
            from pathlib import Path
            import os
            
            model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            pretrained = None
            
            if self.model_path and os.path.isdir(self.model_path):
                model_file = Path(self.model_path) / "open_clip_pytorch_model.bin"
                if model_file.exists():
                    pretrained = self.model_path
                    print(f"Using local BioMedCLIP model from {self.model_path}")
                else:
                    print(f"Model file not found at {model_file}, will download from hub")
            
            self.model, _, self.preprocess = create_model_and_transforms(
                model_name,
                pretrained=pretrained,
            )
            
            self.model.to(self.device)
            self.model.eval()
            self.tokenizer = get_tokenizer(model_name)
        except ImportError:
            raise ImportError(
                "open_clip is required for BioMedCLIP. "
                "Install with: pip install open_clip_torch"
            )
    
    def encode_image(self, image: Image.Image) -> Optional[np.ndarray]:
        try:
            image = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.squeeze(0).cpu().float().numpy()
        except Exception as e:
            print(f"Error encoding image: {e}")
            return None
    
    def encode_text(self, text: str) -> Optional[np.ndarray]:
        try:
            text = self.tokenizer([text]).to(self.device)
            
            with torch.no_grad():
                text_features = self.model.encode_text(text)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            return text_features.squeeze(0).cpu().float().numpy()
        except Exception as e:
            print(f"Error encoding text: {e}")
            return None
    
    def encode_batch_images(self, images: List[Image.Image]) -> List[Optional[np.ndarray]]:
        return [self.encode_image(img) for img in images]
    
    def encode_batch_texts(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        return [self.encode_text(text) for text in texts]
