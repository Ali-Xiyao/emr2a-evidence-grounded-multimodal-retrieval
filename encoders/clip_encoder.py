import torch
import numpy as np
from typing import List, Optional
from pathlib import Path
from PIL import Image

from .base import BaseEncoder


class CLIPEncoder(BaseEncoder):
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
            from transformers import CLIPModel, CLIPProcessor, CLIPConfig
            import torch
            
            config = CLIPConfig.from_pretrained(
                self.model_path,
                local_files_only=True,
                trust_remote_code=True,
            )
            
            self.model = CLIPModel(config)
            
            model_path = Path(self.model_path) / "pytorch_model.bin"
            if model_path.exists():
                state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
                self.model.load_state_dict(state_dict, strict=False)
            
            self.processor = CLIPProcessor.from_pretrained(
                self.model_path,
                local_files_only=True,
                trust_remote_code=True,
            )
            
            self.model.to(self.device)
            self.model.eval()
        except ImportError:
            raise ImportError(
                "transformers is required for CLIP. "
                "Install with: pip install transformers"
            )
    
    def encode_image(self, image: Image.Image) -> Optional[np.ndarray]:
        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.model.vision_model(**inputs).pooler_output
                image_features = self.model.visual_projection(image_features)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.squeeze(0).cpu().float().numpy()
        except Exception as e:
            print(f"Error encoding image: {e}")
            return None
    
    def encode_text(self, text: str) -> Optional[np.ndarray]:
        try:
            inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            
            with torch.no_grad():
                text_features = self.model.text_model(**inputs).pooler_output
                text_features = self.model.text_projection(text_features)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            return text_features.squeeze(0).cpu().float().numpy()
        except Exception as e:
            print(f"Error encoding text: {e}")
            return None
    
    def encode_batch_images(self, images: List[Image.Image]) -> List[Optional[np.ndarray]]:
        try:
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.model.vision_model(**inputs).pooler_output
                image_features = self.model.visual_projection(image_features)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            image_features = image_features.cpu().float().numpy()
            return [image_features[i] for i in range(len(images))]
        except Exception as e:
            print(f"Error encoding batch images: {e}")
            return [None for _ in images]
    
    def encode_batch_texts(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        try:
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            
            with torch.no_grad():
                text_features = self.model.text_model(**inputs).pooler_output
                text_features = self.model.text_projection(text_features)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            text_features = text_features.cpu().float().numpy()
            return [text_features[i] for i in range(len(texts))]
        except Exception as e:
            print(f"Error encoding batch texts: {e}")
            return [None for _ in texts]
