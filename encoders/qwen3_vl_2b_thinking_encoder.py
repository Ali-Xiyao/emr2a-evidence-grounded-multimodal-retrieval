import copy
import torch
import numpy as np
from typing import Dict, List, Optional
from PIL import Image
from pathlib import Path

from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

from .base import BaseEncoder


class Qwen3VL2BThinkingEncoder(BaseEncoder):
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        target_layers: Optional[List[int]] = None,
        dtype: str = "bfloat16",
    ):
        super().__init__(device)
        self.model_path = model_path
        self.target_layers = target_layers or [-3, -2, -1]
        self.dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
        
        self._load_model()
        self._setup_hooks()
    
    def _load_model(self):
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            dtype=self.dtype,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model.eval()
        
        try:
            self.model.generation_config.do_sample = False
            self.model.generation_config.temperature = None
            self.model.generation_config.top_p = None
            self.model.generation_config.top_k = None
        except Exception:
            pass
        
        self.vision_model = self.model.model.visual
        self.embed_dim = getattr(self.vision_model, "embed_dim", None)
        if self.embed_dim is None:
            self.embed_dim = getattr(self.vision_model, "hidden_size", None)
        if self.embed_dim is None:
            self.embed_dim = getattr(self.model.config, "vision_hidden_size", None)
    
    def _setup_hooks(self):
        self.hook_outputs: Dict[str, torch.Tensor] = {}
        self.hooks = []
        
        num_blocks = len(self.vision_model.blocks)
        for layer_offset in self.target_layers:
            layer_idx = num_blocks + layer_offset
            if 0 <= layer_idx < num_blocks:
                hook = self.vision_model.blocks[layer_idx].register_forward_hook(
                    self._create_hook_fn(f"block_{layer_idx}")
                )
                self.hooks.append(hook)
    
    def _create_hook_fn(self, layer_name: str):
        def hook(module, inputs, output):
            self.hook_outputs[layer_name] = output.detach()
        return hook
    
    def _clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def encode_image(self, image: Image.Image) -> Optional[np.ndarray]:
        self.hook_outputs = {}
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "."},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        with torch.no_grad():
            _ = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                output_hidden_states=False,
                return_dict=True,
            )
        
        layer_outputs = []
        for layer_name in self.hook_outputs:
            output = self.hook_outputs[layer_name]
            if output.ndim == 2:
                if (self.embed_dim is not None and 
                    output.shape[0] == self.embed_dim and 
                    output.shape[1] != self.embed_dim):
                    output = output.transpose(0, 1)
                output = output.unsqueeze(0)
            if output.ndim != 3:
                continue
            
            if self.embed_dim is not None:
                if output.shape[-1] == self.embed_dim:
                    pooled = output.mean(dim=1)
                elif output.shape[-2] == self.embed_dim:
                    pooled = output.mean(dim=2)
                else:
                    pooled = output.mean(dim=1)
            else:
                if output.shape[-1] >= output.shape[-2]:
                    pooled = output.mean(dim=1)
                else:
                    pooled = output.mean(dim=2)
            
            layer_emb = pooled.squeeze(0).cpu().float().numpy()
            layer_outputs.append(layer_emb)
        
        if not layer_outputs:
            return None
        
        embedding = np.mean(layer_outputs, axis=0)
        return embedding
    
    def encode_text(self, text: str) -> Optional[np.ndarray]:
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )
        
        last_hidden = outputs.hidden_states[-1]
        embedding = last_hidden.mean(dim=1).squeeze(0).cpu().float().numpy()
        return embedding
    
    def encode_batch_images(self, images: List[Image.Image]) -> List[Optional[np.ndarray]]:
        return [self.encode_image(img) for img in images]
    
    def encode_batch_texts(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        return [self.encode_text(text) for text in texts]
    
    def __del__(self):
        if hasattr(self, 'hooks'):
            self._clear_hooks()
