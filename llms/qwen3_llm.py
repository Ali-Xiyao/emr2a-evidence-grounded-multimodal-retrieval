import torch
from typing import List, Dict, Optional, Union
from PIL import Image

from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

from .base import BaseLLM


class Qwen3LLM(BaseLLM):
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
    ):
        super().__init__(device)
        self.model_path = model_path
        self.dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
        
        self._load_model()
    
    def _load_model(self):
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            dtype=self.dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        self.model.eval()
    
    def generate(
        self,
        prompt: str,
        images: Optional[List[Image.Image]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        
        if images:
            for img in images:
                messages[0]["content"].append({"type": "image", "image": img})
        
        return self.chat(
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
    
    def generate_batch(
        self,
        prompts: List[str],
        images_list: Optional[List[List[Image.Image]]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        **kwargs
    ) -> List[str]:
        results = []
        for i, prompt in enumerate(prompts):
            images = images_list[i] if images_list else None
            result = self.generate(
                prompt=prompt,
                images=images,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )
            results.append(result)
        return results
    
    def chat(
        self,
        messages: List[Dict[str, Union[str, List[Dict]]]],
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
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
        
        do_sample = temperature > 0
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs
            )
        
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(inputs.input_ids, outputs)
        ]
        
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        
        return response
