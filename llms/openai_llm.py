from typing import List, Dict, Optional, Union
from PIL import Image

from .base import BaseLLM


class OpenAILLM(BaseLLM):
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-vision-preview",
        device: str = "cuda",
    ):
        super().__init__(device)
        self.api_key = api_key
        self.model = model
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError(
                "openai is required for OpenAI LLM. "
                "Install with: pip install openai"
            )
    
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
                import io
                import base64
                
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_str}"
                    }
                })
        
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
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
        
        return response.choices[0].message.content
