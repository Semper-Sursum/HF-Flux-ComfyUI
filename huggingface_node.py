import os
import json
import requests
import numpy as np
import torch
from PIL import Image
from io import BytesIO

class HuggingFaceTextToImage:
    def __init__(self):
        self.api_key = ""
        self.model_id = "black-forest-labs/FLUX.1-dev"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model_id": ("STRING", {"default": "black-forest-labs/FLUX.1-dev"}),
                "api_key": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "image generation"

    def generate_image(self, prompt, model_id, api_key):
        # Use provided API key
        if api_key:
            self.api_key = api_key
        
        if not self.api_key:
            raise ValueError("API key is required. Please provide your Hugging Face API key.")
        
        # Update model ID if provided
        if model_id:
            self.model_id = model_id
        
        # API endpoint for text-to-image inference
        api_url = f"https://router.huggingface.co/hf-inference/models/{self.model_id}"
        
        # Headers with authorization
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Data payload
        payload = {
            "inputs": prompt
        }
        
        try:
            # Make the API request
            response = requests.post(api_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                # For image generation models, the response is binary image data
                image = Image.open(BytesIO(response.content))
                
                # Convert to RGB if the image is in RGBA mode
                if image.mode == "RGBA":
                    image = image.convert("RGB")
                
                # Convert to the format ComfyUI expects (numpy array)
                image_np = np.array(image).astype(np.float32) / 255.0
                
                # Add batch dimension if needed
                if len(image_np.shape) == 3:
                    image_np = image_np[None, ...]
                
                # Convert numpy array to PyTorch tensor
                image_tensor = torch.from_numpy(image_np)
                return (image_tensor,)
            else:
                error_message = f"Error: {response.status_code} - {response.text}"
                print(error_message)
                raise Exception(error_message)
                
        except Exception as e:
            print(f"Error generating image: {str(e)}")
            raise e

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "HuggingFaceTextToImage": HuggingFaceTextToImage
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "HuggingFaceTextToImage": "Hugging Face Text to Image"
}
