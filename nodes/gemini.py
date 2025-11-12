import torch
import os
import sys
import subprocess
import io
import numpy as np
import random
import json
import base64

try:
    import PIL
    import PIL.Image
    import PIL.ImageOps
except:
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', '-q', 'install', 'Pillow'])
        import PIL
        import PIL.Image
        import PIL.ImageOps
    except:
        pass

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except:
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', '-q', 'install', 'google-generativeai'])
        import google.generativeai as genai
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
    except:
        pass

try:
    import requests
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', '-q', 'install', 'requests'])
    import requests

def configure(api_key):
    if api_key is None or api_key.strip() == "":
        api_key = os.environ.get("GEMINI_API_KEY")
    if api_key is None or api_key.strip() == "":
        raise Exception("Gemini API Key not found. Please provide it in the node's input or set the GEMINI_API_KEY environment variable.")
    genai.configure(api_key=api_key)

def to_pil(image):
    return PIL.Image.fromarray((image.cpu().numpy().squeeze() * 255).astype('uint8'))

def from_pil(image):
    out = torch.from_numpy(np.array(image)).float().div(255.0)
    if len(out.shape) == 2:
        out = out.unsqueeze(2).expand(-1, -1, 3)
    else:
        out = out[:, :, :3]
    return out.unsqueeze(0)

def get_pils(image_1, image_2, image_3, image_4):
    out = []
    if image_1 is not None:
        out.append(to_pil(image_1))
    if image_2 is not None:
        out.append(to_pil(image_2))
    if image_3 is not None:
        out.append(to_pil(image_3))
    if image_4 is not None:
        out.append(to_pil(image_4))
    return out

def get_safety_settings(level):
    if level == "none":
        return [
            {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
            {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
            {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_NONE},
            {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
        ]
    if level == "minimum":
        return [
            {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH},
            {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH},
            {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH},
            {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH},
        ]
    if level == "medium":
        return [
            {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
            {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
            {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
            {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
        ]
    if level == "maximum":
        return [
            {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE},
            {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE},
            {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE},
            {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE},
        ]
    return []

class ETNodesGeminiApiImage:
    """
    A node to generate and edit images using the Google Gemini API, with adjustable safety settings.
    It supports both text-to-image and image-to-image generation.
    """
    NODE_NAME = "ETNodes Gemini API Image"
    CATEGORY = "ETNodes"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A photorealistic elephant.", "tooltip": "The text prompt to generate an image from."}),
                "model": (["gemini-2.5-flash-image"], {"tooltip": "The model to use for image generation and editing."}),
                "safety_level": (["none", "minimum", "medium", "maximum"], {"default": "none", "tooltip": "The safety level for content moderation."}),
                "aspect_ratio": (["auto", "1:1", "16:9", "9:16", "21:9", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5"], {"default": "auto", "tooltip": "The aspect ratio of the generated image.\nThe 'auto' setting will match the aspect ratio of the input image."}),
            },
            "optional": {
                "API_key": ("STRING", {"multiline": False, "tooltip": "Your Gemini API key.\n\nAdd the API key to a GEMINI_API_KEY environment variable\nand leave this field blank for more convenience and security."}),
                "image_1": ("IMAGE", {"tooltip": "Optional input image 1. If no image is provided, a new image will be generated based on the prompt."}),
                "image_2": ("IMAGE", {"tooltip": "Optional input image 2."}),
                "image_3": ("IMAGE", {"tooltip": "Optional input image 3."}),
                "image_4": ("IMAGE", {"tooltip": "Optional input image 4."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    
    def execute(self, prompt, model, safety_level, aspect_ratio, API_key=None, image_1=None, image_2=None, image_3=None, image_4=None):
        if API_key is None or API_key.strip() == "":
            API_key = os.environ.get("GEMINI_API_KEY")
        if API_key is None or API_key.strip() == "":
            raise Exception("Gemini API Key not found. Please provide it in the node's input or set the GEMINI_API_KEY environment variable.")

        pils = get_pils(image_1, image_2, image_3, image_4)

        if prompt.strip() == "" and not pils:
            raise Exception("Either a prompt or at least one image is required.")

        if prompt.strip() == "" and pils:
            prompt = "What is in this image? Describe it in detail."

        parts = [{"text": prompt}]
        for pil_image in pils:
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            parts.append({
                "inline_data": {
                    "mime_type": "image/png",
                    "data": base64.b64encode(buffer.getvalue()).decode()
                }
            })

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": 1,
                "topP": 0.95,
                "topK": 64,
                "maxOutputTokens": 32768,
                "imageConfig": {
                    "aspectRatio": aspect_ratio
                }
            },
            "safetySettings": get_safety_settings(safety_level)
        }

        if aspect_ratio == "auto":
            del payload["generationConfig"]["imageConfig"]

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={API_key}"
        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            response_data = response.json()
        except requests.exceptions.RequestException as e:
            error_message = f"API request failed: {e}"
            try:
                error_details = response.json()
                error_message += f"\nResponse: {json.dumps(error_details, indent=2)}"
            except (ValueError, AttributeError):
                pass
            raise Exception(error_message)

        images = []
        text_responses = []
        if "candidates" in response_data:
            for candidate in response_data["candidates"]:
                if "content" in candidate:
                    for part in candidate["content"]["parts"]:
                        if "inlineData" in part and part["inlineData"]["mimeType"].startswith("image/"):
                            image_data = base64.b64decode(part["inlineData"]["data"])
                            pil_image = PIL.Image.open(io.BytesIO(image_data))
                            images.append(from_pil(pil_image))
                        elif "text" in part:
                            text_responses.append(part["text"])

        if not images:
            text_response = " ".join(text_responses)
            if "promptFeedback" in response_data and "blockReason" in response_data["promptFeedback"]:
                reason = response_data['promptFeedback']['blockReason']
                message = f"Request was blocked due to safety settings. Reason: {reason}"
                if text_response:
                    message += f". Model response: {text_response}"
                raise Exception(message)
            
            if text_response:
                raise Exception(f"No image was generated. The model returned text instead: {text_response}")
            
            raise Exception(f"No image was generated and no text explanation was provided. Response: {response_data}")

        return (torch.cat(images, dim=0),)

NODE_CLASS_MAPPINGS = {
    "ETNodes-Gemini-API-Image": ETNodesGeminiApiImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ETNodes-Gemini-API-Image": "ΞГNodes Gemini API Image"
}
