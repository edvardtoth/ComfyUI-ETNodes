#ETNodes Gemini API Nodes by Edvard Toth - https://edvardtoth.com

import torch
import os
import sys
import subprocess
import io
import numpy as np
import random
import json
import base64
import folder_paths

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

try:
    import imageio
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', '-q', 'install', 'imageio', 'imageio-ffmpeg'])
    import imageio

GEMINI_MAX_INPUT_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

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
    if level == "off":
        return [
            {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": "OFF"},
            {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": "OFF"},
            {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": "OFF"},
            {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": "OFF"},
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
    
    # Default to "none" for any other case, including "none" or invalid values
    return [
        {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
        {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
        {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_NONE},
        {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
    ]

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
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "The text prompt to generate an image from."}),
                "system_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Optional system prompt to guide the model's behavior.\nParticularly useful for defining a persona for the model."}),
                "model": (["gemini-3-pro-image-preview", "gemini-2.5-flash-image"], {"default": "gemini-3-pro-image-preview", "tooltip": "The model to use for image generation and editing."}),
                "safety_level": (["off", "none", "minimum", "medium", "maximum"], {"default": "none", "tooltip": "The safety level for content moderation.\n'none' - Will still block high-severity content.\n'off' - A new experimental API feature to disable all safety filters. May revert to 'none'."}),
                "aspect_ratio": (["auto", "1:1", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5", "9:16", "16:9", "21:9", ], {"default": "auto", "tooltip": "The aspect ratio of the generated image.\nThe 'auto' setting will match the aspect ratio of the input image(s)."}),
                "resolution": (["1K", "2K", "4K"], {"default": "1K", "tooltip": "The output resolution for the generated image (gemini-3-pro-image-preview only)."}),
                "seed": ("INT", {"default": random.randint(0, 0xffffffffffffffff), "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "API_key": ("STRING", {"multiline": False, "default": "","tooltip": "Your Gemini API key.\n\nAdd the API key to a GEMINI_API_KEY environment variable\nand leave this field blank for more convenience and security."}),
                "images": ("IMAGE", {"tooltip": "Optional batch of input images.\nUp to 14 images for gemini-3-pro-image-preview.\nUp to 4 images for gemini-2.5-flash-image."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    
    def execute(self, prompt, system_prompt, model, safety_level, aspect_ratio, resolution, seed, API_key=None, images=None):
        if API_key is None or API_key.strip() == "":
            API_key = os.environ.get("GEMINI_API_KEY")
        if API_key is None or API_key.strip() == "":
            raise Exception("Gemini API Key not found. Please provide it in the node's input or set the GEMINI_API_KEY environment variable.")

        parts = []
        pils = []
        if images is not None:
            if model == "gemini-3-pro-image-preview" and len(images) > 14:
                raise Exception("The gemini-3-pro-image-preview model supports a maximum of 14 images.")
            for image in images:
                pils.append(to_pil(image))

        if prompt.strip() == "" and not pils:
            raise Exception("Either a prompt or at least one image is required.")

        if prompt.strip() == "" and pils:
            prompt = "What is in this image? Describe it in detail."

        parts = [{"text": prompt}]
        for pil_image in pils:
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            parts.append({
                "inlineData": {
                    "mimeType": "image/png",
                    "data": base64.b64encode(buffer.getvalue()).decode()
                }
            })

        image_config = {}
        if aspect_ratio != "auto":
            image_config["aspectRatio"] = aspect_ratio
        
        if model == "gemini-3-pro-image-preview":
            image_config["imageSize"] = resolution

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": 1,
                "topP": 0.95,
                "topK": 64,
                "maxOutputTokens": 32768,
                "responseModalities": ["IMAGE"],
            },
            "safetySettings": get_safety_settings(safety_level)
        }
        
        if image_config:
            payload["generationConfig"]["imageConfig"] = image_config

        if system_prompt and system_prompt.strip():
            payload["system_instruction"] = {
                "parts": [{"text": system_prompt}]
            }

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
                if "content" in candidate and "parts" in candidate["content"]:
                    for part in candidate["content"]["parts"]:
                        if "inlineData" in part and part["inlineData"]["mimeType"].startswith("image/"):
                            image_data = base64.b64decode(part["inlineData"]["data"])
                            pil_image = PIL.Image.open(io.BytesIO(image_data))
                            images.append(from_pil(pil_image))
                        elif "text" in part:
                            text_responses.append(part["text"])

        if not images:
            text_response = " ".join(text_responses)
            
            # Check for safety feedback first
            if "promptFeedback" in response_data and "blockReason" in response_data["promptFeedback"]:
                reason = response_data['promptFeedback']['blockReason']
                message = f"Request was blocked due to safety settings. Reason: {reason}"
                if text_response:
                    message += f". Model response: {text_response}"
                raise Exception(message)

            # Check for finishMessage in candidates
            if "candidates" in response_data:
                for candidate in response_data["candidates"]:
                    if "finishMessage" in candidate and candidate.get("finishReason") == "IMAGE_SAFETY":
                        raise Exception(f"Image generation failed due to safety settings: {candidate['finishMessage']}")
                    if candidate.get("finishReason") == "NO_IMAGE":
                        raise Exception("Image generation failed: The model chose not to generate an image for this prompt.")

            if text_response:
                raise Exception(f"No image was generated. The model returned text instead: {text_response}")
            
            raise Exception(f"No image was generated and no text explanation was provided. Response: {response_data}")

        return (torch.cat(images, dim=0),)

class ETNodesGeminiApiText:
    """
    A node to generate text from multimodal inputs using the Google Gemini API.
    """
    NODE_NAME = "ETNodes Gemini API Text"
    CATEGORY = "ETNodes"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "The text prompt for the model."}),
                "system_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Optional system prompt to guide the model's behavior.\nParticularly useful for defining a persona for the model."}),
                "model": (["gemini-3-pro-preview", "gemini-2.5-pro", "gemini-2.5-flash"], {"default": "gemini-3-pro-preview", "tooltip": "The model to use for input file analysis and text generation."}),
                "safety_level": (["off", "none", "minimum", "medium", "maximum"], {"default": "none", "tooltip": "The safety level for content moderation.\n'none' - Will still block high-severity content.\n'off' - A new experimental API feature to disable all safety filters. May revert to 'none'."}),
                "seed": ("INT", {"default": random.randint(0, 0xffffffffffffffff), "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "API_key": ("STRING", {"multiline": False, "default": "", "tooltip": "Your Gemini API key.\n\nAdd the API key to a GEMINI_API_KEY environment variable\nand leave this field blank for more convenience and security."}),
                "images": ("IMAGE", {"tooltip": "Optional input images."}),
                "audio": ("AUDIO", {"tooltip": "Optional audio input."}),
                "video": ("VIDEO", {"tooltip": "Optional video input."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"

    def execute(self, prompt, system_prompt, model, safety_level, seed, API_key=None, images=None, audio=None, video=None):
        if API_key is None or API_key.strip() == "":
            API_key = os.environ.get("GEMINI_API_KEY")
        if API_key is None or API_key.strip() == "":
            raise Exception("Gemini API Key not found. Please provide it in the node's input or set the GEMINI_API_KEY environment variable.")

        parts = []
        if prompt and prompt.strip():
            parts.append({"text": prompt})

        if images is not None:
            for image in images:
                pil_image = to_pil(image)
                buffer = io.BytesIO()
                pil_image.save(buffer, format="PNG")
                parts.append({
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": base64.b64encode(buffer.getvalue()).decode()
                    }
                })

        if audio is not None:
            import wave
            waveform = audio['waveform']
            sample_rate = audio['sample_rate']
            
            # Convert float tensor to 16-bit PCM, taking the first channel
            pcm_data = (waveform[0][0].cpu().numpy() * 32767).astype(np.int16)
            
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2) # 2 bytes = 16 bits
                wf.setframerate(sample_rate)
                wf.writeframes(pcm_data.tobytes())
            buffer.seek(0)
            
            parts.append({
                "inlineData": {
                    "mimeType": "audio/wav",
                    "data": base64.b64encode(buffer.read()).decode()
                }
            })

        if video is not None:
            video_path = video._VideoFromFile__file
            if os.path.exists(video_path):
                if os.path.getsize(video_path) > GEMINI_MAX_INPUT_FILE_SIZE:
                    raise Exception(f"Video file size exceeds the 20MB limit for the Gemini API.")
                
                # Re-encode the video to a standard format (MP4, H.264) in memory
                with imageio.get_reader(video_path) as reader:
                    fps = reader.get_meta_data().get('fps', 30)
                    buffer = io.BytesIO()
                    with imageio.get_writer(buffer, format='mp4', mode='I', fps=fps) as writer:
                        for frame in reader:
                            writer.append_data(frame)
                
                buffer.seek(0)
                video_data = base64.b64encode(buffer.read()).decode('utf-8')
                
                parts.append({
                    "inlineData": {
                        "mimeType": "video/mp4",
                        "data": video_data
                    }
                })

        if not parts:
            raise Exception("At least one input (prompt, image, audio, or video) is required.")

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": 1,
                "topP": 0.95,
                "topK": 64,
                "maxOutputTokens": 32768,
            },
            "safetySettings": get_safety_settings(safety_level)
        }

        if system_prompt and system_prompt.strip():
            payload["system_instruction"] = {
                "parts": [{"text": system_prompt}]
            }

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
                error_message += f"\\nResponse: {json.dumps(error_details, indent=2)}"
            except (ValueError, AttributeError):
                pass
            raise Exception(error_message)

        text_responses = []
        if "candidates" in response_data:
            for candidate in response_data["candidates"]:
                if "content" in candidate:
                    for part in candidate["content"]["parts"]:
                        if "text" in part:
                            text_responses.append(part["text"])

        if not text_responses:
            if "promptFeedback" in response_data and "blockReason" in response_data["promptFeedback"]:
                raise Exception(f"Request was blocked due to safety settings. Reason: {response_data['promptFeedback']['blockReason']}")
            raise Exception(f"The model returned no text. Response: {response_data}")

        return (" ".join(text_responses),)

NODE_CLASS_MAPPINGS = {
    "ETNodes-Gemini-API-Image": ETNodesGeminiApiImage,
    "ETNodes-Gemini-API-Text": ETNodesGeminiApiText
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ETNodes-Gemini-API-Image": "ΞΓNodes Gemini API Image",
    "ETNodes-Gemini-API-Text": "ΞΓNodes Gemini API Text"
}
