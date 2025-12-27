#ETNodes Gemini API Nodes by Edvard Toth - https://edvardtoth.com

import torch
import os
import sys
import subprocess
import io
import numpy as np
import random

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
    from google import genai
    from google.genai import types
except:
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', '-q', 'install', 'google-genai'])
        from google import genai
        from google.genai import types
    except:
        pass



try:
    import imageio
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', '-q', 'install', 'imageio', 'imageio-ffmpeg'])
    import imageio

GEMINI_MAX_INPUT_FILE_SIZE = 20 * 1024 * 1024  # 20 MB



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
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="BLOCK_NONE",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="BLOCK_NONE",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="BLOCK_NONE",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_NONE",
            ),
        ]
    if level == "minimum":
        return [
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="BLOCK_ONLY_HIGH",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="BLOCK_ONLY_HIGH",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="BLOCK_ONLY_HIGH",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_ONLY_HIGH",
            ),
        ]
    if level == "medium":
        return [
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="BLOCK_MEDIUM_AND_ABOVE",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="BLOCK_MEDIUM_AND_ABOVE",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="BLOCK_MEDIUM_AND_ABOVE",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_MEDIUM_AND_ABOVE",
            ),
        ]
    if level == "maximum":
        return [
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="BLOCK_LOW_AND_ABOVE",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="BLOCK_LOW_AND_ABOVE",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="BLOCK_LOW_AND_ABOVE",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_LOW_AND_ABOVE",
            ),
        ]
    
    # Default to "none" for any other case
    return [
        types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="BLOCK_NONE",
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="BLOCK_NONE",
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="BLOCK_NONE",
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="BLOCK_NONE",
        ),
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
                "resolution": (["1K", "2K", "4K"], {"default": "1K", "tooltip": "The output resolution for the generated image (gemini-3-pro-image-preview only)."}),
                "aspect_ratio": (["auto", "1:1", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5", "9:16", "16:9", "21:9", ], {"default": "auto", "tooltip": "The aspect ratio of the generated image.\nThe 'auto' setting will match the aspect ratio of the input image(s)."}),
                "safety_level": (["off", "none", "minimum", "medium", "maximum"], {"default": "none", "tooltip": "The safety level for content moderation.\n'none' - Will still block high-severity content.\n'off' - A new experimental API feature to disable all safety filters. May revert to 'none'."}),
                "search_grounding": (["off", "on"], {"default": "off", "tooltip": "Enable search grounding to allow the model to search the web for up-to-date information."}),
                "seed": ("INT", {"default": random.randint(0, 0xffffffffffffffff), "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "API_key": ("STRING", {"multiline": False, "default": "","tooltip": "Your Gemini API key.\n\nAdd the API key to a GEMINI_API_KEY environment variable\nand leave this field blank for more convenience and security."}),
                "images": ("IMAGE", {"tooltip": "Optional batch of input images.\nUp to 14 images for gemini-3-pro-image-preview.\nUp to 4 images for gemini-2.5-flash-image."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    
    def execute(self, prompt, system_prompt, model, safety_level, search_grounding, aspect_ratio, resolution, seed, API_key=None, images=None):
        if API_key is None or API_key.strip() == "":
            API_key = os.environ.get("GEMINI_API_KEY")
        if API_key is None or API_key.strip() == "":
            raise Exception("Gemini API Key not found. Please provide it in the node's input or set the GEMINI_API_KEY environment variable.")

        client = genai.Client(api_key=API_key)

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

        contents = []
        if prompt:
            contents.append(prompt)
        contents.extend(pils)

        image_config = {}
        if aspect_ratio != "auto":
            image_config["aspectRatio"] = aspect_ratio
        
        if model == "gemini-3-pro-image-preview":
            image_config["imageSize"] = resolution

        # Construct configuration
        config_dict = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 32768,
            "response_modalities": ["IMAGE"],
            "safety_settings": get_safety_settings(safety_level),
        }

        if search_grounding == "on":
             if model != "gemini-2.5-flash-image":
                 config_dict["tools"] = [{"google_search": {}}]

        if image_config:
            config_dict["image_config"] = image_config
        
        if system_prompt and system_prompt.strip():
            config_dict["system_instruction"] = system_prompt

        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config_dict
            )
        except Exception as e:
            raise Exception(f"API request failed: {e}")

        images_out = []
        text_responses = []

        # Parse response
        if response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, "content") and candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if part.inline_data:
                            # Handling inline image data
                            # part.inline_data.data should be bytes in new SDK
                            image_data = part.inline_data.data
                            # If it is base64 string (some versions), decode it
                            if isinstance(image_data, str):
                                image_data = base64.b64decode(image_data)
                            
                            pil_image = PIL.Image.open(io.BytesIO(image_data))
                            images_out.append(from_pil(pil_image))
                        elif part.text:
                            text_responses.append(part.text)
        
        if not images_out:
            text_response = " ".join(text_responses)
            
            if text_response:
                raise Exception(f"No image was generated. The model returned text instead: {text_response}")
            
            raise Exception(f"No image was generated and no text explanation was provided. Response object: {response}")

        return (torch.cat(images_out, dim=0),)

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
                "model": (["gemini-3-pro-preview", "gemini-3-flash-preview", "gemini-2.5-pro", "gemini-2.5-flash"], {"default": "gemini-3-pro-preview", "tooltip": "The model to use for input file analysis and text generation."}),
                "safety_level": (["off", "none", "minimum", "medium", "maximum"], {"default": "none", "tooltip": "The safety level for content moderation.\n'none' - Will still block high-severity content.\n'off' - A new experimental API feature to disable all safety filters. May revert to 'none'."}),
                "search_grounding": (["off", "on"], {"default": "off", "tooltip": "Enable search grounding to allow the model to search the web for up-to-date information."}),
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

    def execute(self, prompt, system_prompt, model, safety_level, search_grounding, seed, API_key=None, images=None, audio=None, video=None):
        if API_key is None or API_key.strip() == "":
            API_key = os.environ.get("GEMINI_API_KEY")
        if API_key is None or API_key.strip() == "":
            raise Exception("Gemini API Key not found. Please provide it in the node's input or set the GEMINI_API_KEY environment variable.")

        client = genai.Client(api_key=API_key)

        contents = []
        if prompt and prompt.strip():
            contents.append(prompt)

        if images is not None:
            for image in images:
                pil_image = to_pil(image)
                contents.append(pil_image)

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
            
            contents.append(types.Part(
                inline_data=types.Blob(
                    mime_type="audio/wav",
                    data=buffer.getvalue()
                )
            ))

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
                
                contents.append(types.Part(
                    inline_data=types.Blob(
                        mime_type="video/mp4",
                        data=buffer.getvalue()
                    )
                ))

        if not contents:
            raise Exception("At least one input (prompt, image, audio, or video) is required.")

        config_dict = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 32768,
            "safety_settings": get_safety_settings(safety_level),
            "system_instruction": system_prompt if system_prompt and system_prompt.strip() else None,
        }

        if search_grounding == "on":
            config_dict["tools"] = [{"google_search": {}}]

        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config_dict
            )
        except Exception as e:
            raise Exception(f"API request failed: {e}")

        text_responses = []
        if response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, "content") and candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if part.text:
                            text_responses.append(part.text)

        if not text_responses:
            # Simple error reporting
            raise Exception(f"The model returned no text. Response object: {response}")

        return (" ".join(text_responses),)

NODE_CLASS_MAPPINGS = {
    "ETNodes-Gemini-API-Image": ETNodesGeminiApiImage,
    "ETNodes-Gemini-API-Text": ETNodesGeminiApiText
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ETNodes-Gemini-API-Image": "ΞΓNodes Gemini API Image",
    "ETNodes-Gemini-API-Text": "ΞΓNodes Gemini API Text"
}
