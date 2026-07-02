#ETNodes Gemini API Nodes by Edvard Toth - https://edvardtoth.com

import torch
import os
import io
import time
import numpy as np
import random
import base64
import hashlib
import tempfile
from typing import Any

import PIL.Image
import PIL.ImageOps
import imageio
from google import genai
from google.genai import types, errors

try:
    from comfy_api.latest import InputImpl
    HAS_COMFY_API = True
except ImportError:
    HAS_COMFY_API = False

SUPPORT_THINKING_LEVEL = True
GEMINI_MAX_INPUT_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

_GLOBAL_CONTEXT_CACHES = {}
_CACHE_LIST_RETRIEVED = False
_UNSUPPORTED_CACHING_MODELS = set()

def get_video_path(video) -> str | None:
    if video is None:
        return None
    if isinstance(video, str):
        return video
    if isinstance(video, dict):
        return video.get("video") or video.get("path") or video.get("filename")
    return (
        getattr(video, "video", None) or 
        getattr(video, "path", None) or 
        getattr(video, "filename", None) or 
        getattr(video, "_VideoFromFile__file", None)
    )

def compute_context_hash(model: str, system_prompt: str, search_grounding: str, images=None, audio=None, video=None) -> str:
    hasher = hashlib.sha256()
    hasher.update(model.encode('utf-8'))
    hasher.update((system_prompt or "").encode('utf-8'))
    hasher.update(search_grounding.encode('utf-8'))
    
    if images is not None:
        try:
            for img in images:
                hasher.update(img.cpu().numpy().tobytes())
        except Exception as e:
            print(f"ETNodes Warning: Failed to hash images tensor for cache: {e}")
            
    if audio is not None:
        try:
            waveform = audio.get("waveform")
            sample_rate = audio.get("sample_rate")
            if waveform is not None:
                hasher.update(waveform.cpu().numpy().tobytes())
            if sample_rate is not None:
                hasher.update(str(sample_rate).encode('utf-8'))
        except Exception as e:
            print(f"ETNodes Warning: Failed to hash audio for cache: {e}")
            
    if video is not None:
        try:
            video_path = get_video_path(video)
            if video_path and os.path.exists(video_path):
                hasher.update(video_path.encode('utf-8'))
                hasher.update(str(os.path.getsize(video_path)).encode('utf-8'))
                hasher.update(str(os.path.getmtime(video_path)).encode('utf-8'))
        except Exception as e:
            print(f"ETNodes Warning: Failed to hash video for cache: {e}")
            
    return hasher.hexdigest()

def get_cached_content_name(client, cache_hash: str) -> str | None:
    global _CACHE_LIST_RETRIEVED
    if cache_hash in _GLOBAL_CONTEXT_CACHES:
        return _GLOBAL_CONTEXT_CACHES[cache_hash]
    
    if not _CACHE_LIST_RETRIEVED:
        try:
            for cache in client.caches.list():
                if cache.display_name:
                    _GLOBAL_CONTEXT_CACHES[cache.display_name] = cache.name
            _CACHE_LIST_RETRIEVED = True
        except Exception as e:
            print(f"ETNodes Warning: Failed to list context caches from API: {e}")
            
    return _GLOBAL_CONTEXT_CACHES.get(cache_hash)

def count_tokens_helper(client, model: str, contents, system_prompt: str) -> int:
    try:
        config = types.CountTokensConfig(
            system_instruction=system_prompt if system_prompt and system_prompt.strip() else None
        )
        response = client.models.count_tokens(
            model=model,
            contents=contents,
            config=config
        )
        return response.total_tokens
    except Exception:
        try:
            response = client.models.count_tokens(
                model=model,
                contents=contents
            )
            sys_tokens = len(system_prompt) // 4 if system_prompt else 0
            return response.total_tokens + sys_tokens
        except Exception as e:
            print(f"ETNodes Warning: Failed to count tokens: {e}")
            return 0

def upload_pil_image_to_files(client, pil_image) -> Any:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        pil_image.save(temp_file.name, format="PNG")
        temp_file_path = temp_file.name
    try:
        return client.files.upload(file=temp_file_path)
    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass

def upload_audio_to_files(client, pcm_data, sample_rate) -> Any:
    import wave
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        with wave.open(temp_file.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data.tobytes())
        temp_audio_path = temp_file.name
    try:
        return client.files.upload(file=temp_audio_path)
    finally:
        if os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except Exception:
                pass

def wait_for_files_active(client, files):
    if not files:
        return
    start_time = time.time()
    pending_files = list(files)
    while pending_files and (time.time() - start_time) < 120:
        active_files = []
        for f in pending_files:
            try:
                updated_f = client.files.get(name=f.name)
                state_name = getattr(updated_f.state, "name", str(updated_f.state))
                if state_name == "ACTIVE":
                    active_files.append(f)
                elif state_name == "FAILED":
                    print(f"ETNodes Warning: File upload failed processing for {f.name}")
                    active_files.append(f)
            except Exception as e:
                print(f"ETNodes Warning: Error checking file status for {f.name}: {e}")
                active_files.append(f)
        for f in active_files:
            pending_files.remove(f)
        if pending_files:
            time.sleep(1.0)


def to_pil(image):
    return PIL.Image.fromarray((image.cpu().numpy().squeeze() * 255).astype('uint8'))

def from_pil(image):
    out = torch.from_numpy(np.array(image)).float().div(255.0)  # type: ignore
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

def get_safety_settings(level) -> list[Any]:
    
    threshold = "BLOCK_NONE"
    if level == "minimum":
        threshold = "BLOCK_ONLY_HIGH"
    elif level == "medium":
        threshold = "BLOCK_MEDIUM_AND_ABOVE"
    elif level == "maximum":
        threshold = "BLOCK_LOW_AND_ABOVE"
    # "none" and "off" (legacy) fall through to BLOCK_NONE
    
    try:
        return [
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold=threshold),  # type: ignore
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold=threshold),  # type: ignore
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold=threshold),  # type: ignore
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold=threshold),  # type: ignore
        ]
    except Exception:
        return [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": threshold},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": threshold},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": threshold},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": threshold},
        ]

class ETNodesGeminiApiImage:
    """
    A node to generate and edit images using the Google Gemini API, with adjustable safety settings.
    It supports both text-to-image and image-to-image generation.
    """
    NODE_NAME = "ETNodes Gemini API Image"
    CATEGORY = "ETNodes"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "The text prompt to generate an image from."}),
                "system_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Optional system prompt to guide the model's behavior.\nParticularly useful for defining a persona for the model."}),
                "model": (["gemini-3-pro-image", "gemini-3.1-flash-image", "gemini-3.1-flash-lite-image"], {"default": "gemini-3-pro-image", "tooltip": "The model to use for image generation and editing.\nNano Banana Pro --> gemini-3-pro-image\nNano Banana 2 --> gemini-3.1-flash-image\nNano Banana 2 Lite --> gemini-3.1-flash-lite-image"}),
                "resolution": (["1K", "2K", "4K"], {"default": "1K", "tooltip": "The output resolution for the generated image."}),
                "aspect_ratio": (["auto", "1:1", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5", "9:16", "16:9", "21:9", "1:4", "4:1", "1:8", "8:1"], {"default": "auto", "tooltip": "The aspect ratio of the generated image.\nThe AUTO setting will match the aspect ratio of the input image(s)."}),
                "safety_level": (["none", "minimum", "medium", "maximum"], {"default": "none", "advanced": True, "tooltip": "The safety level for content moderation.\nNONE - Will disable probability-based safety filters for harassment, hate speech, sexual content, and dangerous content (some core protections cannot be disabled)."}),
                "search_grounding": (["off", "on"], {"default": "off", "advanced": True, "tooltip": "Enable search grounding to allow the model to search the web for up-to-date information."}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1, "advanced": True, "tooltip": "Adjusts visual variety and randomness. Lower values are more deterministic.\nDefault is 1.0."}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05, "advanced": True, "tooltip": "Controls the diversity of tokens considered. Lower values increase determinism.\nDefault is 0.95."}),
                "top_k": ("INT", {"default": 64, "min": 1, "max": 8192, "step": 1, "advanced": True, "tooltip": "Limits token selection to the top K most probable. Higher values increase variety.\nDefault is 64."}),
                "context_caching": (["on", "off"], {"default": "on", "advanced": True, "tooltip": "Enable context caching to conserve API credits and reduce latency when reusing the same system prompts or reference media.\nAt least a 4096 token payload is required. Not all models support caching."}),
                "cache_ttl": ("INT", {"default": 300, "min": 60, "max": 86400, "step": 60, "advanced": True, "tooltip": "The lifespan (Time-to-Live) of the cache in seconds. The cache is automatically extended on hit, and deleted after this duration of inactivity."}),
                "seed": ("INT", {"default": random.randint(0, 0xffffffffffffffff), "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "API_key": ("STRING", {"multiline": False, "default": "","tooltip": "Your Gemini API key.\n\nAdd the API key to a GEMINI_API_KEY environment variable\nand leave this field blank for more convenience and security."}),
                "images": ("IMAGE", {"tooltip": "Optional batch of input images.\nUp to 14 images for gemini-3-pro-image."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_NODE = True
    FUNCTION = "execute"
    
    def execute(self, prompt, system_prompt, model, safety_level, search_grounding, aspect_ratio, resolution, temperature, top_p, top_k, context_caching, cache_ttl, seed, API_key=None, images=None):
        # Map legacy/preview models to prevent breaking existing workflows
        if model in ["gemini-2.5-flash-image", "gemini-3-pro-image-preview"]:
            model = "gemini-3-pro-image"
        elif model == "gemini-3.1-flash-image-preview":
            model = "gemini-3.1-flash-image"

        if API_key is None or API_key.strip() == "":
            API_key = os.environ.get("GEMINI_API_KEY")
        if API_key is None or API_key.strip() == "":
            raise Exception("Gemini API Key not found. Please provide it in the node's input or set the GEMINI_API_KEY environment variable.")

        client = genai.Client(api_key=API_key)

        pils = []
        if images is not None:
            if model in ["gemini-3-pro-image", "gemini-3-pro-image-preview"] and len(images) > 14:
                raise Exception("The gemini-3-pro-image model supports a maximum of 14 images.")
            for image in images:
                pils.append(to_pil(image))

        if prompt.strip() == "" and not pils:
            raise Exception("Either a prompt or at least one image is required.")

        if prompt.strip() == "" and pils:
            prompt = "What is in this image? Describe it in detail."

        image_config = None
        if aspect_ratio != "auto" or "gemini-3" in model:
            img_conf_params = {}
            if aspect_ratio != "auto":
                final_aspect_ratio = aspect_ratio
                if model in ["gemini-3-pro-image", "gemini-3-pro-image-preview"]:
                    # Fallback mapping for models that don't support the new aspect ratios
                    ratio_fallback = {
                        "1:4": "2:3",
                        "4:1": "3:2",
                        "1:8": "9:16",
                        "8:1": "16:9"
                    }
                    if aspect_ratio in ratio_fallback:
                        final_aspect_ratio = ratio_fallback[aspect_ratio]
                img_conf_params["aspect_ratio"] = final_aspect_ratio

            if "gemini-3" in model:
                 img_conf_params["image_size"] = resolution
            # Use dict instead of types.ImageConfig to avoid version compatibility issues
            image_config = img_conf_params

        # Retry loop for generation (in case of expired/missing cache 404 error)
        max_attempts = 2
        for attempt in range(max_attempts):
            cache_name = None
            is_cached = False
            
            if context_caching == "on" and model not in _UNSUPPORTED_CACHING_MODELS:
                cache_hash = compute_context_hash(
                    model,
                    system_prompt,
                    "on" if search_grounding == "on" else "off",
                    images,
                    None,
                    None
                )
                
                cached_val = get_cached_content_name(client, cache_hash)
                
                if cached_val == "TOO_SMALL":
                    pass
                elif cached_val is not None:
                    cache_name = cached_val
                    is_cached = True
                    try:
                        client.caches.update(
                            name=cache_name,
                            config=types.UpdateCachedContentConfig(ttl=f"{cache_ttl}s")
                        )
                    except Exception as e:
                        print(f"ETNodes Warning: Failed to refresh cache TTL: {e}")
                else:
                    # Check token count to decide whether to cache
                    total_tokens = count_tokens_helper(client, model, pils, system_prompt)
                    min_tokens = 2048 if "gemini-2" in model else 4096
                    
                    if total_tokens >= min_tokens:
                        print(f"ETNodes Gemini API: Caching enabled. Input context meets minimum token threshold ({total_tokens}/{min_tokens} tokens). Creating cache...")
                        uploaded_files = []
                        
                        try:
                            if images is not None:
                                for img in images:
                                    pil_img = to_pil(img)
                                    ref = upload_pil_image_to_files(client, pil_img)
                                    uploaded_files.append(ref)
                                    
                            # Wait for all uploaded files to transition to ACTIVE state
                            wait_for_files_active(client, uploaded_files)
                                    
                            cache_contents_for_creation = uploaded_files if uploaded_files else ["Cached context instructions:"]
                            
                            cache = client.caches.create(
                                model=model,
                                config=types.CreateCachedContentConfig(
                                    contents=cache_contents_for_creation,
                                    system_instruction=system_prompt if system_prompt and system_prompt.strip() else None,
                                    tools=[{"google_search": {}}] if search_grounding == "on" else None,
                                    ttl=f"{cache_ttl}s",
                                    display_name=cache_hash
                                )
                            )
                            cache_name = cache.name
                            _GLOBAL_CONTEXT_CACHES[cache_hash] = cache_name
                            is_cached = True
                        except Exception as e:
                            print(f"ETNodes Warning: Failed to create context cache: {e}")
                            err_str = str(e).lower()
                            if "not supported" in err_str or "not found" in err_str or "404" in err_str:
                                print(f"ETNodes Gemini API: Caching is not supported for model '{model}'. Disabling caching for this model.")
                                _UNSUPPORTED_CACHING_MODELS.add(model)
                    else:
                        print(f"ETNodes Gemini API: Caching skipped. Context size ({total_tokens} tokens) is below the minimum threshold ({min_tokens} tokens).")
                        _GLOBAL_CONTEXT_CACHES[cache_hash] = "TOO_SMALL"
                        
            # Configure request parameters based on cached state
            if cache_name:
                gen_contents = [prompt]
                gen_config = types.GenerateContentConfig(
                    cached_content=cache_name,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_output_tokens=32768,
                    response_modalities=["IMAGE"],
                    safety_settings=get_safety_settings(safety_level),
                    image_config=image_config
                )
            else:
                # Standard generation
                gen_contents = []
                if prompt:
                    gen_contents.append(prompt)
                gen_contents.extend(pils)
                
                gen_config = types.GenerateContentConfig(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_output_tokens=32768,
                    response_modalities=["IMAGE"],
                    safety_settings=get_safety_settings(safety_level),
                    image_config=image_config,
                    system_instruction=system_prompt if system_prompt and system_prompt.strip() else None
                )
                if search_grounding == "on":
                    gen_config.tools = [{"google_search": {}}]
            
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=gen_contents,
                    config=gen_config
                )
                break
            except errors.APIError as e:
                # Catch 404 Cache Expired/Not Found to retry once
                if e.code == 404 and cache_name and attempt < max_attempts - 1:
                    print(f"ETNodes Gemini API: Cache expired or not found (404). Re-creating cache for hash...")
                    if cache_hash in _GLOBAL_CONTEXT_CACHES:
                        del _GLOBAL_CONTEXT_CACHES[cache_hash]
                    continue
                else:
                    raise Exception(f"API request failed: {e}")

        images_out = []
        text_responses = []

        # Parse response
        if response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, "finish_reason"):
                    reason = str(candidate.finish_reason)
                    if "IMAGE_SAFETY" in reason:
                         raise Exception("Safety Block: The generated content was filtered due to safety settings.\nTry adjusting the 'safety_level' or modifying the prompt.")
                    if "NO_IMAGE" in reason:
                         raise Exception("Model Refusal: The model failed to generate an image.\nThis may be due to prompt complexity or internal hard safety constraints.")

                if hasattr(candidate, "content") and candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if part.inline_data:
                            image_data = part.inline_data.data
                            if image_data is not None:
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

        return {"ui": {"cached": [is_cached]}, "result": (torch.cat(images_out, dim=0),)}

class ETNodesGeminiApiText:
    """
    A node to generate text from multimodal inputs using the Google Gemini API.
    """
    NODE_NAME = "ETNodes Gemini API Text"
    CATEGORY = "ETNodes"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "The text prompt for the model."}),
                "system_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Optional system prompt to guide the model's behavior.\nParticularly useful for defining a persona for the model."}),
                "model": (["gemini-3.5-flash", "gemini-3.1-pro-preview", "gemini-3-flash-preview", "gemini-3.1-flash-lite", "gemini-3.5-pro (Coming Soon)"], {"default": "gemini-3.5-flash", "tooltip": "The model to use for input file analysis and text generation."}),
                "safety_level": (["none", "minimum", "medium", "maximum"], {"default": "none", "advanced": True, "tooltip": "The safety level for content moderation.\nNONE - Will disable probability-based safety filters for harassment, hate speech, sexual content, and dangerous content (some core protections cannot be disabled)."}),
                "thinking_level": (["high", "medium", "low"], {"default": "high", "advanced": True, "tooltip": "Determine the reasoning depth for Gemini 3 and 3.5 models.\nDefault is HIGH for maximum reasoning."}),
                "search_grounding": (["off", "on"], {"default": "off", "advanced": True, "tooltip": "Enable search grounding to allow the model to search the web for up-to-date information."}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1, "advanced": True, "tooltip": "Controls creative flair and randomness. Lower values are more deterministic.\nDefault is 1.0."}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05, "advanced": True, "tooltip": "Controls the diversity of tokens considered. Lower values increase determinism.\nDefault is 0.95."}),
                "top_k": ("INT", {"default": 64, "min": 1, "max": 8192, "step": 1, "advanced": True, "tooltip": "Limits token selection to the top K most probable. Higher values increase variety.\nDefault is 64."}),
                "context_caching": (["on", "off"], {"default": "on", "advanced": True, "tooltip": "Enable context caching to conserve API credits and reduce latency when reusing the same system prompts or reference media.\nAt least a 4096 token payload is required. Not all models support caching."}),
                "cache_ttl": ("INT", {"default": 300, "min": 60, "max": 86400, "step": 60, "advanced": True, "tooltip": "The lifespan (Time-to-Live) of the cache in seconds. The cache is automatically extended on hit, and deleted after this duration of inactivity."}),
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
    OUTPUT_NODE = True
    FUNCTION = "execute"

    def execute(self, prompt, system_prompt, model, safety_level, thinking_level, search_grounding, temperature, top_p, top_k, context_caching, cache_ttl, seed, API_key=None, images=None, audio=None, video=None):
        if "Coming Soon" in model:
            raise Exception("Gemini 3.5 Pro has not been released yet by Google. Please choose an active model.")

        # Map legacy models to prevent breaking existing workflows
        if model == "gemini-2.5-flash":
            model = "gemini-3.5-flash"
        elif model == "gemini-3-flash-lite":
            model = "gemini-3.1-flash-lite"
        elif model in ["gemini-2.5-pro", "gemini-3.5-pro-preview"]:
            model = "gemini-3.1-pro-preview"

        if API_key is None or API_key.strip() == "":
            API_key = os.environ.get("GEMINI_API_KEY")
        if API_key is None or API_key.strip() == "":
            raise Exception("Gemini API Key not found. Please provide it in the node's input or set the GEMINI_API_KEY environment variable.")

        client = genai.Client(api_key=API_key)

        # 1. Build cacheable static contents
        cache_contents = []
        if images is not None:
            for image in images:
                cache_contents.append(to_pil(image))

        pcm_data = None
        sample_rate = None
        audio_part = None
        if audio is not None:
            import wave
            waveform = audio.get('waveform')
            sample_rate = audio.get('sample_rate')

            if waveform is None or sample_rate is None:
                raise Exception("Invalid audio input. Must contain 'waveform' and 'sample_rate'.")
            
            shape = waveform.shape
            if len(shape) < 3:
                raise Exception(f"Invalid audio waveform shape: {shape}. Expected [batch, channels, samples].")
            
            float_data = waveform[0][0].cpu().numpy()
            float_data = np.clip(float_data, -1.0, 1.0)
            pcm_data = (float_data * 32767.0).astype(np.int16)
            
            audio_buffer = io.BytesIO()
            with wave.open(audio_buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(pcm_data.tobytes())
            
            audio_part = types.Part(
                inline_data=types.Blob(
                    mime_type="audio/wav",
                    data=audio_buffer.getvalue()
                )
            )
            cache_contents.append(audio_part)

        video_path = None
        video_part = None
        if video is not None:
            video_path = get_video_path(video)
            if not video_path or not isinstance(video_path, str) or not os.path.exists(video_path):
                raise Exception(f"Failed to find or access video file from input: {video}")

            if os.path.getsize(video_path) > GEMINI_MAX_INPUT_FILE_SIZE:
                raise Exception(f"Video file size exceeds the 20MB limit for the Gemini API.")
            
            reader = imageio.get_reader(video_path)
            with reader:
                fps = reader.get_meta_data().get('fps', 30)
                video_buffer = io.BytesIO()
                writer = imageio.get_writer(video_buffer, format='mp4', mode='I', fps=fps)
                with writer:
                    for frame in reader:
                        writer.append_data(frame)
            
            video_part = types.Part(
                inline_data=types.Blob(
                    mime_type="video/mp4",
                    data=video_buffer.getvalue()
                )
            )
            cache_contents.append(video_part)

        # Retry loop for generation (in case of expired/missing cache 404 error)
        max_attempts = 2
        for attempt in range(max_attempts):
            cache_name = None
            is_cached = False
            
            if context_caching == "on" and model not in _UNSUPPORTED_CACHING_MODELS:
                cache_hash = compute_context_hash(
                    model,
                    system_prompt,
                    "on" if search_grounding == "on" else "off",
                    images,
                    audio,
                    video
                )
                
                cached_val = get_cached_content_name(client, cache_hash)
                
                if cached_val == "TOO_SMALL":
                    pass
                elif cached_val is not None:
                    # Cache hit!
                    cache_name = cached_val
                    is_cached = True
                    try:
                        client.caches.update(
                            name=cache_name,
                            config=types.UpdateCachedContentConfig(ttl=f"{cache_ttl}s")
                        )
                    except Exception as e:
                        print(f"ETNodes Warning: Failed to refresh cache TTL: {e}")
                else:
                    # Check token count to decide whether to cache
                    total_tokens = count_tokens_helper(client, model, cache_contents, system_prompt)
                    min_tokens = 2048 if "gemini-2" in model else 4096
                    
                    if total_tokens >= min_tokens:
                        print(f"ETNodes Gemini API: Caching enabled. Input context meets minimum token threshold ({total_tokens}/{min_tokens} tokens). Creating cache...")
                        uploaded_files = []
                        
                        try:
                            if images is not None:
                                for img in images:
                                    pil_img = to_pil(img)
                                    ref = upload_pil_image_to_files(client, pil_img)
                                    uploaded_files.append(ref)
                                    
                            if audio is not None and pcm_data is not None and sample_rate is not None:
                                ref = upload_audio_to_files(client, pcm_data, sample_rate)
                                uploaded_files.append(ref)
                                
                            if video is not None and video_path is not None:
                                ref = client.files.upload(file=video_path)
                                uploaded_files.append(ref)
                                
                            # Wait for all uploaded files to transition to ACTIVE state
                            wait_for_files_active(client, uploaded_files)
                                
                            cache_contents_for_creation = uploaded_files if uploaded_files else ["Cached context instructions:"]
                            
                            cache = client.caches.create(
                                model=model,
                                config=types.CreateCachedContentConfig(
                                    contents=cache_contents_for_creation,
                                    system_instruction=system_prompt if system_prompt and system_prompt.strip() else None,
                                    tools=[{"google_search": {}}] if search_grounding == "on" else None,
                                    ttl=f"{cache_ttl}s",
                                    display_name=cache_hash
                                )
                            )
                            cache_name = cache.name
                            _GLOBAL_CONTEXT_CACHES[cache_hash] = cache_name
                            is_cached = True
                        except Exception as e:
                            print(f"ETNodes Warning: Failed to create context cache: {e}")
                            err_str = str(e).lower()
                            if "not supported" in err_str or "not found" in err_str or "404" in err_str:
                                print(f"ETNodes Gemini API: Caching is not supported for model '{model}'. Disabling caching for this model.")
                                _UNSUPPORTED_CACHING_MODELS.add(model)
                    else:
                        print(f"ETNodes Gemini API: Caching skipped. Context size ({total_tokens} tokens) is below the minimum threshold ({min_tokens} tokens).")
                        _GLOBAL_CONTEXT_CACHES[cache_hash] = "TOO_SMALL"
                        
            # Configure request parameters based on cached state
            if cache_name:
                gen_contents = [prompt] if prompt and prompt.strip() else ["What is in this context? Describe it."]
                gen_config = types.GenerateContentConfig(
                    cached_content=cache_name,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_output_tokens=32768,
                    safety_settings=get_safety_settings(safety_level)
                )
                if SUPPORT_THINKING_LEVEL:
                    setattr(gen_config, "thinking_config", types.ThinkingConfig(thinking_level=thinking_level.upper()))
            else:
                # Standard generation
                gen_contents = []
                if prompt and prompt.strip():
                    gen_contents.append(prompt)
                gen_contents.extend(cache_contents)
                
                if not gen_contents:
                    raise Exception("At least one input (prompt, image, audio, or video) is required.")
                    
                gen_config = types.GenerateContentConfig(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_output_tokens=32768,
                    safety_settings=get_safety_settings(safety_level),
                    system_instruction=system_prompt if system_prompt and system_prompt.strip() else None
                )
                if SUPPORT_THINKING_LEVEL:
                    setattr(gen_config, "thinking_config", types.ThinkingConfig(thinking_level=thinking_level.upper()))
                if search_grounding == "on":
                    gen_config.tools = [{"google_search": {}}]
            
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=gen_contents,
                    config=gen_config
                )
                break
            except errors.APIError as e:
                # Catch 404 Cache Expired/Not Found to retry once
                if e.code == 404 and cache_name and attempt < max_attempts - 1:
                    print(f"ETNodes Gemini API: Cache expired or not found (404). Re-creating cache for hash...")
                    if cache_hash in _GLOBAL_CONTEXT_CACHES:
                        del _GLOBAL_CONTEXT_CACHES[cache_hash]
                    continue
                else:
                    raise Exception(f"API request failed: {e}")

        text_responses = []
        if response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, "finish_reason"):
                    reason = str(candidate.finish_reason)
                    if "SAFETY" in reason:
                        raise Exception("Safety Block: The generated text was filtered due to safety settings.\nTry adjusting the 'safety_level' or modifying the prompt.")
                    if "RECITATION" in reason:
                        raise Exception("Recitation Block: The model flagged this as potential copyright infringement (recitation).")
                    if "OTHER" in reason:
                        raise Exception("Model Refusal: The model refused to generate text (Reason: OTHER).\nThis may be due to prompt complexity or safety constraints not explicitly flagged.")

                if hasattr(candidate, "content") and candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if part.text:
                            text_responses.append(part.text)

        if not text_responses:
            raise Exception(f"The model returned no text. Response object: {response}")

        return {"ui": {"cached": [is_cached]}, "result": (" ".join(text_responses),)}

NODE_CLASS_MAPPINGS = {
    "ETNodes-Gemini-API-Image": ETNodesGeminiApiImage,
    "ETNodes-Gemini-API-Text": ETNodesGeminiApiText
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ETNodes-Gemini-API-Image": "ETNodes Gemini API Image",
    "ETNodes-Gemini-API-Text": "ETNodes Gemini API Text"
}


def load_video_frames(mp4_path: str) -> torch.Tensor:
    reader = imageio.get_reader(mp4_path)
    frames = []
    for frame in reader:
        frame_float = frame.astype(np.float32) / 255.0
        if frame_float.shape[-1] == 4:
            frame_float = frame_float[:, :, :3]
        elif len(frame_float.shape) == 2:
            frame_float = np.stack([frame_float] * 3, axis=-1)
        frames.append(torch.from_numpy(frame_float))
    return torch.stack(frames, dim=0)


def extract_audio_from_mp4(mp4_path: str) -> dict | None:
    import wave
    import subprocess
    import imageio_ffmpeg

    wav_path = mp4_path + ".wav"
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", mp4_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", "44100",
        wav_path
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except Exception as e:
        print(f"ETNodes Warning: Failed to extract audio using ffmpeg: {e}")
        return None

    try:
        if not os.path.exists(wav_path):
            return None
        with wave.open(wav_path, "rb") as wf:
            sample_rate = wf.getframerate()
            num_frames = wf.getnframes()
            audio_bytes = wf.readframes(num_frames)

            pcm_data = np.frombuffer(audio_bytes, dtype=np.int16)
            float_data = pcm_data.astype(np.float32) / 32768.0

            waveform = torch.from_numpy(float_data).unsqueeze(0).unsqueeze(0)
            return {
                "waveform": waveform,
                "sample_rate": sample_rate
            }
    except Exception as e:
        print(f"ETNodes Warning: Failed to read extracted audio WAV file: {e}")
        return None
    finally:
        if os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass


def strip_audio_from_mp4(mp4_path: str) -> str:
    import subprocess
    import imageio_ffmpeg

    output_path = mp4_path + ".silent.mp4"
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", mp4_path,
        "-c:v", "copy",
        "-an",
        output_path
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        if os.path.exists(output_path):
            os.replace(output_path, mp4_path)
    except Exception as e:
        print(f"ETNodes Warning: Failed to strip audio from MP4: {e}")
    return mp4_path


def scale_mp4(mp4_path: str, resolution: str, aspect_ratio: str) -> str:
    height = 720
    if resolution == "1080p":
        height = 1080
    elif resolution == "4K":
        height = 2160
    else:
        return mp4_path

    if aspect_ratio == "9:16":
        width = int(height * 9 / 16)
    else:
        width = int(height * 16 / 9)

    width = (width // 2) * 2
    height = (height // 2) * 2

    import subprocess
    import imageio_ffmpeg

    output_path = mp4_path + f".{resolution}.mp4"
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", mp4_path,
        "-vf", f"scale={width}:{height}",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        output_path
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        if os.path.exists(output_path):
            os.replace(output_path, mp4_path)
    except Exception as e:
        print(f"ETNodes Warning: Failed to scale MP4: {e}")
    return mp4_path


class ETNodesGeminiApiVideo:
    """
    A node to generate and edit videos using the Google Gemini Omni model, supporting agentic multi-turn conversational video editing.
    """
    NODE_NAME = "ETNodes Gemini API Video"
    CATEGORY = "ETNodes"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "The text prompt describing the video or edit."}),
                "system_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Optional system instruction to guide model behavior."}),
                "model": (["gemini-omni-flash-preview"], {"default": "gemini-omni-flash-preview", "tooltip": "The video generation model to use."}),
                "resolution": (["720p", "1080p", "4K"], {"default": "720p", "tooltip": "The output video resolution."}),
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9", "tooltip": "The aspect ratio of the generated video."}),
                "duration_seconds": ("INT", {"default": 4, "min": 1, "max": 10, "step": 1, "tooltip": "The length of the generated video clip in seconds."}),
                "generate_audio": (["on", "off"], {"default": "on", "tooltip": "Generate synchronized background audio/dialogue."}),
                "safety_level": (["none", "minimum", "medium", "maximum"], {"default": "none", "advanced": True, "tooltip": "The safety level for content filtering."}),
                "seed": ("INT", {"default": random.randint(0, 0xffffffffffffffff), "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "API_key": ("STRING", {"multiline": False, "default": "", "tooltip": "Your Gemini API key."}),
                "image": ("IMAGE", {"tooltip": "Optional starting frame or vision reference image."}),
                "reference_images": ("IMAGE", {"tooltip": "Optional style/context reference images."}),
                "audio": ("AUDIO", {"tooltip": "Optional reference audio."}),
                "video": ("VIDEO", {"tooltip": "Optional source video to perform video-to-video editing."}),
                "session": ("GEMINI_SESSION", {"tooltip": "Chained conversation state from a previous Gemini API Video node. Use in combination with the FIXED seed option to prevent regeneration of unchanged content."}),
            }
        }

    RETURN_TYPES = ("VIDEO", "IMAGE", "AUDIO", "GEMINI_SESSION",)
    RETURN_NAMES = ("video", "images", "audio", "session",)
    FUNCTION = "execute"

    def execute(self, prompt, system_prompt, model, resolution, aspect_ratio, duration_seconds, generate_audio, safety_level, seed,
                API_key=None, image=None, reference_images=None, audio=None, video=None, session=None):

        if API_key is None or API_key.strip() == "":
            API_key = os.environ.get("GEMINI_API_KEY")
        if API_key is None or API_key.strip() == "":
            raise Exception("Gemini API Key not found. Please provide it in the node's input or set the GEMINI_API_KEY environment variable.")

        client = genai.Client(api_key=API_key)

        # 1. Handle dynamic duration clamping/mapping on the backend
        if duration_seconds > 10:
            print(f"ETNodes Warning: Gemini Omni Flash preview supports a maximum duration of 10 seconds. Clamped to 10s.")
            duration_seconds = 10

        # Fallback aspect ratio
        if aspect_ratio == "1:1":
            print(f"ETNodes Warning: Video models do not support 1:1 aspect ratio. Falling back to 16:9.")
            aspect_ratio = "16:9"

        # Define outputs list
        video_bytes = None
        new_interaction_id = None
        history = []

        # If a session is connected, retrieve history/states
        if session is not None and isinstance(session, dict):
            history = list(session.get("history", []))
            new_interaction_id = session.get("previous_interaction_id")

        # --- Gemini Omni Flash Path (Interactions API) ---
        input_parts = []
        uploaded_files = []

        # Determine if this is a conversational edit or a fresh run
        if new_interaction_id is None:
            # Fresh run: upload any connected images, video, audio reference
            if image is not None:
                ref = upload_pil_image_to_files(client, to_pil(image))
                uploaded_files.append(ref)
                input_parts.append({"type": "document", "uri": ref.uri})

            if reference_images is not None:
                for img in reference_images:
                    ref = upload_pil_image_to_files(client, to_pil(img))
                    uploaded_files.append(ref)
                    input_parts.append({"type": "document", "uri": ref.uri})

            if video is not None:
                video_path = get_video_path(video)
                if video_path and os.path.exists(video_path):
                    ref = client.files.upload(file=video_path)
                    uploaded_files.append(ref)
                    input_parts.append({"type": "document", "uri": ref.uri})

            if audio is not None:
                waveform = audio.get('waveform')
                sr = audio.get('sample_rate')
                if waveform is not None and sr is not None:
                    float_data = waveform[0][0].cpu().numpy()
                    float_data = np.clip(float_data, -1.0, 1.0)
                    pcm_data = (float_data * 32767.0).astype(np.int16)
                    ref = upload_audio_to_files(client, pcm_data, sr)
                    uploaded_files.append(ref)
                    input_parts.append({"type": "document", "uri": ref.uri})

            if prompt and prompt.strip():
                enhanced_prompt = f"{prompt} (Duration: {duration_seconds} seconds)"
                input_parts.append({"type": "text", "text": enhanced_prompt})

            if not input_parts:
                raise Exception("Omni Flash requires a prompt or at least one image/video input to start a video generation.")

            wait_for_files_active(client, uploaded_files)
            api_input = input_parts
        else:
            # Conversational Edit
            enhanced_prompt = f"{prompt} (Duration: {duration_seconds} seconds)"
            api_input = [{"type": "text", "text": enhanced_prompt}]

        safety = get_safety_settings(safety_level)
        gen_config = {
            "temperature": 1.0,
            "safety_settings": safety,
        }
        response_format = {
            "type": "video",
            "aspect_ratio": aspect_ratio
        }

        try:
            interaction = client.interactions.create(
                model=model,
                input=api_input,
                previous_interaction_id=new_interaction_id,
                system_instruction=system_prompt if system_prompt and system_prompt.strip() else None,
                generation_config=gen_config,
                response_format=response_format
            )

            if not hasattr(interaction, "output_video") or not interaction.output_video:
                raise Exception(f"Omni Flash did not return a generated video. Output text: {getattr(interaction, 'output_text', 'None')}")

            if hasattr(interaction.output_video, "data") and interaction.output_video.data:
                video_bytes = base64.b64decode(interaction.output_video.data)

            new_interaction_id = interaction.id
            history.append({"role": "user", "text": prompt})
            if getattr(interaction, "output_text", None):
                history.append({"role": "model", "text": interaction.output_text})

        except Exception as e:
            raise Exception(f"Omni Flash Interaction failed: {e}")

        if not video_bytes:
            raise Exception("No video bytes downloaded from the API.")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_mp4:
            temp_mp4.write(video_bytes)
            temp_mp4_path = temp_mp4.name

        try:
            # 1. Scale video if resolution is 1080p or 4K
            if resolution in ["1080p", "4K"]:
                scale_mp4(temp_mp4_path, resolution, aspect_ratio)

            # 2. Strip audio if generate_audio is off
            if generate_audio == "off":
                strip_audio_from_mp4(temp_mp4_path)

            frames_tensor = load_video_frames(temp_mp4_path)

            audio_output = None
            if generate_audio == "on":
                audio_output = extract_audio_from_mp4(temp_mp4_path)

            session_out = {
                "history": history,
                "previous_interaction_id": new_interaction_id,
                "last_video_path": temp_mp4_path,
                "last_video_frames": frames_tensor,
                "last_audio": audio_output
            }

            if HAS_COMFY_API:
                video_output = InputImpl.VideoFromFile(temp_mp4_path)
            else:
                class MockVideo:
                    def __init__(self, path):
                        self.path = path
                    def get_dimensions(self):
                        return (512, 512)
                video_output = MockVideo(temp_mp4_path)
            audio_data = audio_output if audio_output is not None else {"waveform": torch.zeros(1, 1, 1), "sample_rate": 44100}
            return (video_output, frames_tensor, audio_data, session_out)

        finally:
            pass


# Class and Display Mappings
NODE_CLASS_MAPPINGS = {
    "ETNodes-Gemini-API-Image": ETNodesGeminiApiImage,
    "ETNodes-Gemini-API-Text": ETNodesGeminiApiText,
    "ETNodes-Gemini-API-Video": ETNodesGeminiApiVideo
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ETNodes-Gemini-API-Image": "ETNodes Gemini API Image",
    "ETNodes-Gemini-API-Text": "ETNodes Gemini API Text",
    "ETNodes-Gemini-API-Video": "ETNodes Gemini API Video"
}


