#ETNodes Gemini API Nodes by Edvard Toth - https://edvardtoth.com

import torch
import os
import io
import numpy as np
import random
import base64
from typing import Any

import PIL.Image
import PIL.ImageOps
import imageio
from google import genai
from google.genai import types

SUPPORT_THINKING_LEVEL = True
GEMINI_MAX_INPUT_FILE_SIZE = 20 * 1024 * 1024  # 20 MB




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
                "model": (["gemini-3-pro-image-preview", "gemini-3.1-flash-image-preview"], {"default": "gemini-3-pro-image-preview", "tooltip": "The model to use for image generation and editing.\nNano Banana Pro --> gemini-3-pro-image-preview\nNano Banana 2 --> gemini-3.1-flash-image-preview"}),
                "resolution": (["1K", "2K", "4K"], {"default": "1K", "tooltip": "The output resolution for the generated image (Gemini 3 models only)."}),
                "aspect_ratio": (["auto", "1:1", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5", "9:16", "16:9", "21:9", "1:4", "4:1", "1:8", "8:1"], {"default": "auto", "tooltip": "The aspect ratio of the generated image.\nThe AUTO setting will match the aspect ratio of the input image(s)."}),
                "safety_level": (["none", "minimum", "medium", "maximum"], {"default": "none", "advanced": True, "tooltip": "The safety level for content moderation.\nNONE - Will disable probability-based safety filters for harassment, hate speech, sexual content, and dangerous content (some core protections cannot be disabled)."}),
                "search_grounding": (["off", "on"], {"default": "off", "advanced": True, "tooltip": "Enable search grounding to allow the model to search the web for up-to-date information."}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1, "advanced": True, "tooltip": "Adjusts visual variety and randomness. Lower values are more deterministic.\nDefault is 1.0."}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05, "advanced": True, "tooltip": "Controls the diversity of tokens considered. Lower values increase determinism.\nDefault is 0.95."}),
                "top_k": ("INT", {"default": 64, "min": 1, "max": 8192, "step": 1, "advanced": True, "tooltip": "Limits token selection to the top K most probable. Higher values increase variety.\nDefault is 64."}),
                "seed": ("INT", {"default": random.randint(0, 0xffffffffffffffff), "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "API_key": ("STRING", {"multiline": False, "default": "","tooltip": "Your Gemini API key.\n\nAdd the API key to a GEMINI_API_KEY environment variable\nand leave this field blank for more convenience and security."}),
                "images": ("IMAGE", {"tooltip": "Optional batch of input images.\nUp to 14 images for gemini-3-pro-image-preview."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    
    def execute(self, prompt, system_prompt, model, safety_level, search_grounding, aspect_ratio, resolution, temperature, top_p, top_k, seed, API_key=None, images=None):
        # Map legacy 2.5 models to prevent breaking existing workflows
        if model == "gemini-2.5-flash-image":
            model = "gemini-3-pro-image-preview"

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

        image_config = None
        if aspect_ratio != "auto" or "gemini-3" in model:
            img_conf_params = {}
            if aspect_ratio != "auto":
                final_aspect_ratio = aspect_ratio
                if model == "gemini-3-pro-image-preview":
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

        # Construct configuration using types
        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=32768,
            response_modalities=["IMAGE"],
            safety_settings=get_safety_settings(safety_level),
            image_config=image_config,  # type: ignore
            system_instruction=system_prompt if system_prompt and system_prompt.strip() else None
        )

        if search_grounding == "on":
             config.tools = [{"google_search": {}}]  # type: ignore

        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
        except Exception as e:
            raise Exception(f"API request failed: {e}")

        images_out = []
        text_responses = []

        # Parse response
        if response.candidates:
            for candidate in response.candidates:
                # Handle finish reasons specifically
                if hasattr(candidate, "finish_reason"):
                    # Check for safety block
                    # Comparison can be done against the Enum or the string value depending on SDK version
                    # simple string checks are robust
                    reason = str(candidate.finish_reason)
                    if "IMAGE_SAFETY" in reason:
                         raise Exception("Safety Block: The generated content was filtered due to safety settings.\nTry adjusting the 'safety_level' or modifying the prompt.")
                    if "NO_IMAGE" in reason:
                         raise Exception("Model Refusal: The model failed to generate an image.\nThis may be due to prompt complexity or internal hard safety constraints.")

                if hasattr(candidate, "content") and candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if part.inline_data:
                            # Handling inline image data
                            # part.inline_data.data should be bytes in new SDK
                            image_data = part.inline_data.data
                            if image_data is not None:
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

        return (torch.cat(images_out, dim=0),)  # type: ignore

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
                "model": (["gemini-3.5-flash", "gemini-3.1-pro-preview", "gemini-3-flash-preview", "gemini-3.1-flash-lite"], {"default": "gemini-3.5-flash", "tooltip": "The model to use for input file analysis and text generation."}),
                "safety_level": (["none", "minimum", "medium", "maximum"], {"default": "none", "advanced": True, "tooltip": "The safety level for content moderation.\nNONE - Will disable probability-based safety filters for harassment, hate speech, sexual content, and dangerous content (some core protections cannot be disabled)."}),
                "thinking_level": (["high", "medium", "low"], {"default": "high", "advanced": True, "tooltip": "Determine the reasoning depth for Gemini 3 and 3.5 models.\nDefault is HIGH for maximum reasoning."}),
                "search_grounding": (["off", "on"], {"default": "off", "advanced": True, "tooltip": "Enable search grounding to allow the model to search the web for up-to-date information."}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1, "advanced": True, "tooltip": "Controls creative flair and randomness. Lower values are more deterministic.\nDefault is 1.0."}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05, "advanced": True, "tooltip": "Controls the diversity of tokens considered. Lower values increase determinism.\nDefault is 0.95."}),
                "top_k": ("INT", {"default": 64, "min": 1, "max": 8192, "step": 1, "advanced": True, "tooltip": "Limits token selection to the top K most probable. Higher values increase variety.\nDefault is 64."}),
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

    def execute(self, prompt, system_prompt, model, safety_level, thinking_level, search_grounding, temperature, top_p, top_k, seed, API_key=None, images=None, audio=None, video=None):
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

        contents = []
        if prompt and prompt.strip():
            contents.append(prompt)

        if images is not None:
            for image in images:
                pil_image = to_pil(image)
                contents.append(pil_image)

        if audio is not None:
            import wave
            waveform = audio.get('waveform')
            sample_rate = audio.get('sample_rate')

            if waveform is None or sample_rate is None:
                raise Exception("Invalid audio input. Must contain 'waveform' and 'sample_rate'.")
            
            # Check waveform shape dimensions
            shape = waveform.shape
            if len(shape) < 3:
                raise Exception(f"Invalid audio waveform shape: {shape}. Expected [batch, channels, samples].")
            
            # Convert float tensor to 16-bit PCM, taking the first channel of the first batch
            float_data = waveform[0][0].cpu().numpy()
            
            # Clip float audio values to prevent integer overflow distortion
            float_data = np.clip(float_data, -1.0, 1.0)
            pcm_data = (float_data * 32767.0).astype(np.int16)
            
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
            # Video can be a string filepath, a dict, or a custom class/object
            video_path = None
            if isinstance(video, str):
                video_path = video
            elif isinstance(video, dict):
                video_path = video.get("video") or video.get("path") or video.get("filename")
            else:
                # Try getting the filepath from known properties/methods or fallback to name mangling
                video_path = (
                    getattr(video, "video", None) or 
                    getattr(video, "path", None) or 
                    getattr(video, "filename", None) or 
                    getattr(video, "_VideoFromFile__file", None)
                )

            if not video_path or not isinstance(video_path, str) or not os.path.exists(video_path):
                raise Exception(f"Failed to find or access video file from input: {video}")

            if os.path.getsize(video_path) > GEMINI_MAX_INPUT_FILE_SIZE:
                raise Exception(f"Video file size exceeds the 20MB limit for the Gemini API.")
            
            from typing import Any
            # Re-encode the video to a standard format (MP4, H.264) in memory
            reader: Any = imageio.get_reader(video_path)
            with reader:
                fps = reader.get_meta_data().get('fps', 30)
                buffer = io.BytesIO()
                writer: Any = imageio.get_writer(buffer, format='mp4', mode='I', fps=fps)  # type: ignore
                with writer:
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

        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=32768,
            safety_settings=get_safety_settings(safety_level),
            system_instruction=system_prompt if system_prompt and system_prompt.strip() else None,
        )

        if SUPPORT_THINKING_LEVEL:
            setattr(config, "thinking_config", types.ThinkingConfig(thinking_level=thinking_level.upper())) # type: ignore
        else:
            print(f"ETNodes Warning: Ignored thinking_level '{thinking_level}' because the current google-genai package loaded in memory does not support it. Please restart ComfyUI to apply the pending SDK update.")

        if search_grounding == "on":
            config.tools = [{"google_search": {}}]  # type: ignore

        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
        except Exception as e:
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
                        # Sometimes happens with high complexity or internal errors
                        raise Exception("Model Refusal: The model refused to generate text (Reason: OTHER).\nThis may be due to prompt complexity or safety constraints not explicitly flagged.")

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


class ETNodesGeminiApiVideo:
    """
    A node to generate and edit videos using the Google Gemini / Veo API, supporting agentic multi-turn conversational video editing.
    """
    NODE_NAME = "ETNodes Gemini API Video"
    CATEGORY = "ETNodes"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "The text prompt describing the video or edit."}),
                "model": (["gemini-omni-flash-preview", "veo-3.1-generate-preview", "veo-3.1-fast-generate-preview", "veo-3.1-lite-generate-preview"], {"default": "gemini-omni-flash-preview", "tooltip": "The video generation model to use."}),
                "aspect_ratio": (["16:9", "9:16", "1:1"], {"default": "16:9", "tooltip": "The aspect ratio of the generated video."}),
                "duration_seconds": ("INT", {"default": 4, "min": 1, "max": 10, "step": 1, "tooltip": "The length of the generated video clip in seconds."}),
                "resolution": (["720p", "1080p", "4K"], {"default": "720p", "tooltip": "The output resolution for the video."}),
                "generate_audio": (["on", "off"], {"default": "on", "tooltip": "Generate synchronized background audio/dialogue."}),
                "safety_level": (["none", "minimum", "medium", "maximum"], {"default": "none", "advanced": True, "tooltip": "The safety level for content filtering."}),
                "seed": ("INT", {"default": random.randint(0, 0xffffffffffffffff), "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "API_key": ("STRING", {"multiline": False, "default": "", "tooltip": "Your Gemini API key."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Elements to exclude from the video (Veo models only)."}),
                "system_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Optional system instruction to guide model behavior."}),
                "image_first_frame": ("IMAGE", {"tooltip": "Optional starting frame for the video."}),
                "image_last_frame": ("IMAGE", {"tooltip": "Optional ending frame for the video (Veo models only)."}),
                "reference_images": ("IMAGE", {"tooltip": "Optional reference/subject images for character/asset consistency."}),
                "audio": ("AUDIO", {"tooltip": "Optional reference audio."}),
                "video": ("VIDEO", {"tooltip": "Optional source video to perform video-to-video editing."}),
                "session": ("GEMINI_SESSION", {"tooltip": "Chained conversation state from a previous Gemini API Video node."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "GEMINI_SESSION",)
    RETURN_NAMES = ("images", "audio", "session",)
    FUNCTION = "execute"

    def execute(self, prompt, model, aspect_ratio, duration_seconds, resolution, generate_audio, safety_level, seed,
                API_key=None, negative_prompt=None, system_prompt=None, image_first_frame=None, image_last_frame=None,
                reference_images=None, audio=None, video=None, session=None):

        if API_key is None or API_key.strip() == "":
            API_key = os.environ.get("GEMINI_API_KEY")
        if API_key is None or API_key.strip() == "":
            raise Exception("Gemini API Key not found. Please provide it in the node's input or set the GEMINI_API_KEY environment variable.")

        client = genai.Client(api_key=API_key)

        # 1. Handle dynamic duration clamping/mapping on the backend
        if "veo" in model:
            supported_durations = [4, 6, 8]
            closest_duration = min(supported_durations, key=lambda x: abs(x - duration_seconds))
            if closest_duration != duration_seconds:
                print(f"ETNodes Warning: Veo models only support durations of 4, 6, or 8 seconds. Adjusted {duration_seconds}s to {closest_duration}s.")
            duration_seconds = closest_duration
        elif "omni" in model:
            if duration_seconds > 10:
                print(f"ETNodes Warning: Gemini Omni Flash preview supports a maximum duration of 10 seconds. Clamped to 10s.")
                duration_seconds = 10

        # Define outputs list
        video_bytes = None
        new_interaction_id = None
        history = []

        # If a session is connected, retrieve history/states
        if session is not None and isinstance(session, dict):
            history = list(session.get("history", []))
            new_interaction_id = session.get("previous_interaction_id")

        if model == "gemini-omni-flash-preview":
            # --- Gemini Omni Flash Path (Interactions API) ---
            input_parts = []
            uploaded_files = []

            # Determine if this is a conversational edit or a fresh run
            if new_interaction_id is None:
                # Fresh run: upload any connected images, video, audio reference
                if image_first_frame is not None:
                    ref = upload_pil_image_to_files(client, to_pil(image_first_frame))
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
                    input_parts.append({"type": "text", "text": prompt})

                if not input_parts:
                    raise Exception("Omni Flash requires a prompt or at least one image/video input to start a video generation.")

                wait_for_files_active(client, uploaded_files)
                api_input = input_parts
            else:
                # Conversational Edit
                api_input = [{"type": "text", "text": prompt}]

            safety = get_safety_settings(safety_level)
            gen_config = types.GenerateContentConfig(
                temperature=1.0,
                safety_settings=safety,
            )
            # Custom params mapped inside GenerateContentConfig for the interaction
            setattr(gen_config, "aspect_ratio", aspect_ratio)
            setattr(gen_config, "duration_seconds", duration_seconds)
            setattr(gen_config, "resolution", resolution)

            try:
                interaction = client.interactions.create(
                    model=model,
                    input=api_input,
                    previous_interaction_id=new_interaction_id,
                    system_instruction=system_prompt if system_prompt and system_prompt.strip() else None,
                    generation_config=gen_config
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

        else:
            # --- Veo 3.1 Path ---
            uploaded_files = []
            first_frame_ref = None
            last_frame_ref = None
            ref_images_list = []

            try:
                if image_first_frame is not None:
                    first_frame_ref = upload_pil_image_to_files(client, to_pil(image_first_frame))
                    uploaded_files.append(first_frame_ref)

                if image_last_frame is not None:
                    last_frame_ref = upload_pil_image_to_files(client, to_pil(image_last_frame))
                    uploaded_files.append(last_frame_ref)

                if reference_images is not None:
                    for img in reference_images:
                        ref = upload_pil_image_to_files(client, to_pil(img))
                        uploaded_files.append(ref)
                        ref_images_list.append(types.VideoGenerationReferenceImage(
                            image=ref,
                            reference_type="asset"
                        ))

                wait_for_files_active(client, uploaded_files)
            except Exception as e:
                raise Exception(f"Failed to upload Veo reference images: {e}")

            veo_config = types.GenerateVideosConfig(
                aspect_ratio=aspect_ratio,
                duration_seconds=duration_seconds,
                resolution=resolution,
                generate_audio=(generate_audio == "on"),
                person_generation="allow_adult",
                negative_prompt=negative_prompt if negative_prompt and negative_prompt.strip() else None,
                last_frame=last_frame_ref if last_frame_ref else None,
                reference_images=ref_images_list if ref_images_list else None,
            )

            try:
                operation = client.models.generate_videos(
                    model=model,
                    prompt=prompt,
                    image=first_frame_ref if first_frame_ref else None,
                    config=veo_config
                )

                start_time = time.time()
                while not operation.done:
                    if time.time() - start_time > 300:
                        raise Exception("Veo video generation timed out (exceeded 5 minutes).")
                    time.sleep(2.5)
                    operation = client.operations.get(operation.name)

                if not operation.response or not operation.response.generated_videos:
                    raise Exception("Veo API operation completed, but returned no video.")

                generated_video = operation.response.generated_videos[0]
                video_bytes = client.files.download(file=generated_video.video)

            except Exception as e:
                raise Exception(f"Veo video generation failed: {e}")

        if not video_bytes:
            raise Exception("No video bytes downloaded from the API.")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_mp4:
            temp_mp4.write(video_bytes)
            temp_mp4_path = temp_mp4.name

        try:
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

            return (frames_tensor, audio_output if audio_output is not None else {"waveform": torch.zeros(1, 1, 1), "sample_rate": 44100}, session_out)

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

