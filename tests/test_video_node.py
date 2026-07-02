import os
import sys

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nodes.gemini import ETNodesGeminiApiVideo

def test_video_node_metadata():
    # Verify node signature and categories
    assert ETNodesGeminiApiVideo.NODE_NAME == "ETNodes Gemini API Video"
    assert ETNodesGeminiApiVideo.CATEGORY == "ETNodes"
    assert ETNodesGeminiApiVideo.RETURN_TYPES == ("VIDEO", "IMAGE", "AUDIO", "GEMINI_SESSION",)
    assert ETNodesGeminiApiVideo.RETURN_NAMES == ("video", "images", "audio", "session",)
    assert ETNodesGeminiApiVideo.FUNCTION == "execute"

    # Verify input types are correct
    inputs = ETNodesGeminiApiVideo.INPUT_TYPES()
    assert "required" in inputs
    assert "optional" in inputs
    
    required = inputs["required"]
    assert "prompt" in required
    assert "model" in required
    assert "aspect_ratio" in required
    assert "duration_seconds" in required
    assert "resolution" in required
    assert "generate_audio" in required
    assert "safety_level" in required
    assert "seed" in required

    optional = inputs["optional"]
    assert "API_key" in optional
    assert "negative_prompt" in optional
    assert "system_prompt" in optional
    assert "image_first_frame" in optional
    assert "image_last_frame" in optional
    assert "reference_images" in optional
    assert "audio" in optional
    assert "video" in optional
    assert "session" in optional

    print("ETNodes Gemini Video Node: Metadata assertions passed successfully!")

if __name__ == "__main__":
    test_video_node_metadata()
