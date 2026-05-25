# ETNodes by Edvard Toth - https://edvardtoth.com

from .nodes import gemini, utilities

NODE_CLASS_MAPPINGS = {
    **gemini.NODE_CLASS_MAPPINGS,
    **utilities.NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **gemini.NODE_DISPLAY_NAME_MAPPINGS,
    **utilities.NODE_DISPLAY_NAME_MAPPINGS,
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
