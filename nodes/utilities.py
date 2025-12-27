#ETNodes Utility Nodes by Edvard Toth - https://edvardtoth.com

class ETNodesListItems:
    """A node that converts a multiline string into a list of strings."""
    NODE_NAME = "ETNodes List Items"
    CATEGORY = "ETNodes/utils"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "", "tooltip": "Enter list items, one per line."}),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("items",)
    FUNCTION = "execute"

    def execute(self, text):
        items = [line.strip() for line in text.splitlines() if line.strip()]
        return (items,)

class ETNodesListSelector:
    """A node that selects an item from a list by index."""
    NODE_NAME = "ETNodes List Selector"
    CATEGORY = "ETNodes/utils"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "items": ("*",),
                "item": ("INT", {"default": 1, "min": 1, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"

    def execute(self, items, item):
        if not items or not isinstance(items, list):
            return ("",)
        
        # Adjust for 1-based index and clamp to the list size
        clamped_index = max(0, min(item - 1, len(items) - 1))
        selected_item = str(items[clamped_index])

        return (selected_item,)

class ETNodesColorSelector:
    """A node that provides a dropdown for selecting a color."""
    NODE_NAME = "ETNodes Color Selector"
    CATEGORY = "ETNodes/utils"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "color": (["RED", "BLUE", "GREEN", "YELLOW", "PURPLE", "ORANGE", "BROWN", "TEAL", "PINK", "WHITE", "BLACK", "GRAY"],),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"

    def execute(self, color):
        return (color,)

class ETNodesTextPreview:
    """A minimal text preview node."""
    NODE_NAME = "ETNodes Text Preview"
    CATEGORY = "ETNodes/utils"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    OUTPUT_NODE = True

    def execute(self, text):
        return {"ui": {"text": [text]}, "result": (text,)}

NODE_CLASS_MAPPINGS = {
    "ETNodes-List-Items": ETNodesListItems,
    "ETNodes-List-Selector": ETNodesListSelector,
    "ETNodes-Color-Selector": ETNodesColorSelector,
    "ETNodes-Text-Preview": ETNodesTextPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {


    # Title formatting is intentional, don't alter

    "ETNodes-List-Items": "ETNodes List Items",
    "ETNodes-List-Selector": "ETNodes List Selector",
    "ETNodes-Color-Selector": "ETNodes Color Selector",
    "ETNodes-Text-Preview": "ETNodes Text Preview",
}

