from sys import float_info
from typing import Any
from nodes import MAX_RESOLUTION

class SeedGenerator:
    RETURN_TYPES = ("INT",)
    OUTPUT_TOOLTIPS = ("seed (INT)",)
    FUNCTION = "get_seed"

    CATEGORY = "ImageSaver/utils"
    DESCRIPTION = "Provides seed as integer"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "The random seed used for creating the noise."}),
                "increment": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "tooltip": "number to add to the final seed value"}),
            }
        }

    def get_seed(self, seed: int, increment: int) -> tuple[int,]:
        return (seed + increment,)

class StringLiteral:
    RETURN_TYPES = ("STRING",)
    OUTPUT_TOOLTIPS = ("string (STRING)",)
    FUNCTION = "get_string"

    CATEGORY = "ImageSaver/utils"
    DESCRIPTION = "Provides a string"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "string": ("STRING", {"default": "", "multiline": True, "tooltip": "string"}),
            }
        }

    def get_string(self, string: str) -> tuple[str,] :
        return (string,)

class SizeLiteral:
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("size",)
    OUTPUT_TOOLTIPS = ("size (INT)",)
    FUNCTION = "get_int"

    CATEGORY = "ImageSaver/utils"
    DESCRIPTION = f"Provides integer number between 0 and {MAX_RESOLUTION} (step=8)"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "size": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "size as integer (in steps of 8)"}),
            }
        }

    def get_int(self, size: int) -> tuple[int,]:
        return (size,)

class IntLiteral:
    RETURN_TYPES = ("INT",)
    OUTPUT_TOOLTIPS = ("int (INT)",)
    FUNCTION = "get_int"

    CATEGORY = "ImageSaver/utils"
    DESCRIPTION = "Provides integer number between 0 and 1000000"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "int": ("INT", {"default": 0, "min": 0, "max": 1000000, "tooltip": "integer number"}),
            }
        }

    def get_int(self, int: int) -> tuple[int,]:
        return (int,)

class FloatLiteral:
    RETURN_TYPES = ("FLOAT",)
    OUTPUT_TOOLTIPS = ("float (FLOAT)",)
    FUNCTION = "get_float"

    CATEGORY = "ImageSaver/utils"
    DESCRIPTION = f"Provides a floating point number between {float_info.min} and {float_info.max} (step=0.01)"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "float": ("FLOAT", {"default": 1.0, "min": float_info.min, "max": float_info.max, "step": 0.01, "tooltip": "floating point number"}),
            }
        }

    def get_float(self, float: float):
        return (float,)

class CfgLiteral:
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("value",)
    OUTPUT_TOOLTIPS = ("cfg (FLOAT)",)
    FUNCTION = "get_float"

    CATEGORY = "ImageSaver/utils"
    DESCRIPTION = "Provides CFG value between 0.0 and 100.0"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "tooltip": "CFG as a floating point number"}),
            }
        }

    def get_float(self, cfg: float) -> tuple[float,]:
        return (cfg,)
