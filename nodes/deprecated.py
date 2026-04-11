import warnings

# These nodes are deprecated and will be removed in a future release.
# They are not related to the core image saving functionality.

# Import the original implementations
import torch
import numpy as np
from PIL import Image, ImageDraw
import math
import random
import csv
import os
import requests
from typing import Any
from nodes import MAX_RESOLUTION


class ConditioningConcatOptional:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "conditioning_to": ("CONDITIONING", {"tooltip": "base conditioning to concat to (or pass through, if second is empty)"}),
            },
            "optional": {
                "conditioning_from": ("CONDITIONING", {"tooltip": "conditioning to concat to conditioning_to, if empty, then conditioning_to is passed through unchanged"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "concat"
    CATEGORY = "conditioning"

    def concat(self, conditioning_to, conditioning_from=None):
        warnings.warn("ConditioningConcatOptional is deprecated and will be removed in a future release.", DeprecationWarning, stacklevel=2)
        if conditioning_from is None:
            return (conditioning_to,)

        out = []
        if len(conditioning_from) > 1:
            print("Warning: ConditioningConcat conditioning_from contains more than 1 cond, only the first one will actually be applied to conditioning_to.")

        cond_from = conditioning_from[0][0]
        for i in range(len(conditioning_to)):
            t1 = conditioning_to[i][0]
            tw = torch.cat((t1, cond_from), 1)
            n = [tw, conditioning_to[i][1].copy()]
            out.append(n)

        return (out,)


class RandomShapeGenerator:
    """
    A ComfyUI node that generates images with random shapes.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "width": ("INT", { "default": 512, "min": 64, "max": 4096, "step": 64, "tooltip": "Width of the generated image in pixels" }),
                "height": ("INT", { "default": 512, "min": 64, "max": 4096, "step": 64, "tooltip": "Height of the generated image in pixels" }),
                "bg_color": (["random", "white", "black", "red", "green", "blue", "yellow", "cyan", "magenta"], { "tooltip": "Background color preset or random" }),
                "fg_color": (["random", "black", "white", "red", "green", "blue", "yellow", "cyan", "magenta"], { "tooltip": "Foreground shape color preset or random" }),
                "shape_type": (["random", "circle", "oval", "triangle", "square", "rectangle", "rhombus", "pentagon", "hexagon"], { "tooltip": "Type of shape to generate or random" }),
                "seed": ("INT", { "default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "Random seed for reproducible shape generation" }),
            },
            "optional": {
                "bg_color_override": ("STRING", { "default": "", "multiline": False, "tooltip": "Override background color with hex (#AABBCC) or RGB(r, g, b) format" }),
                "fg_color_override": ("STRING", { "default": "", "multiline": False, "tooltip": "Override foreground color with hex (#AABBCC) or RGB(r, g, b) format" }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "bg_rgb", "fg_rgb")
    OUTPUT_TOOLTIPS = ("Generated image with random shape", "Background color as RGB/hex", "Foreground color as RGB/hex")
    FUNCTION = "generate_shape"
    CATEGORY = "image/generators"
    DESCRIPTION = "Generates images with random shapes for testing and prototyping"

    def __init__(self):
        self.color_map = {
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "cyan": (0, 255, 255),
            "magenta": (255, 0, 255),
        }

    def parse_rgb_string(self, rgb_str: str) -> tuple[int, int, int] | None:
        """Parse RGB string like 'RGB(123, 45, 67)' or '#AABBCC' into tuple (123, 45, 67)"""
        if not rgb_str or rgb_str.strip() == "":
            return None

        rgb_str = rgb_str.strip()

        try:
            # Try hex format first (#AABBCC or AABBCC)
            if rgb_str.startswith("#"):
                hex_str = rgb_str[1:]
            else:
                hex_str = rgb_str

            # Check if it's a valid hex string (6 characters)
            if len(hex_str) == 6 and all(c in '0123456789ABCDEFabcdef' for c in hex_str):
                r = int(hex_str[0:2], 16)
                g = int(hex_str[2:4], 16)
                b = int(hex_str[4:6], 16)
                return (r, g, b)

            # Try RGB(r, g, b) format
            rgb_str_upper = rgb_str.upper()
            if rgb_str_upper.startswith("RGB(") and rgb_str_upper.endswith(")"):
                values = rgb_str[4:-1].split(",")
                r, g, b = [int(v.strip()) for v in values]
                # Validate range
                if all(0 <= val <= 255 for val in [r, g, b]):
                    return (r, g, b)
        except (ValueError, IndexError):
            return None

        return None

    def draw_shape(self, draw: ImageDraw.ImageDraw, img_width: int, img_height: int, shape_type: str, shape_color: tuple[int, int, int]) -> None:
        """Draw a random shape on the image."""

        # Random size - prefer larger sizes (40-70% of image dimensions)
        size_factor = random.uniform(0.4, 0.7)
        shape_width = int(img_width * size_factor)
        shape_height = int(img_height * size_factor)

        # Random position (ensure shape stays fully within bounds)
        x = random.randint(0, max(0, img_width - shape_width))
        y = random.randint(0, max(0, img_height - shape_height))

        # Draw the shape based on type
        if shape_type == 'circle':
            # Make it a perfect circle using the minimum dimension
            radius = min(shape_width, shape_height) // 2
            draw.ellipse([x, y, x + radius * 2, y + radius * 2], fill=shape_color)

        elif shape_type == 'oval':
            draw.ellipse([x, y, x + shape_width, y + shape_height], fill=shape_color)

        elif shape_type == 'square':
            # Make it a perfect square
            side = min(shape_width, shape_height)
            draw.rectangle([x, y, x + side, y + side], fill=shape_color)

        elif shape_type == 'rectangle':
            draw.rectangle([x, y, x + shape_width, y + shape_height], fill=shape_color)

        elif shape_type == 'triangle':
            # Equilateral-ish triangle
            points = [
                (x + shape_width // 2, y),  # top
                (x, y + shape_height),  # bottom left
                (x + shape_width, y + shape_height)  # bottom right
            ]
            draw.polygon(points, fill=shape_color)

        elif shape_type == 'rhombus':
            # Diamond shape
            points = [
                (x + shape_width // 2, y),  # top
                (x + shape_width, y + shape_height // 2),  # right
                (x + shape_width // 2, y + shape_height),  # bottom
                (x, y + shape_height // 2)  # left
            ]
            draw.polygon(points, fill=shape_color)

        elif shape_type == 'pentagon':
            # Regular pentagon
            cx, cy = x + shape_width // 2, y + shape_height // 2
            radius = min(shape_width, shape_height) // 2
            points = []
            for i in range(5):
                angle = i * 2 * math.pi / 5 - math.pi / 2
                px = cx + radius * math.cos(angle)
                py = cy + radius * math.sin(angle)
                points.append((px, py))
            draw.polygon(points, fill=shape_color)

        elif shape_type == 'hexagon':
            # Regular hexagon
            cx, cy = x + shape_width // 2, y + shape_height // 2
            radius = min(shape_width, shape_height) // 2
            points = []
            for i in range(6):
                angle = i * 2 * math.pi / 6
                px = cx + radius * math.cos(angle)
                py = cy + radius * math.sin(angle)
                points.append((px, py))
            draw.polygon(points, fill=shape_color)

    def generate_shape(self, width: int, height: int, bg_color: str, fg_color: str, shape_type: str, seed: int, bg_color_override: str = "", fg_color_override: str = "") -> tuple[torch.Tensor, str, str]:
        """Generate an image with a random shape."""
        warnings.warn("RandomShapeGenerator is deprecated and will be removed in a future release.", DeprecationWarning, stacklevel=2)

        # Set random seed for reproducibility
        random.seed(seed)

        # Get colors from map or generate random RGB values
        # Check for override first
        bg_override = self.parse_rgb_string(bg_color_override)
        if bg_override is not None:
            bg_rgb = bg_override
        elif bg_color == "random":
            bg_rgb = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        else:
            bg_rgb = self.color_map.get(bg_color, (255, 255, 255))

        fg_override = self.parse_rgb_string(fg_color_override)
        if fg_override is not None:
            fg_rgb = fg_override
        elif fg_color == "random":
            fg_rgb = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        else:
            fg_rgb = self.color_map.get(fg_color, (0, 0, 0))

        # Create image
        img = Image.new('RGB', (width, height), bg_rgb)
        draw = ImageDraw.Draw(img)

        # Select shape type
        if shape_type == "random":
            shapes = ['circle', 'oval', 'triangle', 'square', 'rectangle', 'rhombus', 'pentagon', 'hexagon']
            selected_shape = random.choice(shapes)
        else:
            selected_shape = shape_type

        # Draw the shape
        self.draw_shape(draw, width, height, selected_shape, fg_rgb)

        # Convert PIL Image to torch tensor (ComfyUI format)
        # ComfyUI expects images in format [batch, height, width, channels] with values 0-1
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)[None,]

        # Format RGB values as strings for output (both formats)
        bg_hex = f"#{bg_rgb[0]:02X}{bg_rgb[1]:02X}{bg_rgb[2]:02X}"
        fg_hex = f"#{fg_rgb[0]:02X}{fg_rgb[1]:02X}{fg_rgb[2]:02X}"
        bg_rgb_str = f"RGB({bg_rgb[0]}, {bg_rgb[1]}, {bg_rgb[2]}) / {bg_hex}"
        fg_rgb_str = f"RGB({fg_rgb[0]}, {fg_rgb[1]}, {fg_rgb[2]}) / {fg_hex}"

        return (img_tensor, bg_rgb_str, fg_rgb_str)


class CivitaiHashFetcher:
    """
    A ComfyUI custom node that fetches the AutoV3 hash of a model from Civitai
    based on the provided username and model name.
    """

    def __init__(self):
        self.last_username = None
        self.last_model_name = None
        self.last_version = None
        self.last_hash = None  # Store the last fetched hash

    RETURN_TYPES = ("STRING",)  # The node outputs a string (AutoV3 hash)
    FUNCTION = "get_autov3_hash"
    CATEGORY = "CivitaiAPI"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "username": ("STRING", {"default": "", "multiline": False}),
                "model_name": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "version": ("STRING", {"default": "", "multiline": False, "tooltip": "Specify version keyword to fetch a particular model version (optional)"}),
            }
        }

    def get_autov3_hash(self, username, model_name, version=""):
        """
        Fetches the latest model version from Civitai and extracts its AutoV3 hash.
        Uses caching to avoid redundant API calls.
        """
        warnings.warn("CivitaiHashFetcher is deprecated and will be removed in a future release.", DeprecationWarning, stacklevel=2)

        # Check if inputs are the same as last time
        if (self.last_username is not None and self.last_model_name is not None and self.last_version is not None and
            username == self.last_username and model_name == self.last_model_name and version == self.last_version):
            return self.last_hash

        base_url = "https://civitai.com/api/v1/models"
        params = {
            "username": username,
            "query": model_name,
            "limit": 20,  # Fetch more results due to API ranking issues
            "nsfw": "true"  # Include NSFW models in search results
        }

        try:
            # Fetch models by username and model name
            response = requests.get(base_url, params=params, timeout=10)
            if response.status_code != 200:
                return (f"Error: API request failed with status {response.status_code}",)

            data = response.json()
            items = data.get("items", [])

            # If no results with query, try without query (fallback for API search issues)
            if not items and params.get("query"):
                print("ComfyUI-Image-Saver: No results with query, trying without query parameter...")
                params_no_query = {
                    "username": username,
                    "limit": 100,
                    "nsfw": "true"
                }
                response = requests.get(base_url, params=params_no_query, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    items = data.get("items", [])

            if not items:
                return (f"No models found for user '{username}' with name '{model_name}'",)

            # Find best matching model (prefer exact/partial matches)
            model_name_lower = model_name.lower()
            best_match = None

            # Try exact match first
            for item in items:
                if item.get("name", "").lower() == model_name_lower:
                    best_match = item
                    break

            # If no exact match, try partial match
            if not best_match:
                for item in items:
                    item_name_lower = item.get("name", "").lower()
                    if model_name_lower in item_name_lower or item_name_lower.startswith(model_name_lower):
                        best_match = item
                        break

            # Fall back to first result if no good match
            if not best_match:
                best_match = items[0]

            model = best_match
            model_versions = model.get("modelVersions", [])
            if not model_versions:
                return ("No model versions found.",)

            # If a version keyword is provided, search for a model version whose name contains it (case-insensitive).
            chosen_version = None
            if version:
                for v in model_versions:
                    if version.lower() in v.get("name", "").lower():
                        chosen_version = v
                        break
            # If no version is provided or no match was found, use the first (latest) version.
            if chosen_version is None:
                chosen_version = model_versions[0]
            version_id = chosen_version.get("id")

            # Fetch detailed version info
            version_url = f"https://civitai.com/api/v1/model-versions/{version_id}"
            version_response = requests.get(version_url, timeout=10)
            if version_response.status_code != 200:
                return (f"Error: Version API request failed with status {version_response.status_code}",)

            version_data = version_response.json()

            # Extract the AutoV3 hash from the model version files
            for file_info in version_data.get("files", []):
                autov3_hash = file_info.get("hashes", {}).get("AutoV3")
                if autov3_hash:
                    # Cache the result before returning
                    self.last_username = username
                    self.last_model_name = model_name
                    self.last_version = version  # Store version to track changes
                    self.last_hash = autov3_hash
                    return (autov3_hash,)  # Return the first found hash

            return ("No AutoV3 hash found in version files.",)

        except Exception as e:
            return (f"Error: {e}",)


class RandomTagPicker:
    """Pick N random tags from a CSV file (first column) and join them with a delimiter."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "", "multiline": False}),
                "count": ("INT", {"default": 5, "min": 1, "max": 1000, "step": 1}),
                "delimiter": ("STRING", {"default": ", ", "multiline": False}),
                "replace_underscore": ("BOOLEAN", {"default": False}),
                "trailing_comma": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags",)
    FUNCTION = "pick_random_tags"
    CATEGORY = "utils"

    def pick_random_tags(self, file_path: str, count: int, delimiter: str, replace_underscore: bool, trailing_comma: bool, seed: int) -> tuple[str]:
        warnings.warn("RandomTagPicker is deprecated and will be removed in a future release.", DeprecationWarning, stacklevel=2)
        with open(os.path.expanduser(file_path), newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            if rows and rows[0][0].strip().lower() == "tag":
                rows = rows[1:]

        tags = [row[0] for row in rows if row and row[0].strip()]
        sample_size = min(count, len(tags))
        rng = random.Random(seed)
        selected = rng.sample(tags, sample_size)
        if replace_underscore:
            selected = [t.replace("_", " ") for t in selected]
        escaped = [t.replace("(", "\\(").replace(")", "\\)") for t in selected]
        result = delimiter.join(escaped)
        if trailing_comma:
            result += ","
        return (result,)
