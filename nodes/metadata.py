# ComfyUI-Image-Saver — Metadata Compiler Node
# Assembles generation metadata into structured form + A1111-compatible parameters string.

import os
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import folder_paths
from nodes import MAX_RESOLUTION

from ..services.hashing import get_sha256
from ..services.file_utils import full_checkpoint_path_for, parse_checkpoint_name_without_extension
from ..services.civitai import get_civitai_sampler_name, get_civitai_metadata, MAX_HASH_LENGTH
from ..services.prompt_parser import PromptMetadataExtractor


@dataclass
class Metadata:
    model_name: str
    positive: str
    negative: str
    width: int
    height: int
    seed: int
    steps: int
    cfg: float
    sampler_name: str
    scheduler_name: str
    denoise: float
    clip_skip: int
    custom: str
    additional_hashes: str
    ckpt_path: str
    a111_params: str
    final_hashes: str


class MetadataCompiler:
    """Compiles generation metadata into structured form + A1111-compatible parameters string."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "optional": {
                "model_name":            ("STRING",  {"default": '', "multiline": False,                           "tooltip": "model name (can be multiple, separated by commas)"}),
                "positive":              ("STRING",  {"default": 'unknown', "multiline": True,                     "tooltip": "positive prompt"}),
                "negative":              ("STRING",  {"default": 'unknown', "multiline": True,                     "tooltip": "negative prompt"}),
                "width":                 ("INT",     {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8,  "tooltip": "image width"}),
                "height":                ("INT",     {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8,  "tooltip": "image height"}),
                "seed_value":            ("INT",     {"default": 0, "min": 0, "max": 0xffffffffffffffff,           "tooltip": "seed"}),
                "steps":                 ("INT",     {"default": 20, "min": 1, "max": 10000,                       "tooltip": "number of steps"}),
                "cfg":                   ("FLOAT",   {"default": 7.0, "min": 0.0, "max": 100.0,                    "tooltip": "CFG value"}),
                "sampler_name":          ("STRING",  {"default": '', "multiline": False,                           "tooltip": "sampler name (as string)"}),
                "scheduler_name":        ("STRING",  {"default": 'normal', "multiline": False,                     "tooltip": "scheduler name (as string)"}),
                "denoise":               ("FLOAT",   {"default": 1.0, "min": 0.0, "max": 1.0,                      "tooltip": "denoise value"}),
                "clip_skip":             ("INT",     {"default": 0, "min": -24, "max": 24,                         "tooltip": "skip last CLIP layers (positive or negative value, 0 for no skip)"}),
                "additional_hashes":     ("STRING",  {"default": "", "multiline": False,                           "tooltip": "hashes separated by commas, optionally with names. 'Name:HASH' (e.g., 'MyLoRA:FF735FF83F98')\nWith download_civitai_data set to true, weights can be added as well. (e.g., 'HASH:Weight', 'Name:HASH:Weight')"}),
                "download_civitai_data": ("BOOLEAN", {"default": True,                                             "tooltip": "Download and cache data from civitai.com to save correct metadata. Allows LoRA weights to be saved to the metadata."}),
                "easy_remix":            ("BOOLEAN", {"default": True,                                             "tooltip": "Strip LoRAs and simplify 'embedding:path' from the prompt to make the Remix option on civitai.com more seamless."}),
                "custom":                ("STRING",  {"default": "", "multiline": False,                           "tooltip": "custom string to add to the metadata, inserted into the a111 string between clip skip and model hash"}),
            },
        }

    RETURN_TYPES = ("METADATA", "STRING", "STRING")
    RETURN_NAMES = ("metadata", "hashes", "a1111_params")
    OUTPUT_TOOLTIPS = ("metadata for Image Saver", "Comma-separated list of the hashes to chain with other Image Saver additional_hashes", "Written parameters to the image metadata")
    FUNCTION = "get_metadata"
    CATEGORY = "ImageSaver"
    DESCRIPTION = "Prepare metadata for Image Saver"

    def get_metadata(
        self,
        model_name: str = "",
        positive: str = "unknown",
        negative: str = "unknown",
        width: int = 512,
        height: int = 512,
        seed_value: int = 0,
        steps: int = 20,
        cfg: float = 7.0,
        sampler_name: str = "",
        scheduler_name: str = "normal",
        denoise: float = 1.0,
        clip_skip: int = 0,
        custom: str = "",
        additional_hashes: str = "",
        download_civitai_data: bool = True,
        easy_remix: bool = True,
    ) -> tuple[Metadata, str, str]:
        metadata = MetadataCompiler.make_metadata(model_name, positive, negative, width, height, seed_value, steps, cfg, sampler_name, scheduler_name, denoise, clip_skip, custom, additional_hashes, download_civitai_data, easy_remix)
        return (metadata, metadata.final_hashes, metadata.a111_params)

    @staticmethod
    def make_metadata(model_name: str, positive: str, negative: str, width: int, height: int, seed_value: int, steps: int, cfg: float, sampler_name: str, scheduler_name: str, denoise: float, clip_skip: int, custom: str, additional_hashes: str, download_civitai_data: bool, easy_remix: bool) -> Metadata:
        model_name, additional_hashes = get_multiple_models(model_name, additional_hashes)

        ckpt_path = full_checkpoint_path_for(model_name)
        if ckpt_path:
            modelhash = get_sha256(ckpt_path)[:10]
        else:
            modelhash = ""

        metadata_extractor = PromptMetadataExtractor([positive, negative])
        embeddings = metadata_extractor.get_embeddings()
        loras = metadata_extractor.get_loras()
        civitai_sampler_name = get_civitai_sampler_name(sampler_name.replace('_gpu', ''), scheduler_name)
        basemodelname = parse_checkpoint_name_without_extension(model_name)

        # Get existing hashes from model, loras, and embeddings
        existing_hashes = {modelhash.lower()} | {t[2].lower() for t in loras.values()} | {t[2].lower() for t in embeddings.values()}
        # Parse manual hashes
        manual_entries = parse_manual_hashes(additional_hashes, existing_hashes, download_civitai_data)
        # Get Civitai metadata
        civitai_resources, hashes, add_model_hash = get_civitai_metadata(model_name, ckpt_path, modelhash, loras, embeddings, manual_entries, download_civitai_data)

        if easy_remix:
            positive = clean_prompt(positive, metadata_extractor)
            negative = clean_prompt(negative, metadata_extractor)

        positive_a111_params = positive.strip()
        negative_a111_params = f"\nNegative prompt: {negative.strip()}"
        clip_skip_str = f", Clip skip: {abs(clip_skip)}" if clip_skip != 0 else ""
        custom_str = f", {custom}" if custom else ""
        model_hash_str = f", Model hash: {add_model_hash}" if add_model_hash else ""
        hashes_str = f", Hashes: {json.dumps(hashes, separators=(',', ':'))}" if hashes else ""

        a111_params = (
            f"{positive_a111_params}{negative_a111_params}\n"
            f"Steps: {steps}, Sampler: {civitai_sampler_name}, Scheduler: {scheduler_name}, "
            f"CFG scale: {cfg}, Seed: {seed_value}, "
            f"Size: {width}x{height}{clip_skip_str}{custom_str}{model_hash_str}, Model: {basemodelname}{hashes_str}, Version: ComfyUI"
        )

        # Add Civitai resource listing
        if download_civitai_data and civitai_resources:
            a111_params += f", Civitai resources: {json.dumps(civitai_resources, separators=(',', ':'))}"

        # Combine all resources for final hash string
        all_resources = { model_name: ( ckpt_path, None, modelhash ) } | loras | embeddings | manual_entries

        hash_parts = []
        for name, (_, weight, hash_value) in (all_resources.items() if isinstance(all_resources, dict) else all_resources):
            if name:
                filename = name.split(':')[-1]
                name_without_ext, ext = os.path.splitext(filename)
                supported_extensions = folder_paths.supported_pt_extensions | {".gguf"}

                if ext.lower() in supported_extensions:
                    clean_name = name_without_ext
                else:
                    clean_name = filename

                name_part = f"{clean_name}:"
            else:
                name_part = ""

            if not hash_value:
                continue

            weight_part = f":{weight}" if weight is not None and download_civitai_data else ""
            hash_parts.append(f"{name_part}{hash_value}{weight_part}")

        final_hashes = ",".join(hash_parts)

        metadata = Metadata(model_name, positive, negative, width, height, seed_value, steps, cfg, sampler_name, scheduler_name, denoise, clip_skip, custom, additional_hashes, ckpt_path, a111_params, final_hashes)
        return metadata


# --- Helper functions (extracted from ImageSaver static methods) ---

# Match 'anything' or 'anything:anything' with trimmed white space
re_manual_hash = re.compile(r'^\s*([^:]+?)(?:\s*:\s*([^\s:][^:]*?))?\s*$')
# Match 'anything', 'anything:anything' or 'anything:anything:number' with trimmed white space
re_manual_hash_weights = re.compile(r'^\s*([^:]+?)(?:\s*:\s*([^\s:][^:]*?))?(?:\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)))?\s*$')


def get_multiple_models(model_name: str, additional_hashes: str) -> tuple[str, str]:
    """Parse comma-separated model names. First becomes primary, rest added to additional_hashes."""
    model_names = [m.strip() for m in model_name.split(',')]
    model_name = model_names[0]

    for additional_model in model_names[1:]:
        additional_ckpt_path = full_checkpoint_path_for(additional_model)
        if additional_ckpt_path:
            additional_modelhash = get_sha256(additional_ckpt_path)[:10]
            if additional_hashes:
                additional_hashes += ","
            additional_hashes += f"{additional_model}:{additional_modelhash}"
    return model_name, additional_hashes


def parse_manual_hashes(additional_hashes: str, existing_hashes: set[str], download_civitai_data: bool) -> dict[str, tuple[str | None, float | None, str]]:
    """Process additional_hashes input string into normalized dict."""
    manual_entries: dict[str, tuple[str | None, float | None, str]] = {}
    unnamed_count = 0

    additional_hash_split = additional_hashes.replace("\n", ",").split(",") if additional_hashes else []
    for entry in additional_hash_split:
        match = (re_manual_hash_weights if download_civitai_data else re_manual_hash).search(entry)
        if match is None:
            print(f"ComfyUI-Image-Saver: Invalid additional hash string: '{entry}'")
            continue

        groups = tuple(group for group in match.groups() if group)

        weight = None
        if download_civitai_data and len(groups) > 1:
            try:
                weight = float(groups[-1])
                groups = groups[:-1]
            except (ValueError, TypeError):
                pass

        name, hash = groups if len(groups) > 1 else (None, groups[0])

        if len(hash) > MAX_HASH_LENGTH:
            print(f"ComfyUI-Image-Saver: Skipping hash. Length exceeds maximum of {MAX_HASH_LENGTH} characters: {hash}")
            continue

        if any(hash.lower() == existing_hash.lower() for _, _, existing_hash in manual_entries.values()):
            print(f"ComfyUI-Image-Saver: Skipping duplicate hash: {hash}")
            continue

        if hash.lower() in existing_hashes:
            print(f"ComfyUI-Image-Saver: Skipping manual hash already present in resources: {hash}")
            continue

        if name is None:
            unnamed_count += 1
            name = f"manual{unnamed_count}"
        elif name in manual_entries:
            print(f"ComfyUI-Image-Saver: Duplicate manual hash name '{name}' is being overwritten.")

        manual_entries[name] = (None, weight, hash)

        if len(manual_entries) > 29:
            print("ComfyUI-Image-Saver: Reached maximum limit of 30 manual hashes. Skipping the rest.")
            break

    return manual_entries


def clean_prompt(prompt: str, metadata_extractor: PromptMetadataExtractor) -> str:
    """Clean prompts for easier remixing by removing LoRAs and simplifying embeddings."""
    prompt = re.sub(metadata_extractor.LORA, "", prompt)
    prompt = re.sub(metadata_extractor.EMBEDDING, lambda match: Path(match.group(1)).stem, prompt)
    prompt = re.sub(r'\b[A-Z]+\([^)]*\)', "", prompt)
    return prompt
