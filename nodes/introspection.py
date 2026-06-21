import json
from typing import Any

class AnyToString:
    """Converts any input type to a string. Useful for connecting sampler/scheduler outputs from various custom nodes."""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    OUTPUT_TOOLTIPS = ("String representation of the input",)
    FUNCTION = "convert"
    CATEGORY = "ImageSaver/utils"
    DESCRIPTION = "Converts any input type to string"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "value": ("*",),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    def convert(self, value: Any) -> tuple[str,]:
        return (str(value),)


class WorkflowInputValue:
    """Extracts an input value from the workflow by node ID and input name."""

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("value",)
    OUTPUT_TOOLTIPS = ("Input value from the specified node",)
    FUNCTION = "get_input_value"
    CATEGORY = "ImageSaver/utils"
    DESCRIPTION = "Extract an input value from the workflow by node ID and input name"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "node_id": ("STRING", {"default": "", "multiline": False, "tooltip": "The ID of the node to extract from"}),
                "input_name": ("STRING", {"default": "", "multiline": False, "tooltip": "The name of the input to extract"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    def get_input_value(self, node_id: str, input_name: str, prompt: dict[str, Any] | None = None, extra_pnginfo: dict[str, Any] | None = None):
        if prompt is None:
            return (None,)

        # Verify the node exists in the workflow structure
        if extra_pnginfo and "workflow" in extra_pnginfo:
            workflow = extra_pnginfo["workflow"]
            node_exists = any(str(node.get("id")) == node_id for node in workflow.get("nodes", []))
            if not node_exists:
                print(f"WorkflowInputValue: Node {node_id} not found in workflow structure")
                return (None,)

        # Get the node from the prompt (execution values)
        node = prompt.get(node_id)
        if node is None:
            print(f"WorkflowInputValue: Node {node_id} not found in prompt")
            return (None,)

        # Get the inputs from the node
        inputs = node.get("inputs", {})
        if input_name not in inputs:
            print(f"WorkflowInputValue: Input '{input_name}' not found in node {node_id}")
            print(f"WorkflowInputValue: Available inputs: {list(inputs.keys())}")
            return (None,)

        value = inputs[input_name]
        return (value,)


# --- Multi-binding resolver ---------------------------------------------------
#
# Instead of wiring loaders/selectors through the graph to route values into the
# saver, declare a list of bindings — `field: #node.input` — and resolve them all
# from the live PROMPT at save time. The workflow JSON already holds every value;
# this just addresses into it. See WorkflowInputValue above for the single-field
# version this generalises.


def parse_bindings(text: str) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Parse a multi-line binding spec into (field, node_id, input_name) tuples.

    Each line is `field: #node_id.input_name` (the `:` may be `=`, the `#` is
    optional). Blank lines and lines starting with `#` or `//` are ignored.
    Returns (bindings, errors); malformed lines are skipped and reported.
    """
    bindings: list[tuple[str, str, str]] = []
    errors: list[str] = []

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        # A leading '#' is a comment only when it isn't an inline field binding.
        if line.startswith("#") and not _has_separator(line):
            continue

        sep_idx = _separator_index(line)
        if sep_idx is None:
            errors.append(f"line {lineno}: missing ':' or '=' separator — '{raw}'")
            continue

        field = line[:sep_idx].strip()
        pointer = line[sep_idx + 1:].strip().lstrip("#").strip()
        node_id, dot, input_name = pointer.partition(".")
        node_id = node_id.strip()
        input_name = input_name.strip()

        if not field:
            errors.append(f"line {lineno}: empty field name — '{raw}'")
            continue
        if not dot or not node_id or not input_name:
            errors.append(f"line {lineno}: pointer must be 'node_id.input_name' — '{raw}'")
            continue

        bindings.append((field, node_id, input_name))

    return bindings, errors


def _has_separator(line: str) -> bool:
    return _separator_index(line) is not None


def _separator_index(line: str) -> int | None:
    """Index of the field/pointer separator — the earliest ':' or '='."""
    candidates = [line.index(c) for c in (":", "=") if c in line]
    return min(candidates) if candidates else None


def _is_link(value: Any) -> bool:
    """A ComfyUI link is `[node_id: str, output_slot: int]` — distinct from a
    literal list like `[width, height]` (both ints)."""
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
    )


# Keys under which primitive/literal nodes carry their scalar value in the PROMPT.
_LITERAL_KEYS = ("value", "int", "float", "string", "text", "boolean", "number")


def _follow_link(prompt: dict[str, Any], value: Any, depth: int = 0) -> Any:
    """Resolve a value to a literal, following links through the PROMPT graph.

    A direct literal returns as-is. A link is followed to its source node; if that
    node is a primitive carrying a scalar (or a text node carrying `text`), that
    value is returned (recursively). Non-scalar outputs resolve to None.
    """
    if not _is_link(value):
        return value
    if depth > 16:
        return None  # cycle or pathologically deep chain — cannot resolve to a literal

    source = prompt.get(value[0])
    if not isinstance(source, dict):
        return None

    inputs = source.get("inputs", {})
    for key in _LITERAL_KEYS:
        if key in inputs:
            return _follow_link(prompt, inputs[key], depth + 1)
    return None


def resolve_bindings(
    bindings: list[tuple[str, str, str]],
    prompt: dict[str, Any],
    workflow: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Resolve parsed bindings against the PROMPT. Returns (resolved, errors).

    `workflow` (the UI graph from EXTRA_PNGINFO) is used only to give a clearer
    error when a node id is absent from the graph entirely.
    """
    resolved: dict[str, Any] = {}
    errors: list[str] = []

    workflow_ids: set[str] | None = None
    if isinstance(workflow, dict) and isinstance(workflow.get("nodes"), list):
        workflow_ids = {str(n.get("id")) for n in workflow["nodes"] if isinstance(n, dict)}

    for field, node_id, input_name in bindings:
        node = prompt.get(node_id)
        if node is None:
            where = "workflow" if workflow_ids is not None and node_id not in workflow_ids else "prompt"
            errors.append(f"{field}: node #{node_id} not found in {where}")
            continue

        inputs = node.get("inputs", {})
        if input_name not in inputs:
            available = ", ".join(inputs.keys()) or "(none)"
            errors.append(f"{field}: input '{input_name}' not on node #{node_id} — available: {available}")
            continue

        resolved[field] = _follow_link(prompt, inputs[input_name])

    return resolved, errors


class WorkflowMetadataResolver:
    """Resolve many `field: #node.input` bindings from the live workflow at once.

    A wiring-free alternative to the loader/selector nodes: point at where each
    value lives in the graph and this fetches them all from the PROMPT at save
    time, emitting a gallery-metadata dict ready for the Image Saver.
    """

    RETURN_TYPES = ("METADATA", "STRING")
    RETURN_NAMES = ("metadata", "gallery_metadata_json")
    OUTPUT_TOOLTIPS = (
        "Resolved metadata for Image Saver (plug into its 'metadata' input)",
        "Resolved bindings as a JSON string",
    )
    FUNCTION = "resolve"
    CATEGORY = "ImageSaver/utils"
    DESCRIPTION = "Resolve multiple workflow fields by node-id pointers into gallery metadata"

    @classmethod
    def IS_CHANGED(cls, **kwargs) -> float:
        # The pointed-at nodes are not wired inputs, so they don't enter this
        # node's cache key. Force re-execution every run so resolved values
        # always reflect the current PROMPT rather than a cached binding string.
        return float("nan")

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "bindings": ("STRING", {
                    "default": "// Right-click any node -> 'Send to Metadata Resolver',\n"
                               "// or use the 'Auto-fill from sampler' button below.\n"
                               "// One binding per line:  field: #node_id.input",
                    "multiline": True,
                    "tooltip": "One binding per line: `field: #node_id.input_name`.\n"
                               "Separator may be ':' or '='; the '#' is optional.\n"
                               "Lines starting with '#' or '//' are comments.",
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    def resolve(
        self,
        bindings: str,
        prompt: dict[str, Any] | None = None,
        extra_pnginfo: dict[str, Any] | None = None,
    ):
        parsed, parse_errors = parse_bindings(bindings)
        for err in parse_errors:
            print(f"WorkflowMetadataResolver: {err}")

        if not prompt:
            print("WorkflowMetadataResolver: no PROMPT available; returning empty metadata")
            return (_build_metadata({}), "{}")

        workflow = extra_pnginfo.get("workflow") if isinstance(extra_pnginfo, dict) else None
        resolved, resolve_errors = resolve_bindings(parsed, prompt, workflow)
        for err in resolve_errors:
            print(f"WorkflowMetadataResolver: {err}")

        return (_build_metadata(resolved), json.dumps(resolved))


# Aliases mapping resolved field names onto the Metadata dataclass attributes used
# for filename templating. Resolved values are also embedded verbatim as
# gallery_metadata, so unknown field names are preserved — these only feed the
# scalar attributes the saver reads for `%seed`, `%steps`, etc.
_METADATA_ALIASES = {
    "model": "model_name", "model_name": "model_name", "model_path": "model_name",
    "positive": "positive", "negative": "negative",
    "width": "width", "height": "height",
    "seed": "seed", "steps": "steps", "cfg": "cfg",
    "sampler": "sampler_name", "sampler_name": "sampler_name",
    "scheduler": "scheduler_name", "scheduler_name": "scheduler_name",
    "denoise": "denoise", "clip_skip": "clip_skip",
}


_INT_ATTRS = {"width", "height", "seed", "steps", "clip_skip"}
_FLOAT_ATTRS = {"cfg", "denoise"}


def map_to_metadata_attrs(resolved: dict[str, Any]) -> dict[str, Any]:
    """Map a resolved dict onto Metadata scalar attributes for filename templating.

    Recognised field names (and aliases) are coerced to the attribute's type;
    a `size: [w, h]` value is unpacked onto width/height. Unrecognised or
    uncoercible fields are dropped — they still travel verbatim in
    gallery_metadata, this only feeds the saver's `%seed`, `%steps`, etc.
    """
    size = resolved.get("size")
    if isinstance(size, list) and len(size) == 2:
        resolved = {**resolved, "width": size[0], "height": size[1]}

    attrs: dict[str, Any] = {}
    for field, value in resolved.items():
        attr = _METADATA_ALIASES.get(field)
        if attr is None or value is None:
            continue
        try:
            attrs[attr] = _coerce(attr, value)
        except (ValueError, TypeError):
            pass
    return attrs


def _coerce(attr: str, value: Any) -> Any:
    if attr in _INT_ATTRS:
        return int(value)
    if attr in _FLOAT_ATTRS:
        return float(value)
    return value if isinstance(value, str) else str(value)


def _build_metadata(resolved: dict[str, Any]):
    """Wrap a resolved dict in a Metadata object for the Image Saver.

    The resolved dict becomes gallery_metadata verbatim; recognised field names
    also populate the scalar attributes the saver uses for filename templating.
    Imported lazily so this module stays free of ComfyUI (`folder_paths`) deps
    and the resolution helpers above remain unit-testable in isolation.
    """
    from .metadata import Metadata

    meta = Metadata(
        model_name="", positive="", negative="", width=512, height=512, seed=0,
        steps=20, cfg=7.0, sampler_name="", scheduler_name="normal", denoise=1.0,
        clip_skip=0, additional_hashes="", ckpt_path="",
        gallery_metadata=dict(resolved),
    )
    for attr, value in map_to_metadata_attrs(resolved).items():
        setattr(meta, attr, value)
    return meta
