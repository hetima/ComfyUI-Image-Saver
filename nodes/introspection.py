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
