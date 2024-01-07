from model_helpers.auto_device import auto_to_device
from diffusers import DiffusionPipeline
import torch


def auto_diffusion_pipeline(model_id, variant="fp16", use_safetensors=True, safety_checker=None,
                            requires_safety_checker=False, other_arguments=None):
    """
    Helper function to load a model from the HuggingFace model hub and return a pipeline.
    Tries to load the fp16 variant if available.
    """
    if other_arguments is None:
        other_arguments = {}
    pipe = None

    try:
        pipe = DiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, variant=variant,
            use_safetensors=use_safetensors, safety_checker=safety_checker,
            requires_safety_checker=requires_safety_checker,
            **other_arguments
        )
        pipe = auto_to_device(pipe)
        print(f"Loaded {pipe}")
    except Exception as e:
        print(f"Could not load {model_id}: {e}")

    return pipe
