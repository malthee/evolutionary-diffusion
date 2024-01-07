from model_helpers.auto_device import auto_to_device
from diffusers import DiffusionPipeline
from diffusers.utils import make_image_grid
import pandas as pd
import matplotlib.pyplot as plt
import time
import csv
import torch
from pathlib import Path


def setup_diffusion_pipeline(model_id, variant="fp16"):
    """
    Helper function to load a model from the HuggingFace model hub and return a pipeline.
    Tries to load the fp16 variant if available.
    """
    pipe = None
    try:
        pipe = DiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, variant=variant,
            use_safetensors=True, safety_checker=None, requires_safety_checker=False
        )
        pipe = auto_to_device(pipe)
        print(f"Loaded {pipe}")
    except Exception as e:
        print(f"Could not load {model_id}: {e}")

    return pipe
