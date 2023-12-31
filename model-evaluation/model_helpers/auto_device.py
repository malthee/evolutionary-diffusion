import torch
import os
import urllib.request

"""
This module contains helper functions to automatically select the best device for torch and
instantiates models and pipelines with it.
"""


def auto_clip_device():
    """
    Since CLIP is not optimized for mps, this will exclude it and otherwise work the same as auto_device()
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def auto_device():
    """
    Gets the most appropriate device as string in the order "cuda" -> "mps" -> "cpu"
    """
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def auto_to_device(model_or_pipeline, attention_slicing=False):
    """
    Automatically moves the given PyTorch model or pipeline to the most appropriate device.
    It checks for CUDA or MPS availability and defaults to CPU if neither is available.

    :param model_or_pipeline: A PyTorch model or pipeline.
    :param attention_slicing: If True, attention slicing is enabled (MAC: Recommended if your computer has < 64 GB of RAM, may lead to black images in some cases)
    :return: The model or pipeline moved to the appropriate device.
    """

    device = auto_device()
    pipe = model_or_pipeline.to(device)
    if attention_slicing: pipe.enable_attention_slicing();
    return pipe


def auto_generator(seed=None):
    """
    Automatically returns a generator for the installed device.

    :param seed: The seed to use for the generator. If None, the default generator is returned.
    :return: A generator.
    """

    device = auto_device()
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    return generator


def auto_batch_pipeline_arguments(prompt, inference_steps=1, batch_size=1, deterministic=True):
    """
     Returns a dictionary of pipeline arguments for the installed device.
     Guidance is disabled, as it is not supported by all models.

    :param prompt: The prompt to use for the pipeline.
    :param inference_steps: The number of inference steps to use for the pipeline.
    :param batch_size: How many images to generate in parallel.
    :param deterministic: If True, the pipeline will be deterministic and use a fixed seed for the images.
    :return: A dictionary of pipeline arguments.
    """

    pipeline_args = {
        "prompt": [prompt] * batch_size,
        "num_inference_steps": inference_steps,
        "guidance_scale": 0.0  # Guidance disabled
    }

    if deterministic:
        pipeline_args["generator"] = [auto_generator(i) for i in range(batch_size)]

    return pipeline_args


def load_torch_model(model_path, url, device=auto_device()):
    """
    Loads a PyTorch model. Downloads the model from a URL if it's not present locally.

    :param model_path: Path where the model is saved or will be saved.
    :param url: URL to download the model if not present locally.
    :param device: Device the model is loaded on.
    :return: Loaded PyTorch model.
    """
    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Model file not found. Downloading from {url}...")
        # Create the models directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        # Download the file from `url` and save it locally under `model_path`
        urllib.request.urlretrieve(url, model_path)

    # Load the model
    model = torch.load(model_path, map_location=torch.device(device))
    print("Model loaded successfully.")
    return model
