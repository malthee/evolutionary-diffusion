from abc import abstractmethod, ABC
from typing import List
from PIL import Image
from diffusers import DiffusionPipeline
from evolutionary.evolution_base import SolutionCreator, SolutionCandidate, A
import torch


class ImageSolutionData:
    def __init__(self, images: List[Image.Image]):
        self._images = images

    @property
    def images(self) -> List[Image.Image]:
        return self._images


class ImageCreator(SolutionCreator[A, ImageSolutionData], ABC):
    """Base class for image creators. Image creators create image solutions from arguments."""

    def __init__(self, pipeline: DiffusionPipeline, inference_steps: int, batch_size: int, deterministic: bool = True):
        """
        :param pipeline: The diffusion pipeline to use for image generation.
        :param inference_steps: The number of inference steps to perform.
        :param batch_size: The batch size (number of images per prompt).
        :param deterministic: Whether to use a deterministic seed for the diffusion process, recommended for
        evolutionary exploration.
        """
        self._pipeline = pipeline
        self._inference_steps = inference_steps
        self._batch_size = batch_size
        self._deterministic = deterministic
        self._generators = [torch.Generator(device=pipeline.device).manual_seed(i)
                            for i in range(batch_size)] if deterministic else None

    @abstractmethod
    def create_solution(self, argument: A) -> SolutionCandidate[A, ImageSolutionData]:
        pass
