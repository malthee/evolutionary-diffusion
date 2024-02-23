from abc import abstractmethod, ABC
from typing import List, Callable, Any, Generic
from PIL import Image
from diffusers import DiffusionPipeline
from evolutionary.evolution_base import SolutionCreator, SolutionCandidate, A
import torch

PipelineFactory = Callable[[], DiffusionPipeline]
"""Factory to allow reinitialization of the pipeline on error."""


class ImageSolutionData:
    def __init__(self, images: List[Image.Image]):
        self._images = images

    @property
    def images(self) -> List[Image.Image]:
        return self._images


class ImageCreator(SolutionCreator[A, ImageSolutionData], ABC):
    """Base class for image creators. Image creators create image solutions from arguments."""

    def __init__(self, pipeline_factory: PipelineFactory,
                 inference_steps: int,
                 batch_size: int,
                 deterministic: bool = True):
        """
        :param pipeline_factory: The diffusion pipeline factory is called once at the start
         and reinitialized on error as a fallback mechanism.
        :param inference_steps: The number of inference steps to perform. Directly affects quality.
        :param batch_size: The batch size (number of images per prompt). Affects evaluation - averages out
        evolution across multiple images.
        :param deterministic: Whether to use a deterministic seed for the diffusion process, recommended for
        evolutionary exploration.
        """
        self._pipeline_factory = pipeline_factory
        self._pipeline = pipeline_factory()
        self._inference_steps = inference_steps
        self._batch_size = batch_size
        self._deterministic = deterministic
        self._generators = [torch.Generator(device=self._pipeline.device).manual_seed(i)
                            for i in range(batch_size)] if deterministic else None

    def __getstate__(self):
        # Exclude the pipeline, generators from pickling
        state = self.__dict__.copy()
        del state['_pipeline']
        del state['_generators']
        return state

    def __setstate__(self, state):
        # Reinitialize the pipeline and generators
        self.__dict__.update(state)
        self._pipeline = self._pipeline_factory()
        self._generators = [torch.Generator(device=self._pipeline.device).manual_seed(i)
                            for i in range(self._batch_size)] if self._deterministic else None

    @abstractmethod
    def create_solution(self, argument: A) -> SolutionCandidate[A, ImageSolutionData, Any]:
        """
        Create an image solution from the given arguments. On error should create a new pipeline and retry once.
        Otherwise, raises the error.
        """
        pass
