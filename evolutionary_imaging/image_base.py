from abc import abstractmethod, ABC
from typing import List, Any
from PIL import Image
from diffusers import DiffusionPipeline
from evolutionary.evolution_base import SolutionCreator, SolutionCandidate, A
import torch

from evolutionary_model_helpers.auto_pipeline import auto_diffusion_pipeline


class ImageSolutionData:
    def __init__(self, images: List[Image.Image]):
        self._images = images

    @property
    def images(self) -> List[Image.Image]:
        return self._images


class ImageCreator(SolutionCreator[A, ImageSolutionData], ABC):
    """Base class for image creators. Image creators create image solutions from arguments."""

    def __init__(self,
                 model_id: str,
                 inference_steps: int,
                 batch_size: int,
                 deterministic: bool = True):
        """
        :param model_id: The model ID to use for image generation. This has to be compatible with the ImageCreator.
        This model is then loaded through the diffusers DiffusionPipeline.
        :param inference_steps: The number of inference steps to perform. Directly affects quality.
        :param batch_size: The batch size (number of images per prompt). Affects evaluation - averages out
        evolution across multiple images.
        :param deterministic: Whether to use a deterministic seed for the diffusion process, recommended for
        evolutionary exploration.
        """
        self._model_id = model_id
        self._pipeline = self._setup_diffusers_pipeline()
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
        self._pipeline = self._setup_diffusers_pipeline()
        self._generators = [torch.Generator(device=self._pipeline.device).manual_seed(i)
                            for i in range(self._batch_size)] if self._deterministic else None

    def _setup_diffusers_pipeline(self) -> DiffusionPipeline:
        """
        Reusable function to set up the diffusion pipeline. This is called in the constructor and on error.
        """
        pipe = auto_diffusion_pipeline(self._model_id)
        pipe.set_progress_bar_config(disable=True)  # Disabling to avoid progress bar spamming
        return pipe

    @abstractmethod
    def create_solution(self, argument: A) -> SolutionCandidate[A, ImageSolutionData, Any]:
        """
        Create an image solution from the given arguments. On error should create a new pipeline and retry once.
        Otherwise, raises the error.
        """
        pass
