from abc import abstractmethod, ABC
from typing import List, Any
from PIL import Image
from diffusers import DiffusionPipeline
from evolutionary.evolution_base import SolutionCreator, SolutionCandidate, A
import torch

from evolutionary_model_helpers.auto_pipeline import auto_diffusion_pipeline


class SoundSolutionData:
    def __init__(self, sounds: List[str]):
        self._sounds = sounds

    @property
    def sounds(self) -> List[str]:
        """
        Returns the list of sound file paths in the solution.
        """
        return self._sounds


class SoundCreator(SolutionCreator[A, SoundSolutionData], ABC):
    """Base class for sound creators. Sound creators create sound solutions from arguments."""

    def __init__(self,
                 inference_steps: int,
                 batch_size: int,
                 audio_length_s: int = 5,
                 deterministic: bool = True):
        """
        :param inference_steps: The number of inference steps to perform. Directly affects quality.
        :param batch_size: The batch size (number of sounds per prompt). Affects evaluation - averages out
        evolution across multiple images.
        :param audio_length_s: The length of the generated audio in seconds. Default is 5 seconds.
        :param deterministic: Whether to use a deterministic seed for the generation process, recommended for
        evolutionary exploration.
        """
        self._pipeline = self._setup_pipeline()
        self._inference_steps = inference_steps
        self._batch_size = batch_size
        self._audio_length_s = audio_length_s
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
        self._pipeline = self._setup_pipeline()
        self._generators = [torch.Generator(device=self._pipeline.device).manual_seed(i)
                            for i in range(self._batch_size)] if self._deterministic else None

    @abstractmethod
    def _setup_pipeline(self) -> DiffusionPipeline:
        """
        Reusable function to set up the pipeline. This is called in the constructor and on error.
        Has to be implemented by subclasses.
        """
        pass

    @abstractmethod
    def create_solution(self, argument: A) -> SolutionCandidate[A, SoundSolutionData, Any]:
        """
        Create a sound solution from the given arguments. On error should create a new pipeline and retry once.
        Otherwise raises the error.
        """
        pass
