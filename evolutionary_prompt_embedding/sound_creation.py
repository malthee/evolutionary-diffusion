from abc import ABC, abstractmethod
from typing import Any

from diffusers import AudioLDMPipeline

from evolutionary_model_helpers.auto_device import auto_device
from evolutionary_prompt_embedding.argument_types import PooledPromptEmbedData, PromptEmbedData
from evolutionary.evolution_base import SolutionCandidate
from evolutionary_imaging.image_base import A
import torch
import soundfile as sf
import os
import uuid

from evolutionary_sound.sound_base import SoundCreator, SoundSolutionData

SAMPLE_RATE = 16000
RESULT_FOLDER = "results"

class PromptEmbeddingSoundCreator(SoundCreator[A], ABC):
    @abstractmethod
    def arguments_from_prompt(self, prompt: str) -> A:
        """
        Creates prompt embeddings from a prompt string using the pipeline's tokenizer and text encoder.
        Useful for initializing the first generation of an evolutionary algorithm.
        """
        pass

class AudioLDMSoundCreator(PromptEmbeddingSoundCreator[PooledPromptEmbedData]):
    """
    A class that creates sound solutions from prompt embeddings using the AudioLDM pipeline.
    """

    def __init__(self,
                 inference_steps: int,
                 batch_size: int,
                 deterministic: bool = True,
                 audio_length_s: int = 5,
                 guidance_scale: float = 1.0, # Guidance makes the evolutionary process more complex (form of embeddings), disable it
                 model_id: str = "cvssp/audioldm-l-full"):
        """
        Initializes the SoundCreator with the given parameters.
        By default, uses the "cvssp/audioldm-l-full" model.
        """
        self._model_id = model_id
        self._guidance_scale = guidance_scale
        super().__init__(inference_steps, batch_size, audio_length_s, deterministic)

    def _setup_pipeline(self) -> AudioLDMPipeline:
        pipe = AudioLDMPipeline.from_pretrained(self._model_id, torch_dtype=torch.float16)
        pipe.set_progress_bar_config(disable=True)
        device = auto_device()
        return pipe.to(device)

    def arguments_from_prompt(self, prompt: str) -> PromptEmbedData:
        with torch.no_grad():
            prompt_embeds = self._pipeline._encode_prompt(
                prompt,
                self._pipeline.device,
                num_waveforms_per_prompt=self._batch_size,
                do_classifier_free_guidance=self._guidance_scale > 1.0,
            )
            return PromptEmbedData(prompt_embeds=prompt_embeds)


    def create_solution(self, argument: PromptEmbedData) \
            -> SolutionCandidate[PromptEmbedData, SoundSolutionData, Any]:
        prompt_embeds = argument.prompt_embeds.unsqueeze(0)

        sounds = None
        try:
            sounds = self._pipeline(
                prompt_embeds=prompt_embeds,
                num_inference_steps=self._inference_steps,
                num_waveforms_per_prompt=self._batch_size,
                generator=self._generators,
                audio_length_in_s=self._audio_length_s,
                guidance_scale=self._guidance_scale,
            ).audios
        except Exception as e:
            # This most likely happens because an out of memory error, so we reinitialize the pipeline and retry
            print(f"Sound generation failed, retrying once: {e}")
            self._pipeline = self._setup_pipeline()
            sounds = self._pipeline(
                prompt_embeds=prompt_embeds,
                num_inference_steps=self._inference_steps,
                num_waveforms_per_prompt=self._batch_size,
                generator=self._generators,
                audio_length_in_s=self._audio_length_s,
                guidance_scale=self._guidance_scale,
            ).audios

        # Create output directory if it doesn't exist
        os.makedirs(RESULT_FOLDER, exist_ok=True)

        # Save each sound with a unique filename
        sound_paths = []
        for i, audio in enumerate(sounds):
            # Generate a unique filename
            uid = str(uuid.uuid4())[:8]
            filename = f"sound_{uid}_{i}.wav"
            filepath = os.path.join(RESULT_FOLDER, filename)

            # Save the sound file
            sf.write(filepath, audio, SAMPLE_RATE)
            sound_paths.append(filepath)

        return SolutionCandidate(argument, SoundSolutionData(sound_paths))
