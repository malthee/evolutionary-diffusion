from abc import ABC, abstractmethod
from evolutionary.prompt_encoding.argument_types import PooledPromptEmbedData, PromptEmbedData
from evolutionary.evolution_base import SolutionCandidate
from evolutionary.image_base import ImageCreator, ImageSolutionData, A
import torch


class PromptEmbeddingImageCreator(ImageCreator[A], ABC):
    @abstractmethod
    def arguments_from_prompt(self, prompt: str) -> A:
        """
        Creates prompt embeddings from a prompt string using the pipeline's tokenizer and text encoder.
        """
        pass


class SDXLPromptEmbeddingImageCreator(PromptEmbeddingImageCreator[PooledPromptEmbedData]):
    def create_solution(self, argument: PooledPromptEmbedData) \
            -> SolutionCandidate[PooledPromptEmbedData, ImageSolutionData]:
        images = self._pipeline(
            prompt_embeds=argument.prompt_embeds,
            pooled_prompt_embeds=argument.pooled_prompt_embeds,
            num_inference_steps=self._inference_steps,
            num_images_per_prompt=self._batch_size,
            guidance_scale=0.0, # 0 for Turbo models
            generator=self._generators,
        ).images
        return SolutionCandidate(argument, ImageSolutionData(images))

    def arguments_from_prompt(self, prompt: str) -> PooledPromptEmbedData:
        tokenizer = self._pipeline.tokenizer
        tokenizer_2 = self._pipeline.tokenizer_2
        text_encoder = self._pipeline.text_encoder
        text_encoder_2 = self._pipeline.text_encoder_2

        tokenizers = [tokenizer, tokenizer_2] if tokenizer is not None else [tokenizer_2]
        text_encoders = ([text_encoder, text_encoder_2] if text_encoder is not None else [text_encoder_2])

        # We only use one prompt here, but you could also use two prompts for SDXL
        prompt_2 = prompt
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

        # textual inversion: procecss multi-vector tokens if necessary
        prompt_embeds_list = []
        prompts = [prompt, prompt_2]
        pooled_prompt_embeds = None

        for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids
            untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
            ):
                removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1: -1])
                print(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {tokenizer.model_max_length} tokens: {removed_text}"
                )

            prompt_embeds = text_encoder(text_input_ids.to(self._pipeline.device), output_hidden_states=True)

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            # "2" because SDXL always indexes from the penultimate layer.
            prompt_embeds = prompt_embeds.hidden_states[-2]
            prompt_embeds_list.append(prompt_embeds)
        # end for

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        return PooledPromptEmbedData(prompt_embeds, pooled_prompt_embeds)


class SDPromptEmbeddingImageCreator(PromptEmbeddingImageCreator[PromptEmbedData]):
    def create_solution(self, argument: PromptEmbedData) -> SolutionCandidate[PromptEmbedData, ImageSolutionData]:
        pass

    def arguments_from_prompt(self, prompt: str) -> PromptEmbedData:
        tokenizer = self._pipeline.tokenizer
        text_encoder = self._pipeline.text_encoder

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        # Get the output from the text encoder
        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
        ):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1: -1])
            print(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {tokenizer.model_max_length} tokens: {removed_text}"
            )

        with torch.no_grad():
            prompt_embeds = text_encoder(text_input_ids.to(self._pipeline.device))
            prompt_embeds = prompt_embeds[0]

        return PromptEmbedData(prompt_embeds)
