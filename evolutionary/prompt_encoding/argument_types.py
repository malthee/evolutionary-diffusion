import torch


class PromptEmbedData:
    """
    Simple prompt embeddings data class.
    """
    def __init__(self, prompt_embeds: torch.Tensor):
        self._prompt_embeds = prompt_embeds

    @property
    def prompt_embeds(self) -> torch.Tensor:
        return self._prompt_embeds


class PooledPromptEmbedData(PromptEmbedData):
    """
    Prompt embeddings data class with additional pooled prompt embeddings.
    Used by SDXL models.
    """
    def __init__(self, prompt_embeds: torch.Tensor, pooled_prompt_embeds: torch.Tensor):
        super().__init__(prompt_embeds)
        self._pooled_prompt_embeds = pooled_prompt_embeds

    @property
    def pooled_prompt_embeds(self) -> torch.Tensor:
        return self._pooled_prompt_embeds
