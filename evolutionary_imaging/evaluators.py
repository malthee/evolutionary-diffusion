import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
import clip
from torchvision.transforms.functional import pil_to_tensor
from typing import Union, Tuple, Literal, Dict, Any, Callable
from torchmetrics.multimodal import CLIPScore, CLIPImageQualityAssessment
from transformers import pipeline

from evolutionary.evolution_base import SingleObjectiveEvaluator, SingleObjectiveFitness, MultiObjectiveFitness, \
    Evaluator
from evolutionary_imaging.image_base import ImageSolutionData
from evolutionary_model_helpers.auto_device import auto_clip_device, load_torch_model

_model_cache: Dict[str, Any] = {}
"""
Cache for models used in evaluation to avoid keeping multiple copies in memory.
Can be cleared with `clear_model_cache`.
"""


def clear_model_cache():
    """
    Clear the model cache of all evaluators to free up memory.
    """
    global _model_cache
    _model_cache.clear()


def get_or_create_model(model_id: str, creator: Callable[[], Any]) -> Any:
    """
    Get a model from the cache or create it with the creator function.
    """
    global _model_cache
    if model_id not in _model_cache:
        _model_cache[model_id] = creator()
    return _model_cache[model_id]


def _normalized(a, axis=-1, order=2):
    """
    Utility function for normalizing an array with axis and order.
    """
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


class AestheticsImageEvaluator(SingleObjectiveEvaluator[ImageSolutionData]):
    DEFAULT_MODEL_PATH = "./models/sac+logos+ava1-l14-linearMSE.pth"
    MODEL_URL = ("https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14"
                 "-linearMSE.pth")
    CLIP_MODEL_NAME = "ViT-L/14"
    CLIP_EMBEDDING_SIZE = 768

    class _MLP(pl.LightningModule):
        def __init__(self, input_size):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, 1024),
                nn.Dropout(0.2),
                nn.Linear(1024, 128),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.Dropout(0.1),
                nn.Linear(64, 16),
                nn.Linear(16, 1)
            )

        def forward(self, x):
            return self.layers(x)

    def _setup_model(self, model_path: str):
        predictor_model = load_torch_model(model_path=model_path, url=AestheticsImageEvaluator.MODEL_URL,
                                           device=self.device)
        model = AestheticsImageEvaluator._MLP(input_size=AestheticsImageEvaluator.CLIP_EMBEDDING_SIZE)
        model.load_state_dict(predictor_model)
        model.to(self.device)
        model.eval()
        clip_model, preprocess = clip.load(AestheticsImageEvaluator.CLIP_MODEL_NAME, device=self.device)
        return model, clip_model, preprocess

    def __init__(self,
                 device: torch.device = auto_clip_device(),
                 model_path: str = DEFAULT_MODEL_PATH):
        self.device = device
        self.model, self.clip_model, self.preprocess = get_or_create_model(f"AestheticsImageEvaluator_{model_path}",
                                                                           lambda: self._setup_model(model_path))

    @torch.no_grad()
    def evaluate(self, result: ImageSolutionData) -> SingleObjectiveFitness:
        scores = []
        for img in result.images:
            image = self.preprocess(img).unsqueeze(0).to(self.device)
            image_features = self.clip_model.encode_image(image)
            im_emb_arr = _normalized(image_features.cpu().detach())
            prediction = self.model(im_emb_arr.to(self.device))
            scores.append(prediction.item())
        return np.mean(scores) if scores else 0.0


class AIDetectionImageEvaluator(SingleObjectiveEvaluator[ImageSolutionData]):
    """
    Evaluate the AI-likeliness of an image.
    Maximizes human-likeness.
    """

    SupportedModels = Literal["umm-maybe/AI-image-detector", "Organika/sdxl-detector"]
    """
    The original AI-image-detector was designed for detecting VQGAN+CLIP
    the sdxl-detector was fine-tuned on SDXL generated images.
    """

    def _setup_model(self, model: SupportedModels):
        return pipeline("image-classification", model=model, device=self.device)

    def __init__(self,
                 device: torch.device = auto_clip_device(),
                 model: SupportedModels = "Organika/sdxl-detector"):
        self.device = device
        self.model = get_or_create_model(f"AIDetectionImageEvaluator_{model}",
                                         lambda: self._setup_model(model))

    @torch.no_grad()
    def evaluate(self, result: ImageSolutionData) -> SingleObjectiveFitness:
        scores = []
        for img in result.images:
            predictions = self.model(img)
            for pred in predictions:
                if pred["label"] == "human":
                    scores.append(pred["score"] * 100)  # Convert from percentages for better comparison
                    break
        return np.mean(scores) if scores else 0.0


class CLIPScoreEvaluator(SingleObjectiveEvaluator[ImageSolutionData]):
    """
    Evaluates the CLIP score of an image.
    https://lightning.ai/docs/torchmetrics/stable/multimodal/clip_score.html
    Supported CLIP models are defined in `SupportedCLIPModels`.
    """

    SupportedCLIPModels = Literal["openai/clip-vit-base-patch16", "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14-336", "openai/clip-vit-large-patch14"]

    def _setup_model(self, model_name_or_path: SupportedCLIPModels):
        return CLIPScore(model_name_or_path=model_name_or_path)

    def __init__(self, prompt: str, clip_model: SupportedCLIPModels = "openai/clip-vit-base-patch16"):
        self._prompt = prompt
        self._model = get_or_create_model(f"CLIPScoreEvaluator_{clip_model}",
                                          lambda: self._setup_model(clip_model))

    @torch.no_grad()
    def evaluate(self, result: ImageSolutionData) -> SingleObjectiveFitness:
        scores = []
        for img in result.images:
            t = pil_to_tensor(img)
            score = self._model(t, self._prompt)
            scores.append(score.item())
        return np.mean(scores) if scores else 0.0


class MultiCLIPIQAEvaluator(Evaluator[ImageSolutionData, MultiObjectiveFitness]):
    """
    Evaluates the CLIP Image Quality Assessment score of images for multiple objectives.
    For single-objective evaluation, use the SingleCLIPIQAEvaluator.
    https://lightning.ai/docs/torchmetrics/stable/multimodal/clip_iqa.html
    """

    SupportedCLIPModels = Literal['clip_iqa', 'openai/clip-vit-base-patch16', 'openai/clip-vit-base-patch32',
                                  'openai/clip-vit-large-patch14-336', 'openai/clip-vit-large-patch14']

    SupportedMetricLiteral = Literal[
        "quality", "brightness", "noisiness", "colorfulness",
        "sharpness", "contrast", "complexity", "natural",
        "happy", "scary", "new", "warm", "real",
        "beautiful", "lonely", "relaxing"
    ]

    SupportedMetrics = Union[SupportedMetricLiteral, Tuple[str, str]]
    """
    Metrics are either predefined or a tuple of two strings (positive, negative) for custom prompts.
    """

    def _setup_model(self, model: SupportedCLIPModels, metrics: Tuple[SupportedMetrics, ...]):
        # noinspection PyTypeChecker
        return CLIPImageQualityAssessment(model_name_or_path=model, prompts=metrics)  # type is correct

    def __init__(self, metrics: Tuple[SupportedMetrics, ...],
                 clip_model: SupportedCLIPModels = "openai/clip-vit-base-patch16"):
        self._metrics = metrics
        self._model = get_or_create_model(f"MultiCLIPIQAEvaluator_{clip_model}_{metrics}",
                                          lambda: self._setup_model(clip_model, metrics))

    @torch.no_grad()
    def evaluate(self, result: ImageSolutionData) -> MultiObjectiveFitness:
        scores = [[] for _ in range(len(self._metrics))]  # Initialize empty lists for each metric

        for img in result.images:
            t = pil_to_tensor(img).unsqueeze(0)
            score_dict = self._model(t)

            if isinstance(score_dict, dict):
                for i, score in enumerate(score_dict.values()):
                    scores[i].append(score.item())
            else:  # Only single tensor value
                scores[0].append(score_dict.item())

        return [np.mean(scores[i]) for i in range(len(self._metrics))]


class SingleCLIPIQAEvaluator(SingleObjectiveEvaluator[ImageSolutionData]):
    """
    Evaluates the CLIP Image Quality Assessment score of images based on a single metric.
    For multi-objective evaluation, use the MultiCLIPIQAEvaluator.
    https://lightning.ai/docs/torchmetrics/stable/multimodal/clip_iqa.html
    """

    def _setup_model(self, metric: MultiCLIPIQAEvaluator.SupportedMetrics,
                     clip_model: MultiCLIPIQAEvaluator.SupportedCLIPModels):
        return MultiCLIPIQAEvaluator(metrics=(metric,), clip_model=clip_model)

    def __init__(self, metric: MultiCLIPIQAEvaluator.SupportedMetrics,
                 clip_model: MultiCLIPIQAEvaluator.SupportedCLIPModels = "openai/clip-vit-base-patch16"):
        # Initialize the MultiCLIPIQAEvaluator with a single metric
        self._multi_evaluator = get_or_create_model(f"SingleCLIPIQAEvaluator_{clip_model}_{metric}",
                                                    lambda: self._setup_model(metric, clip_model))

    @torch.no_grad()
    def evaluate(self, result: ImageSolutionData) -> float:
        scores = self._multi_evaluator.evaluate(result)
        return scores[0]  # Return the score for the single metric
