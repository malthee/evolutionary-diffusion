import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
import clip
from typing import Union, Tuple
from evolutionary.evolution_base import Evaluator
from evolutionary.image_base import ImageSolutionData
from model_helpers.auto_device import auto_clip_device, load_torch_model


def _normalized(a, axis=-1, order=2):
    """
    Utility function for normalizing an array with axis and order.
    """
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


class AestheticsImageEvaluator(Evaluator[ImageSolutionData]):
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

    def __init__(self,
                 device: torch.device = auto_clip_device(),
                 model_path: str = DEFAULT_MODEL_PATH):
        self.device = device
        predictor_model = load_torch_model(model_path=model_path, url=self.MODEL_URL, device=self.device)
        self.model = self._MLP(input_size=self.CLIP_EMBEDDING_SIZE)
        self.model.load_state_dict(predictor_model)
        self.model.to(self.device)
        self.model.eval()

        self.clip_model, self.preprocess = clip.load(self.CLIP_MODEL_NAME, device=self.device)

    def evaluate(self, result: ImageSolutionData) -> float:
        scores = []
        for img in result.images:
            image = self.preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image)
                im_emb_arr = _normalized(image_features.cpu().detach())
                prediction = self.model(im_emb_arr.to(self.device))
                scores.append(prediction.item())
        return np.mean(scores) if scores else 0.0


class AIDetectionImageEvaluator(Evaluator[ImageSolutionData]):
    def evaluate(self, result: ImageSolutionData) -> float:
        # Implement specific logic
        pass


class CLIPScoreEvaluator(Evaluator[ImageSolutionData]):
    def __init__(self, prompt: str):
        self.prompt = prompt

    def evaluate(self, result: ImageSolutionData) -> float:
        # Implement specific logic
        pass


class CLIPIQAEvaluator(Evaluator[ImageSolutionData]):
    def __init__(self, metric: Union[str, Tuple[str, str]]):
        self.metric = metric

    def evaluate(self, result: ImageSolutionData) -> float:
        # Implement specific logic
        pass
