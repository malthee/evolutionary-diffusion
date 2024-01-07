from typing import Union, Tuple
from evolution_base import Evaluator
from image_base import ImageSolutionData


class AestheticsImageEvaluator(Evaluator[ImageSolutionData]):
    def evaluate(self, result: ImageSolutionData) -> float:
        # Implement specific logic
        pass


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
