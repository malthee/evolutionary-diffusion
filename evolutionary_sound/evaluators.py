from typing import Dict, List, Literal, Union, Tuple, Any
import numpy as np
import torch

from audiobox_aesthetics.infer import initialize_predictor
from evolutionary.evolution_base import SingleObjectiveEvaluator, MultiObjectiveFitness, Evaluator
from evolutionary_sound.sound_base import SoundSolutionData

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


def get_or_create_model(model_id: str, creator: callable) -> Any:
    """
    Get a model from the cache or create it with the creator function.
    """
    global _model_cache
    if model_id not in _model_cache:
        _model_cache[model_id] = creator()
    return _model_cache[model_id]


class MultiAudioboxAestheticsEvaluator(Evaluator[SoundSolutionData, MultiObjectiveFitness]):
    """
    Computes multiple Audiobox-Aesthetics metrics for a batch of sounds.

    Metrics are selected from the fixed set: "CE", "CU", "PC", "PQ".
    """

    SupportedMetricLiteral = Literal["CE", "CU", "PC", "PQ"]
    SupportedMetrics = Tuple[SupportedMetricLiteral, ...]
    ALL_METRICS: SupportedMetrics = ("CE", "CU", "PC", "PQ")

    def _setup_model(self) -> Any:
        return initialize_predictor()

    def __init__(self,
                 metrics: SupportedMetrics = ALL_METRICS):
        self._metrics = metrics
        tag = f"MultiAudioboxAesthetics_{metrics}"
        self._model = get_or_create_model(tag, self._setup_model)

    @torch.no_grad()
    def evaluate(self, result: SoundSolutionData) -> MultiObjectiveFitness:
        scores: Dict[str, List[float]] = {m: [] for m in self._metrics}

        for path in result.sounds:
            preds: List[Dict[str, float]] = self._model.forward([{"path": path}])
            p = preds[0]
            for m in self._metrics:
                scores[m].append(p[m])

        return [float(np.mean(scores[m])) if scores[m] else 0.0
                for m in self._metrics]


class AudioboxAestheticsEvaluator(SingleObjectiveEvaluator[SoundSolutionData]):
    """
    Evaluate a single Audiobox-Aesthetics metric, the average, or the sum of all.

    Uses MultiAudioboxAestheticsEvaluator under the hood.
    """

    SupportedMetric = Union[MultiAudioboxAestheticsEvaluator.SupportedMetricLiteral, Literal["average", "sum"]]

    def _setup_model(self,
                     metric: SupportedMetric) -> MultiAudioboxAestheticsEvaluator:
        if metric in ("average", "sum"):
            m = MultiAudioboxAestheticsEvaluator.ALL_METRICS
        else:
            m = (metric,)
        return MultiAudioboxAestheticsEvaluator(metrics=m)

    def __init__(self, metric: SupportedMetric = "average"):
        self._metric = metric
        tag = f"SingleAudioboxAesthetics_{metric}"
        self._multi = get_or_create_model(tag,
                                          lambda: self._setup_model(metric))

    @torch.no_grad()
    def evaluate(self, result: SoundSolutionData) -> float:
        vals = self._multi.evaluate(result)
        if self._metric == "average":
            return float(np.mean(vals))
        if self._metric == "sum":
            return float(np.sum(vals))
        return float(vals[0])
