"""Experiment variation pools and validated optional operator settings."""

import numpy as np

from evolutionary.operators import MultiCrossover, MultiMutator
from evolutionary_prompt_embedding import variation as v


def operators(bounds, initial, config=None):
    token = (bounds["token_min"], bounds["token_max"])
    pooled = (bounds["pooled_min"], bounds["pooled_max"])
    crossover = MultiCrossover(
        {
            "coordinate_uniform": v.PooledUniformCrossover(0.5, 0.5),
            "row_uniform": v.PooledRowUniformCrossover(0.5),
            "one_point": v.PooledOnePointCrossover(),
            "two_point": v.PooledTwoPointCrossover(),
            "arithmetic": v.PooledArithmeticCrossover(0.5, 0.5, 1.0, 1.0),
            "slerp": v.PooledSlerpCrossover(0.5, 0.5, token, pooled),
            "blx_alpha": v.PooledBLXAlphaCrossover(token, pooled, alpha=0.5),
            "sbx": v.PooledSBXCrossover(token, pooled, index=15, participation=0.5),
        }
    )
    mutator = MultiMutator(
        {
            "gaussian": v.PooledUniformGaussianMutator(
                v.UniformGaussianMutatorArguments(0.2, 2.0, token),
                v.UniformGaussianMutatorArguments(0.2, 0.4, pooled),
            ),
            "spherical": v.PooledSphericalRotationMutator(
                0.1, 0.1, 0.1, 0.1, token, pooled
            ),
            "donor": v.PooledDonorReplacementMutator(initial, 0.1, token, pooled),
            "polynomial": v.PooledPolynomialMutator(
                token, pooled, index=20, participation=0.1
            ),
        }
    )
    parameters = {
        "coordinate_uniform": {"swap_rate": 0.5, "swap_rate_pooled": 0.5},
        "row_uniform": {"swap_rate": 0.5},
        "one_point": {"prompt_axis": "row", "pooled_axis": "coordinate"},
        "two_point": {"prompt_axis": "row", "pooled_axis": "coordinate"},
        "arithmetic": {
            "interpolation_weight": 0.5,
            "interpolation_weight_pooled": 0.5,
            "proportion": 1.0,
            "proportion_pooled": 1.0,
        },
        "slerp": {"ratio": 0.5, "ratio_pooled": 0.5},
        "blx_alpha": {"alpha": 0.5},
        "sbx": {"index": 15, "participation": 0.5},
        "gaussian": {
            "prompt_participation": 0.2,
            "prompt_strength": 2.0,
            "pooled_participation": 0.2,
            "pooled_strength": 0.4,
        },
        "spherical": {"participation": 0.1, "angle": 0.1},
        "donor": {"participation": 0.1, "bank": "frozen initial population"},
        "polynomial": {"index": 20, "participation": 0.1},
    }
    if config is not None:
        validate_operator_settings(
            config.crossover_weights,
            config.mutation_weights,
            config.operator_parameters,
        )
        parameters = {name: dict(values) for name, values in parameters.items()}
        for name, values in (config.operator_parameters or {}).items():
            parameters[name].update(values)
        if config.operator_parameters:
            g = parameters["gaussian"]
            mutator.operators["gaussian"] = v.PooledUniformGaussianMutator(
                v.UniformGaussianMutatorArguments(
                    g["prompt_participation"], g["prompt_strength"], token
                ),
                v.UniformGaussianMutatorArguments(
                    g["pooled_participation"], g["pooled_strength"], pooled
                ),
            )
            sp = parameters["spherical"]
            mutator.operators["spherical"] = v.PooledSphericalRotationMutator(
                sp["participation"],
                sp["angle"],
                sp["participation"],
                sp["angle"],
                token,
                pooled,
            )

        def pool(cls, original, weights):
            selected = (
                original.operators
                if weights is None
                else {
                    name: op
                    for name, op in original.operators.items()
                    if name in weights
                }
            )
            return cls(
                selected,
                None if weights is None else [weights[name] for name in selected],
            )

        crossover = pool(MultiCrossover, crossover, config.crossover_weights)
        mutator = pool(MultiMutator, mutator, config.mutation_weights)
        parameters = {
            name: values
            for name, values in parameters.items()
            if name in crossover.operators or name in mutator.operators
        }
    return crossover, mutator, parameters


def validate_operator_settings(cross, mutation, parameters):
    names = {
        "crossover": {
            "coordinate_uniform",
            "row_uniform",
            "one_point",
            "two_point",
            "arithmetic",
            "slerp",
            "blx_alpha",
            "sbx",
        },
        "mutation": {"gaussian", "spherical", "donor", "polynomial"},
    }
    for label, weights in [("crossover", cross), ("mutation", mutation)]:
        if weights is not None and (
            not isinstance(weights, dict)
            or not weights
            or set(weights) - names[label]
            or any(
                isinstance(w, bool)
                or not isinstance(w, (int, float))
                or not np.isfinite(w)
                or w <= 0
                for w in weights.values()
            )
            or not np.isfinite(sum(weights.values()))
        ):
            raise ValueError(f"Invalid {label} pool weights")
    allowed = {
        "gaussian": {
            "prompt_participation",
            "pooled_participation",
            "prompt_strength",
            "pooled_strength",
        },
        "spherical": {"participation", "angle"},
    }
    if parameters is None:
        return
    if not isinstance(parameters, dict) or set(parameters) - set(allowed):
        raise ValueError("Only Gaussian/spherical local parameters are configurable")
    for name, values in parameters.items():
        if not isinstance(values, dict) or set(values) - allowed[name]:
            raise ValueError("Unknown operator parameter")
        for key, value in values.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value < 0
                or ("participation" in key and value > 1)
                or (key == "angle" and value > np.pi)
            ):
                raise ValueError("Invalid operator parameter value")
