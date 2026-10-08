"""Single-objective prompt-embedding experiments with lazy public imports."""

from importlib import import_module

_EXPORTS = {
    "BoundedPNGWriter": "artifacts",
    "ExperimentConfig": "config",
    "Runtime": "runtime",
    "export_figures": "artifacts",
    "initial_arguments": "initialization",
    "load_bounds": "initialization",
    "operators": "operator_pool",
    "run_experiment": "experiment",
    "validate_artifacts": "artifacts",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value
