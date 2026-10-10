"""Frozen benchmark inputs; importing preparation does not initialize CUDA."""

import hashlib
import json
from pathlib import Path

DETECTORS = ("safe", "universal_fake_detect", "ssp", "bfree", "aeroblade")
CODECS = ("png", "jpeg95")
MODEL_REVISION = "71153311d3dbb46851df1931d3ca6e939de83304"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    if tuple(config["detectors"]) != DETECTORS:
        raise ValueError("Require exactly SAFE, UniFD, SSP, B-Free and AEROBLADE")
    if config.get("model_revision") != MODEL_REVISION:
        raise ValueError("Require the pinned SDXL-Turbo revision")
    for key in ("test_count", "calibration_count", "bootstrap_samples"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("output", "raise_csv", "prompts_csv", "assets"):
        if not config.get(key):
            raise ValueError(f"Missing {key}")
        p = Path(config[key]).expanduser()
        config[key] = str(p if p.is_absolute() else path.parent / p)
    return config


def scientific_identity(config):
    return {
        key: config[key]
        for key in (
            "detectors",
            "model_revision",
            "test_count",
            "calibration_count",
            "bootstrap_samples",
            "sample_seed",
            "diffusion_seed",
            "patch_seed",
        )
    }
