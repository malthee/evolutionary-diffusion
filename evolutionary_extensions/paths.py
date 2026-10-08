"""Resolve an exported or Git checkout without relying on module nesting depth."""

import os
from pathlib import Path


def checkout_root():
    hint = Path(
        os.environ.get("EVOLUTIONARY_CHECKOUT", Path(__file__).resolve().parent)
    ).resolve()
    for path in (hint, *hint.parents):
        if (path / "setup.py").is_file() and (
            path / "evolutionary_extensions"
        ).is_dir():
            return path
    raise RuntimeError("Set EVOLUTIONARY_CHECKOUT to the experiment checkout")
