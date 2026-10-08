"""Extract verified full-corpus DiffusionDB bounds, without subset studies or caches."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from evolutionary_prompt_embedding.archive import sha256_file

EXPECTED_SOURCE_SHA256 = (
    "7bd1d4afaaaee5bead3e753cabdc27cfb7ac3fc7053cae9c6ff22c26334f3b53"
)


def extract(source, output):
    source, output = Path(source), Path(output)
    if sha256_file(source) != EXPECTED_SOURCE_SHA256:
        raise ValueError("Expected the verified full DiffusionDB experiment artifact")
    original = torch.load(source, weights_only=True, map_location="cpu")
    full = original["bounds"]["full"]
    if full["count"] != 1528510:
        raise ValueError("Full-corpus count differs")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("Refuse overwriting a bounds input")
    tensors = {
        f"{kind}_{side}": full[kind][side].float().contiguous()
        for kind in ("token", "pooled")
        for side in ("min", "max")
    }
    save_file(tensors, str(output))
    manifest = {
        "dataset": "DiffusionDB 2M",
        "count": full["count"],
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "model_id": "stabilityai/sdxl-turbo",
        "model_revision": "71153311d3dbb46851df1931d3ca6e939de83304",
        "encoding_device": "CUDA",
        "encoding_dtype": "float16",
        "extrema_dtype": "float32",
        "sha256": sha256_file(output),
        "bytes": output.stat().st_size,
        "initialization": "per-coordinate uniform synthetic embeddings; not actual prompt samples",
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(extract(args.source, args.output), indent=2))
