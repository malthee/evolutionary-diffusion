"""CPU reference using the original evaluator's NumPy L2 normalization."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from evolutionary_imaging.evaluators import AestheticsImageEvaluator, _normalized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluator = AestheticsImageEvaluator(
        device="cpu",
        model_path=str(args.cache / "aesthetics/sac+logos+ava1-l14-linearMSE.pth"),
        clip_model_path=str(args.cache / "clip/ViT-L-14.pt"),
    )
    scores = []
    with torch.inference_mode():
        for path in sorted(args.images.glob("reference-*.png")):
            with Image.open(path) as image:
                features = evaluator.clip_model.encode_image(
                    evaluator.preprocess(image).unsqueeze(0)
                ).float()
            normalized = _normalized(features.cpu().detach())
            scores.append(evaluator.model(normalized).item())
    args.output.write_text(json.dumps(scores))


if __name__ == "__main__":
    main()
