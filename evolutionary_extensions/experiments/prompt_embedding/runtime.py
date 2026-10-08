"""Reusable SDXL runtime and strict aesthetic precision preflight."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch

from evolutionary_extensions.paths import checkout_root
from evolutionary_prompt_embedding.archive import (
    _atomic_json,
    sha256_file,
)

from .config import CLIP_SHA256, PREDICTOR_SHA256
from .initialization import initial_arguments, load_bounds


class Runtime:
    """One reusable pipeline/scorer; owns no experiment population or history."""

    def __init__(self, config):
        if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
            raise RuntimeError(
                "Set NVIDIA_TF32_OVERRIDE=0 before kernel/PyTorch startup"
            )
        config.guard()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if torch.arange(64, device=config.device).square().sum().item() != 85344:
            raise RuntimeError("CUDA tensor smoke test failed")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,mig.mode.current,mig.mode.pending",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if "Enabled" in gpu:
            raise RuntimeError("MIG must be disabled, including its pending mode")
        cache = Path(config.cache_dir).expanduser().resolve()
        clip_path = cache / "clip/ViT-L-14.pt"
        predictor_path = cache / "aesthetics/sac+logos+ava1-l14-linearMSE.pth"
        for path, expected in (
            (clip_path, CLIP_SHA256),
            (predictor_path, PREDICTOR_SHA256),
        ):
            if sha256_file(path) != expected:
                raise ValueError(f"Model checksum differs: {path.name}")
        os.environ["HF_HOME"] = str(cache / "huggingface")
        from diffusers import DiffusionPipeline
        from huggingface_hub import hf_hub_download

        from evolutionary_imaging.evaluators import AestheticsImageEvaluator
        from evolutionary_prompt_embedding.image_creation import (
            SDXLPromptEmbeddingImageCreator,
        )

        index = hf_hub_download(
            config.model_id,
            "model_index.json",
            revision=config.model_revision,
            cache_dir=str(cache / "huggingface/hub"),
            local_files_only=True,
        )
        snapshot = Path(index).parent
        model_files = {
            p.relative_to(snapshot).as_posix(): sha256_file(p)
            for p in sorted(snapshot.rglob("*"))
            if p.is_file()
        }
        self.pipe = DiffusionPipeline.from_pretrained(
            snapshot,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            local_files_only=True,
        ).to(config.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.creator = SDXLPromptEmbeddingImageCreator(
            config.inference_steps,
            1,
            fixed_noise_seeds=[config.diffusion_seed],
            pipeline=self.pipe,
        )
        self.evaluator = AestheticsImageEvaluator(
            device=config.device,
            model_path=str(predictor_path),
            clip_model_path=str(clip_path),
        )
        self.signature = (
            config.cache_dir,
            config.device,
            config.model_id,
            config.model_revision,
            config.inference_steps,
            config.diffusion_seed,
        )
        versions = {"python": platform.python_version()}
        for package in (
            "torch",
            "torchvision",
            "diffusers",
            "transformers",
            "accelerate",
            "Pillow",
            "numpy",
            "clip",
            "pytorch-lightning",
            "safetensors",
            "nbformat",
            "aiohttp",
        ):
            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                versions[package] = None
        self.environment = {
            "versions": versions,
            "gpu": gpu,
            "cuda_runtime": torch.version.cuda,
            "tf32_override": os.environ["NVIDIA_TF32_OVERRIDE"],
            "matmul_tf32": False,
            "cudnn_tf32": False,
        }
        self.models = {
            "model_id": config.model_id,
            "model_revision": config.model_revision,
            "files": model_files,
            "clip_sha256": CLIP_SHA256,
            "predictor_sha256": PREDICTOR_SHA256,
            "inference_dtype": "float16",
            "scorer_dtype": "float32",
        }
        self.parity = None

    def compatible(self, config):
        return self.signature == (
            config.cache_dir,
            config.device,
            config.model_id,
            config.model_revision,
            config.inference_steps,
            config.diffusion_seed,
        )

    def create_solutions(self, config, arguments):
        """Fixed physical shape avoids FP16 renderer changes on partial batches.

        Padding is a copy of an already required argument: no extra offspring,
        variation RNG, fitness query, selection feedback or evaluation ID.
        """
        if config.render_batch_size is None:
            return self.creator.create_solutions(arguments)
        output = []
        size = config.render_batch_size
        for offset in range(0, len(arguments), size):
            group = arguments[offset : offset + size]
            padded = group + [group[-1]] * (size - len(group))
            output.extend(self.creator.create_solutions(padded)[: len(group)])
            self.padding_images = getattr(self, "padding_images", 0) + size - len(group)
        return output

    def check_parity(self, config, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        bounds, _ = load_bounds(config.bounds_file, config.bounds_source)
        state = torch.get_rng_state()
        torch.manual_seed(10000)
        args = initial_arguments(bounds, 16)
        torch.set_rng_state(state)
        images = []
        for offset in range(0, len(args), config.candidate_batch_size):
            images.extend(
                c.result.images[0]
                for c in self.create_solutions(
                    config, args[offset : offset + config.candidate_batch_size]
                )
            )
        for i, image in enumerate(images):
            image.save(directory / f"reference-{i:02d}.png")
        from evolutionary_imaging.image_base import ImageSolutionData

        scores = []
        for offset in range(0, len(images), config.candidate_batch_size):
            scores.extend(
                self.evaluator.evaluate_batch(
                    [
                        ImageSolutionData([image])
                        for image in images[
                            offset : offset + config.candidate_batch_size
                        ]
                    ]
                )
            )
        if config.canonical_scoring:
            scores = [
                self.evaluator.evaluate(ImageSolutionData([image])) for image in images
            ]
        same_size_fixed_noise = fixed_scorer = rng_unchanged = fixed_positions = True
        cross_batch = []
        repeated = []
        if config.canonical_scoring:
            # Diagnostic rerenders never enter the optimizer or consume its RNG.
            import hashlib

            def image_hash(image):
                return hashlib.sha256(image.tobytes()).hexdigest()

            rng_before = torch.get_rng_state()
            cuda_before = torch.cuda.get_rng_state_all()
            first = image_hash(images[0])
            repeated = self.create_solutions(
                config, args[: config.candidate_batch_size]
            )
            same_size_fixed_noise = all(
                image_hash(c.result.images[0]) == image_hash(images[i])
                for i, c in enumerate(repeated)
            )
            cross_batch = []
            for batch_size in (1, 2, 3, 4, 16):
                candidate = self.create_solutions(config, args[:batch_size])[0]
                candidate.result.images[0].save(
                    directory / f"diagnostic-batch-{batch_size}.png"
                )
                value = self.evaluator.evaluate(candidate.result)
                cross_batch.append(
                    {
                        "batch_size": batch_size,
                        "first_image_equal": image_hash(candidate.result.images[0])
                        == first,
                        "first_score": float(value),
                    }
                )
            scorer_repeat = [
                float(self.evaluator.evaluate(ImageSolutionData([images[0]])))
                for _ in range(3)
            ]
            fixed_scorer = len(set(scorer_repeat)) == 1
            position_checks = []
            if config.render_batch_size is not None:
                for position in range(config.render_batch_size):
                    group = list(args[: config.render_batch_size])
                    group[position] = args[0]
                    candidate = self.create_solutions(config, group)[position]
                    candidate.result.images[0].save(
                        directory / f"diagnostic-position-{position}.png"
                    )
                    position_checks.append(
                        image_hash(candidate.result.images[0]) == first
                    )
            fixed_positions = all(position_checks)

            rng_unchanged = torch.equal(rng_before, torch.get_rng_state()) and all(
                torch.equal(a, b)
                for a, b in zip(cuda_before, torch.cuda.get_rng_state_all())
            )
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="",
            PYTHONPATH=str(checkout_root()),
            NVIDIA_TF32_OVERRIDE="0",
        )
        import sys

        subprocess.run(
            [
                sys.executable,
                "-m",
                "evolutionary_extensions.experiments.prompt_embedding.cpu_reference",
                "--images",
                str(directory),
                "--cache",
                str(Path(config.cache_dir).expanduser()),
                "--output",
                str(directory / "cpu.json"),
            ],
            env=env,
            check=True,
            timeout=min(
                600, max(1, (config.deadline_unix or time.time() + 600) - time.time())
            ),
        )
        reference = json.loads((directory / "cpu.json").read_text())
        errors = np.abs(np.array(scores) - np.array(reference))
        self.parity = {
            "mean_absolute_error": float(errors.mean()),
            "maximum_absolute_error": float(errors.max()),
            "image_count": len(images),
            "candidate_batch_size": config.candidate_batch_size,
            "passed": bool(
                errors.mean() <= 0.001
                and errors.max() <= 0.005
                and same_size_fixed_noise
                and fixed_scorer
                and rng_unchanged
                and fixed_positions
                and (
                    not config.canonical_scoring
                    or all(row["first_image_equal"] for row in cross_batch)
                )
            ),
            "canonical_scoring": config.canonical_scoring,
            "render_batch_size": config.render_batch_size,
            "fixed_positions": fixed_positions,
            "same_size_fixed_noise": same_size_fixed_noise,
            "fixed_scorer_repeated_identical": fixed_scorer,
            "diagnostic_rng_unchanged": rng_unchanged,
            "cross_batch": cross_batch,
            "cross_batch_images_identical": all(
                row["first_image_equal"] for row in cross_batch
            )
            if cross_batch
            else None,
            "diagnostic_image_count": len(images)
            + len(repeated)
            + (
                sum((1, 2, 3, 4, 16)) + (config.render_batch_size or 0) ** 2
                if config.canonical_scoring
                else 0
            ),
            "cpu_scores": reference,
            "cuda_scores": scores,
        }
        _atomic_json(directory / "parity.json", self.parity)
        if not self.parity["passed"]:
            raise RuntimeError("CPU/CUDA scorer parity failed")
        return self.parity

    def check_anchor(self, config, directory):
        """Verify a frozen local input without adding search candidates or feedback."""
        from PIL import Image
        from safetensors.torch import load_file

        from evolutionary_prompt_embedding.argument_types import PooledPromptEmbedData

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        source = Path(config.initial_embeddings_file)
        values = load_file(str(source))
        if sha256_file(source) != config.initial_embeddings_sha256:
            raise ValueError("Anchor input changed")
        argument = PooledPromptEmbedData(
            values["prompt_embeds"][0], values["pooled_prompt_embeds"][0]
        )
        candidates = self.create_solutions(
            config, [argument] * config.candidate_batch_size
        )
        with Image.open(source.with_suffix(".png")) as reference:
            expected = np.asarray(reference.convert("RGB"))
        identical = all(
            np.array_equal(np.asarray(c.result.images[0].convert("RGB")), expected)
            for c in candidates
        )
        scores = [float(self.evaluator.evaluate(c.result)) for c in candidates]
        report = {
            "passed": identical and len(set(scores)) == 1,
            "images_identical_to_retained_png": identical,
            "position_scores": scores,
            "diagnostic_image_count": len(candidates),
            "input_sha256": config.initial_embeddings_sha256,
        }
        _atomic_json(directory / "anchor_verification.json", report)
        if not report["passed"]:
            raise RuntimeError("Frozen anchor rerender verification failed")
        return report


def score_results(evaluator, results, *, canonical=False):
    """Optional fixed scorer shape for reproducible strict parent comparisons."""
    return (
        [evaluator.evaluate(result) for result in results]
        if canonical
        else evaluator.evaluate_batch(results)
    )
