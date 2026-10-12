"""Shared generator pixels, repeated and reloaded offline before optimizer work."""

import gc
import hashlib
import importlib
import json
import os
import time
from pathlib import Path

from evolutionary_extensions.experiments.detector_evaluation.config import (
    DETECTORS,
    MODEL_REVISION,
    digest,
    load_config,
)
from evolutionary_extensions.persistence.packaging import atomic_json, hashes

INDICES = [*range(8), *range(100, 108)]
SEEDS = [0, 1, 2, 3]
PASSES = ["initial", "repeat", "offline_reload"]


def load_generator(assets):
    import torch
    from diffusers import DiffusionPipeline
    from huggingface_hub import snapshot_download

    if assets["generation"]["revision"] != MODEL_REVISION:
        raise ValueError("Shared generator revision differs")
    snapshot = snapshot_download(
        "stabilityai/sdxl-turbo",
        revision=MODEL_REVISION,
        cache_dir=assets["hf_cache"],
        local_files_only=True,
    )
    pipe = DiffusionPipeline.from_pretrained(
        snapshot,
        variant="fp16",
        torch_dtype=torch.float16,
        use_safetensors=True,
        local_files_only=True,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def runtime_identity():
    import diffusers

    modules = {}
    for name in (
        "torch",
        "diffusers",
        "transformers",
        "huggingface_hub",
        "safetensors",
        "numpy",
        "PIL",
        diffusers.StableDiffusionXLPipeline.__module__,
        diffusers.EulerDiscreteScheduler.__module__,
        diffusers.AutoencoderKL.__module__,
    ):
        module = importlib.import_module(name)
        modules[name] = {
            "version": getattr(module, "__version__", None),
            "path": str(Path(module.__file__).resolve()),
            "sha256": hashes(module.__file__)["sha256"],
        }
    source = {
        name: hashes(Path(__file__).parent / name)["sha256"]
        for name in ("generator_validation.py", "rendering.py", "heldout_rendering.py")
    }
    for name in (
        "evolutionary_prompt_embedding.image_creation",
        "evolutionary_imaging.image_base",
    ):
        source[name] = hashes(importlib.import_module(name).__file__)["sha256"]
    return {
        "modules": modules,
        "source_files": source,
        "environment": {
            k: os.environ.get(k)
            for k in ("NVIDIA_TF32_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG")
        },
    }


def parity_recipe(config):
    benchmark = load_config(config["benchmark_config"])
    root = Path(benchmark["output"])
    embedding_sha = hashes(root / "embeddings.safetensors")["sha256"]
    if (
        json.loads((root / "generation.json").read_text())["embedding_sha256"]
        != embedding_sha
    ):
        raise ValueError("Benchmark generator embeddings changed")
    assets = json.loads((Path(benchmark["assets"]) / "assets.json").read_text())
    return {
        "model_revision": MODEL_REVISION,
        "generation_assets": assets["generation"],
        "embedding_sha256": embedding_sha,
        "embedding_indices": INDICES,
        "noise_seeds": SEEDS,
        "batch": 4,
        "steps": 1,
        "guidance": 0,
        "precision": "FP16",
        "size": [512, 512],
    }


def pixel_hash(image):
    if image.mode != "RGB" or image.size != (512, 512):
        raise ValueError("Require unchanged 512x512 RGB generator pixels")
    return hashlib.sha256(image.tobytes()).hexdigest()


def render_pass(root, label, render, *, after_batch=lambda count: None):
    """Callback generates exactly four ordered embeddings with independently reset noise."""
    root = Path(root)
    rows = []
    for seed in SEEDS:
        for offset in range(0, len(INDICES), 4):
            indices = INDICES[offset : offset + 4]
            images = render(indices, seed)
            if len(images) != 4:
                raise ValueError("Generator diagnostic batch order/count differs")
            after_batch(4)
            for index, image in zip(indices, images):
                rgb = pixel_hash(image)
                relative = Path(label) / f"seed{seed}-embedding{index:03d}.png"
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path)
                rows.append(
                    {
                        "seed": seed,
                        "embedding_index": index,
                        "path": relative.as_posix(),
                        "sha256": hashes(path)["sha256"],
                        "rgb_sha256": rgb,
                    }
                )
    return rows


def check_pixels(root, rows):
    from PIL import Image

    expected = [(seed, index) for seed in SEEDS for index in INDICES]
    if [(r["seed"], r["embedding_index"]) for r in rows] != expected:
        raise ValueError("Generator diagnostic coverage/order differs")
    values = []
    root = Path(root).resolve()
    for row in rows:
        path = root / row["path"]
        if (
            not path.resolve().is_relative_to(root)
            or hashes(path)["sha256"] != row["sha256"]
        ):
            raise ValueError("Generator diagnostic image bytes changed")
        with Image.open(path) as image:
            value = pixel_hash(image)
        if value != row["rgb_sha256"]:
            raise ValueError("Generator diagnostic decoded pixels changed")
        values.append(value)
    return values


def verify_receipt(root, stage, recipe, runtime):
    root = Path(root)
    if stage not in ("generate", *DETECTORS):
        raise ValueError("Unknown generator diagnostic stage")
    reference_path = root / "generate/parity.json"
    reference = json.loads(reference_path.read_text())
    path = root / stage / "parity.json"
    receipt = json.loads(path.read_text())
    if any(
        r.get("passed") is not True or r.get("recipe") != recipe
        for r in (reference, receipt)
    ):
        raise ValueError("Missing or changed shared generator parity")
    if (
        receipt["stage"] != stage
        or reference["stage"] != "generate"
        or receipt["runtime"] != runtime
    ):
        raise ValueError("Generator runtime changed after validation")
    if (
        stage != "generate"
        and receipt["reference_sha256"] != hashes(reference_path)["sha256"]
    ):
        raise ValueError("Generator reference receipt changed")
    for item, directory in ((reference, root / "generate"), (receipt, root / stage)):
        receipt_sha = hashes(directory / "parity.json")["sha256"]
        attempts = [
            json.loads(p.read_text()) for p in (directory / "attempts").glob("*.json")
        ]
        if not any(
            a.get("state") == "complete"
            and a.get("receipt_sha256") == receipt_sha
            and a.get("recipe_sha256") == digest(recipe)
            and a.get("runtime") == item["runtime"]
            and a.get("known_physical_images") == 192
            and a.get("candidate_budget_used") == 0
            and a.get("maximum_physical_images") == 192
            for a in attempts
        ):
            raise ValueError(
                "Generator receipt lacks its completed physical query ledger"
            )
    ref_values = check_pixels(root / "generate", reference["passes"]["initial"])
    for value, folder in ((reference, root / "generate"), (receipt, root / stage)):
        if set(value["passes"]) != set(PASSES):
            raise ValueError("Require repeat and offline generator reload")
        if value["known_physical_images"] != 192 or value["candidate_budget_used"] != 0:
            raise ValueError("Generator diagnostic query accounting differs")
        if any(
            not isinstance(value.get(k), int) or not 0 <= value[k] < 64 * 1024**3
            for k in ("peak_allocated_bytes", "peak_reserved_bytes")
        ):
            raise ValueError("Generator diagnostic memory proof differs")
        for label in PASSES:
            if check_pixels(folder, value["passes"][label]) != ref_values:
                raise ValueError(
                    "Generator decoded pixels differ across stacks or repeats"
                )
    return receipt


def verify_generator_parity(config, stage):
    return verify_receipt(
        Path(config["output"]) / "generator_validation",
        stage,
        parity_recipe(config),
        runtime_identity(),
    )


def audit_generator_validation(search_root):
    """Revalidate all pixel proofs and account for failed diagnostic attempts."""
    root = Path(search_root) / "generator_validation"
    recipe = json.loads((root / "generate/parity.json").read_text())["recipe"]
    records = []
    for stage in ("generate", *DETECTORS):
        folder = root / stage
        proof = json.loads((folder / "parity.json").read_text())
        verify_receipt(root, stage, recipe, proof["runtime"])
        paths = sorted((folder / "attempts").glob("*.json"))
        if [p.name for p in paths] != [f"{i:04d}.json" for i in range(len(paths))]:
            raise ValueError("Noncontiguous generator diagnostic ledger")
        known, upper, completed = 0, 0, 0
        for path in paths:
            item = json.loads(path.read_text())
            count, state = item["known_physical_images"], item["state"]
            if (
                item.get("stage") != stage
                or state not in ("started", "failed", "complete")
                or not isinstance(count, int)
                or not 0 <= count <= 192
                or count % 4
                or item.get("maximum_physical_images") != 192
                or item.get("candidate_budget_used") != 0
            ):
                raise ValueError("Generator diagnostic query intent differs")
            if state == "complete":
                if (
                    count != 192
                    or item.get("receipt_sha256")
                    != hashes(folder / "parity.json")["sha256"]
                ):
                    raise ValueError("Completed generator query proof differs")
                completed += count
            known += count
            upper += count if state == "complete" else 192
        records.append(
            {
                "stage": stage,
                "known_physical_images": known,
                "maximum_attempted_physical_images": upper,
                "completed_validation_images": completed,
                "candidate_budget_used": 0,
                "attempts": len(paths),
                "proof_sha256": hashes(folder / "parity.json")["sha256"],
            }
        )
    return {
        "stages": records,
        "known_physical_images": sum(r["known_physical_images"] for r in records),
        "maximum_attempted_physical_images": sum(
            r["maximum_attempted_physical_images"] for r in records
        ),
        "candidate_budget_used": 0,
        "note": "Generator diagnostics only; failed/started intents are conservative upper bounds. Separate from optimizer candidates and detector score queries.",
    }


def run_validation(config_path, stage):
    import torch
    from safetensors.torch import load_file

    from evolutionary_extensions.experiments.detector_evaluation.runner import (
        assets_for,
        cuda_preflight,
        guard_disk,
    )
    from evolutionary_prompt_embedding.argument_types import PooledPromptEmbedData
    from evolutionary_prompt_embedding.image_creation import (
        SDXLPromptEmbeddingImageCreator,
    )

    from .runner import verify_prerequisite

    config = json.loads(Path(config_path).read_text())
    benchmark, benchmark_rows, _, _ = verify_prerequisite(config)
    if stage not in ("generate", *DETECTORS):
        raise ValueError("Unknown generator validation kernel")
    root = Path(config["output"]) / "generator_validation"
    folder = root / stage
    folder.mkdir(parents=True, exist_ok=True)
    recipe, runtime = parity_recipe(config), runtime_identity()
    if (folder / "parity.json").exists():
        verify_receipt(root, stage, recipe, runtime)
        return
    reference_sha = None
    if stage != "generate":
        reference_path = root / "generate/parity.json"
        reference = json.loads(reference_path.read_text())
        # Verify the immutable reference before any diagnostic rendering.
        verify_receipt(root, "generate", recipe, reference["runtime"])
        reference_sha = hashes(reference_path)["sha256"]
    guard = lambda: guard_disk({**benchmark, "output": str(folder)})
    guard()
    cuda_preflight(folder)
    assets = assets_for(benchmark, "generate")
    tensors = load_file(str(Path(benchmark["output"]) / "embeddings.safetensors"))
    if (
        set(tensors) != {"prompt_embeds", "pooled_prompt_embeds"}
        or tensors["prompt_embeds"].shape != (200, 77, 2048)
        or tensors["pooled_prompt_embeds"].shape != (200, 1280)
        or any(
            t.dtype != torch.float16 or not torch.isfinite(t).all()
            for t in tensors.values()
        )
    ):
        raise ValueError("Require the frozen finite FP16 benchmark embeddings")
    attempts = folder / "attempts"
    attempts.mkdir(exist_ok=True)
    existing = sorted(attempts.glob("*.json"))
    if [p.name for p in existing] != [f"{i:04d}.json" for i in range(len(existing))]:
        raise ValueError("Generator diagnostic attempt ledger is not contiguous")
    attempt = attempts / f"{len(existing):04d}.json"
    intent = {
        "state": "started",
        "stage": stage,
        "recipe_sha256": digest(recipe),
        "runtime": runtime,
        "known_physical_images": 0,
        "maximum_physical_images": 192,
        "candidate_budget_used": 0,
        "started_unix": time.time(),
    }
    atomic_json(attempt, intent)
    passes, pipe = {}, None

    def after_batch(count):
        intent["known_physical_images"] += count
        atomic_json(attempt, intent)
        if (
            max(torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
            >= 64 * 1024**3
        ):
            raise RuntimeError("Generator validation exceeds 64 GiB peak memory")

    def render(indices, seed):
        guard()
        arguments = [
            PooledPromptEmbedData(
                tensors["prompt_embeds"][i : i + 1],
                tensors["pooled_prompt_embeds"][i : i + 1],
            )
            for i in indices
        ]
        creator = SDXLPromptEmbeddingImageCreator(
            1, 1, fixed_noise_seeds=[seed], pipeline=pipe
        )
        outputs = creator.create_solutions(arguments)
        torch.cuda.synchronize()
        return [o.result.images[0] for o in outputs]

    try:
        torch.cuda.reset_peak_memory_stats()
        pipe = load_generator(assets)
        after_batch(0)
        for label in PASSES:
            if label == "offline_reload":
                pipe = None
                gc.collect()
                torch.cuda.empty_cache()
                pipe = load_generator(assets)
                after_batch(0)
            passes[label] = render_pass(folder, label, render, after_batch=after_batch)
        values = check_pixels(folder, passes["initial"])
        if any(check_pixels(folder, passes[label]) != values for label in PASSES):
            raise RuntimeError("Generator repeat/offline decoded pixels differ")
        if stage == "generate":
            from PIL import Image

            by_index = {
                r["embedding_index"]: r
                for r in benchmark_rows
                if r["codec"] == "png" and r["group"] in ("random", "prompted")
            }
            for row in passes["initial"]:
                if row["seed"] == 0:
                    baseline = by_index[row["embedding_index"]]
                    path = Path(benchmark["output"]) / baseline["path"]
                    if hashes(path)["sha256"] != baseline["sha256"]:
                        raise ValueError("Benchmark reference image changed")
                    with Image.open(path) as image:
                        if pixel_hash(image) != row["rgb_sha256"]:
                            raise RuntimeError(
                                "Common generator differs from benchmark pixels"
                            )
        elif values != check_pixels(root / "generate", reference["passes"]["initial"]):
            raise RuntimeError(
                "Detector environment generator differs from common pixels"
            )
        receipt = {
            "passed": True,
            "stage": stage,
            "recipe": recipe,
            "runtime": runtime,
            "reference_sha256": reference_sha,
            "passes": passes,
            "known_physical_images": intent["known_physical_images"],
            "candidate_budget_used": 0,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "finished_unix": time.time(),
        }
        atomic_json(folder / "parity.json", receipt)
        intent.update(
            state="complete",
            finished_unix=time.time(),
            receipt_sha256=hashes(folder / "parity.json")["sha256"],
        )
        atomic_json(attempt, intent)
        verify_receipt(root, stage, recipe, runtime)
    except BaseException as error:
        intent.update(
            state="failed", error_type=type(error).__name__, finished_unix=time.time()
        )
        atomic_json(attempt, intent)
        raise
    finally:
        if pipe is not None:
            del pipe
        gc.collect()
        torch.cuda.empty_cache()
