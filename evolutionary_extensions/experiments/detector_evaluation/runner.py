"""Dedicated detector benchmark stages; lifecycle remains externally supervised."""

import argparse
import gc
import importlib.metadata
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from evolutionary_extensions.persistence.packaging import atomic_json, hashes

from .config import DETECTORS, load_config, scientific_identity
from .journal import ScoreJournal


def status(root, stage, **details):
    record = {"stage": stage, "updated_unix": time.time(), **details}
    atomic_json(Path(root) / "progress.json", record)
    print(json.dumps(record), flush=True)


def guard_disk(config):
    for path in (config["output"], config["assets"]):
        existing = Path(path)
        while not existing.exists():
            existing = existing.parent
        if shutil.disk_usage(existing).free < 20 * 1024**3:
            raise RuntimeError(
                "Disk reserve below 20 GiB; preserve partial evidence and stop"
            )


def cuda_preflight(root):
    import torch

    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise RuntimeError("Set NVIDIA_TF32_OVERRIDE=0 before kernel startup")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    actual = float(
        torch.arange(64, device="cuda", dtype=torch.float32).square().sum().item()
    )
    if actual != 85344:
        raise RuntimeError("CUDA computation failed")
    mig = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=mig.mode.current,mig.mode.pending",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    if any(s.strip() != "Disabled" for s in mig.split(",")):
        raise RuntimeError("Require disabled current and pending MIG")
    free, total = torch.cuda.mem_get_info()
    name = torch.cuda.get_device_name()
    if "A100" not in name or total < 64 * 1024**3 or free < 20 * 1024**3:
        raise RuntimeError("Insufficient A100 GPU memory")
    memory = dict(
        line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()
    )
    available_ram = int(memory["MemAvailable"].split()[0]) * 1024
    if available_ram < 32 * 1024**3:
        raise RuntimeError("Available RAM below 32 GiB")
    atomic_json(
        Path(root) / "cuda.json",
        {
            "device": name,
            "mig": mig,
            "available_ram_bytes": available_ram,
            "free_bytes": free,
            "total_bytes": total,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    )


def assets_for(config, detector=None):
    root = Path(config["assets"])
    assets = json.loads((root / "assets.json").read_text())
    if (
        not assets.get("complete")
        or len(assets["detectors"]["aeroblade"]["autoencoders"]) != 3
    ):
        raise ValueError(
            "Require all five detectors and all three AEROBLADE autoencoders before GPU execution"
        )
    if detector in (None, "generate"):
        for filename, expected in assets["generation"]["files"].items():
            if hashes(filename)["sha256"] != expected:
                raise ValueError("Changed frozen generation asset")
    for name in (
        DETECTORS
        if detector is None
        else (() if detector == "generate" else (detector,))
    ):
        item = assets["detectors"][name]
        source = Path(item["source"])
        commit = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        if commit != item["commit"]:
            raise ValueError(f"Source revision changed: {name}")
        if subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
            text=True,
        ):
            raise ValueError(f"Modified official source: {name}")
        for filename, expected in item["files"].items():
            if hashes(filename)["sha256"] != expected:
                raise ValueError(f"Changed model/config asset: {name}")
    return assets


def freeze_inputs(config):
    root = Path(config["output"])
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        **scientific_identity(config),
        "raise_sha256": hashes(config["raise_csv"])["sha256"],
        "prompts_sha256": hashes(config["prompts_csv"])["sha256"],
    }
    path = root / "recipe.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("Frozen evaluation recipe differs; use a new output root")
    atomic_json(path, identity)
    for key in ("raise_csv", "prompts_csv"):
        destination = root / "inputs" / f"{key}.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(config[key], destination)
        if hashes(destination)["sha256"] != hashes(config[key])["sha256"]:
            raise ValueError("Archived input CSV differs")
    return identity


def smoke_pool(rows):
    pool = []
    for group in ("calibration", "real", "random", "prompted"):
        subset = sorted(
            [r for r in rows if r["group"] == group and r["codec"] == "png"],
            key=lambda r: r["image_id"],
        )
        pool.extend(subset[:4])
    if len(pool) != 16:
        raise ValueError(
            "Require four images from each group for the 16-image smoke pool"
        )
    return pool


def profile_classifier(
    adapter, tensors, root, detector, batches=None, atol=1e-4, rtol=0
):
    import torch

    reset = getattr(adapter, "reset", lambda: None)
    adapter.infer(tensors[:1])
    reset()
    baseline = np.concatenate(
        [adapter.infer(tensors[i : i + 1]) for i in range(len(tensors))]
    )
    if not np.isfinite(baseline).all():
        raise RuntimeError("Non-finite scalar smoke scores")
    reset()
    repeat = np.concatenate(
        [adapter.infer(tensors[i : i + 1]) for i in range(len(tensors))]
    )
    if not np.allclose(baseline, repeat, atol=atol, rtol=rtol):
        raise RuntimeError("Scalar scoring is not repeatable")
    records = []
    sizes = (
        batches
        if batches is not None
        else (1,)
        if detector == "bfree"
        else (1, 4, 8, 16)
    )
    for batch in sizes:
        durations, peak, valid, error = [], 0, True, None
        try:
            for _ in range(3):
                reset()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.perf_counter()
                values = np.concatenate(
                    [
                        adapter.infer(tensors[i : i + batch])
                        for i in range(0, len(tensors), batch)
                    ]
                )
                torch.cuda.synchronize()
                durations.append(time.perf_counter() - start)
                peak = max(
                    peak,
                    torch.cuda.max_memory_allocated(),
                    torch.cuda.max_memory_reserved(),
                )
                valid &= bool(np.allclose(values, baseline, atol=atol, rtol=rtol))
        except torch.OutOfMemoryError:
            valid, error = False, "CUDA out of memory"
            torch.cuda.empty_cache()
        valid &= peak < 64 * 1024**3
        records.append(
            {
                "batch": batch,
                "valid": valid,
                "error": error,
                "peak_bytes": peak,
                "seconds": float(np.median(durations)) if durations else None,
                "images_per_second": len(tensors) / float(np.median(durations))
                if durations
                else None,
            }
        )
    accepted = [r for r in records if r["valid"]]
    if not accepted or not records[0]["valid"]:
        raise RuntimeError("Scalar path failed validation")
    best = max(accepted, key=lambda r: r["images_per_second"])
    atomic_json(
        root / "profiles" / f"{detector}.json",
        {
            "candidates": records,
            "chosen_batch": best["batch"],
            "score_atol": atol,
            "score_rtol": rtol,
            "smoke_scores": baseline.tolist(),
        },
    )
    return best["batch"], baseline


def score(config, detector):
    import torch

    from .adapters import Aeroblade, Classifier, score_polarity

    root = Path(config["output"])
    guard_disk(config)
    cuda_preflight(root)
    assets = assets_for(config, detector)
    os.environ["TORCH_HOME"] = assets["torch_home"]
    identity = {
        "recipe": freeze_inputs(config),
        "detector": assets["detectors"][detector],
        "scorer_files": {
            name: hashes(Path(__file__).parent / name)["sha256"]
            for name in ("adapters.py", "data.py", "config.py")
        },
        "precision": "native FP16 AE, FP32 LPIPS, released FP16 error reduction"
        if detector == "aeroblade"
        else "FP32",
        "tf32": False,
        "score_comparison": "strict >",
        "packages": {
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        },
    }
    atomic_json(root / "provenance" / f"{detector}.json", identity)
    journal = ScoreJournal(root, detector, identity)
    rows = json.loads((root / "images.json").read_text())
    pending = [r for r in rows if journal.existing(r) is None]
    if not pending:
        status(root, detector, completed=len(rows), recovered=True)
        return
    start = time.perf_counter()
    if detector == "aeroblade":
        adapter = Aeroblade(config, assets)
        adapter.run(
            rows,
            journal,
            lambda **d: status(root, detector, **d),
            lambda: guard_disk(config),
        )
    else:
        adapter = Classifier(detector, config, assets)
        torch.cuda.synchronize()
        cold = time.perf_counter() - start
        pool = smoke_pool(rows)
        prep_start = time.perf_counter()
        tensors, _ = adapter.prepare(pool)
        preprocessing = time.perf_counter() - prep_start
        batch, baseline = profile_classifier(adapter, tensors, root, detector)
        native = np.array([adapter.official_scalar(row) for row in pool])
        if not np.allclose(native, baseline, atol=1e-4, rtol=0):
            raise RuntimeError("Official scalar inference parity failed")
        del adapter
        gc.collect()
        torch.cuda.empty_cache()
        # Explicit offline model reload is a gate, not a metadata assertion.
        adapter = Classifier(detector, config, assets)
        reload_scores = np.concatenate(
            [adapter.infer(tensors[i : i + 1]) for i in range(16)]
        )
        if not np.allclose(reload_scores, baseline, atol=1e-4, rtol=0):
            raise RuntimeError("Offline reload parity failed")
        atomic_json(
            root / "parity" / f"{detector}.json",
            {
                "official_scores": native.tolist(),
                "offline_scores": reload_scores.tolist(),
                "passed": True,
            },
        )
        timing = {
            "cold_load_seconds": cold,
            "smoke_preprocessing_seconds": preprocessing,
            "preprocessing_seconds": 0.0,
            "inference_seconds": 0.0,
            "persistence_seconds": 0.0,
        }
        for offset in range(0, len(pending), batch):
            guard_disk(config)
            group = pending[offset : offset + batch]
            tick = time.perf_counter()
            inputs, details = adapter.prepare(group)
            timing["preprocessing_seconds"] += time.perf_counter() - tick
            torch.cuda.synchronize()
            tick = time.perf_counter()
            raw = adapter.infer(inputs)
            torch.cuda.synchronize()
            timing["inference_seconds"] += time.perf_counter() - tick
            tick = time.perf_counter()
            for row, value, detail in zip(group, raw, details):
                if detector == "ssp":
                    detail["human_probability"] = float(value)
                detail["batch_size"] = batch
                journal.write(row, value, score_polarity(detector, value), **detail)
            timing["persistence_seconds"] += time.perf_counter() - tick
            completed = len(rows) - len(pending) + min(offset + batch, len(pending))
            status(
                root,
                detector,
                completed=completed,
                total=len(rows),
                batch=batch,
                elapsed_seconds=time.perf_counter() - start,
                estimated_remaining_stage_seconds=(
                    len(pending) - min(offset + batch, len(pending))
                )
                / (
                    min(offset + batch, len(pending))
                    / max(time.perf_counter() - start - cold, 1e-6)
                ),
            )
            atomic_json(root / "timing" / f"{detector}.json", timing)
        atomic_json(root / "timing" / f"{detector}.json", timing)
        del adapter
    gc.collect()
    torch.cuda.empty_cache()
    status(
        root,
        detector,
        completed=len(rows),
        total=len(rows),
        elapsed_seconds=time.perf_counter() - start,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "generate", *DETECTORS, "analyze", "report")
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    freeze_inputs(config)
    try:
        if args.stage == "prepare":
            from .data import prepare_real

            prepare_real(config)
        elif args.stage == "generate":
            from .generation import generate

            generate(config)
        elif args.stage == "analyze":
            from .analysis import analyze

            analyze(config)
        elif args.stage == "report":
            from .report import export_report

            export_report(config)
        else:
            score(config, args.stage)
    except BaseException as error:
        status(
            config["output"],
            args.stage,
            failed=True,
            error_type=type(error).__name__,
            error=str(error),
        )
        raise


if __name__ == "__main__":
    main()
