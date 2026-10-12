"""Continuous target objectives with preserved official probabilities/errors."""

import importlib
import importlib.metadata
import json
from pathlib import Path

from evolutionary_extensions.experiments.detector_evaluation.adapters import Classifier
from evolutionary_extensions.experiments.detector_evaluation.data import (
    patch_coordinates,
    simplest_patch,
)
from evolutionary_extensions.persistence.packaging import atomic_json, hashes


def package_versions():
    """Resolve overlay versions before shared read-only base distributions."""
    return {
        d.metadata["Name"]: importlib.metadata.version(d.metadata["Name"])
        for d in importlib.metadata.distributions()
    }


def scoring_procedure(detector, asset, packages, batch):
    """Bind reusable numbers to the complete effective inference procedure.

    Output directories and optimizer settings do not affect inference. SSP's
    actual coordinates belong to each image key rather than this descriptor.
    Distinct batch sizes remain distinct even after a tolerance parity check.
    """
    from evolutionary_extensions.experiments import detector_evaluation

    evaluation = Path(detector_evaluation.__file__).parent
    names = ["torch", "torchvision", "numpy", "PIL"]
    if detector == "aeroblade":
        names += ["diffusers", "lpips"]
    elif detector == "safe":
        names += ["kornia", "cv2"]
    elif detector == "bfree":
        names += ["timm"]
    runtime = {}
    for name in names:
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        runtime[name] = {
            "version": getattr(module, "__version__", None),
            "path": str(path),
            "sha256": hashes(path)["sha256"],
        }
    return {
        "detector": detector,
        "asset": asset,
        "packages": packages,
        "actual_runtime_imports": runtime,
        "search_source_files": {
            p.name: hashes(p)["sha256"] for p in Path(__file__).parent.glob("*.py")
        },
        "benchmark_source_files": {
            name: hashes(evaluation / name)["sha256"]
            for name in ("adapters.py", "data.py", "config.py")
        },
        "batch": batch,
        "tf32": False,
        "precision": "native FP16 AE, FP32 LPIPS, FP16 error reduction"
        if detector == "aeroblade"
        else "FP32",
        "reconstruction_seed": 1 if detector == "aeroblade" else None,
        "reconstruction_seed_scope": "independently reset per image"
        if detector == "aeroblade"
        else None,
        "decoder_loader": "KandinskyImg2ImgPipeline for pinned decoder-only Kandinsky 2.1; AutoPipelineForImage2Image for SD"
        if detector == "aeroblade"
        else None,
        "native_reference": "official reconstruction/LPIPS under no_grad, reusing the same offline-loaded pipeline"
        if detector == "aeroblade"
        else None,
        "ssp_selection": "simplest of 64 deterministic 32x32 patches"
        if detector == "ssp"
        else None,
    }


def classifier_scores(detector, logits):
    """Margins retain ranking after released probabilities saturate in FP32."""
    import torch

    if logits.ndim != 2 or not torch.isfinite(logits).all():
        raise ValueError("Require finite classifier logits with a batch dimension")
    if detector == "safe" and logits.shape[1] == 2:
        objective = logits[:, 1] - logits[:, 0]
        raw = logits.softmax(dim=1)[:, 1]
        synthetic = raw
    elif detector in ("universal_fake_detect", "ssp") and logits.shape[1] == 1:
        raw = logits[:, 0].sigmoid()
        objective = -logits[:, 0] if detector == "ssp" else logits[:, 0]
        synthetic = 1 - raw if detector == "ssp" else raw
    elif detector == "bfree" and logits.shape[1] in (1, 2):
        raw = logits[:, 0] if logits.shape[1] == 1 else logits[:, 1] - logits[:, 0]
        objective, synthetic = raw, raw
    else:
        raise ValueError("Detector output shape differs from the released model")
    return [
        {
            "objective": float(o),
            "raw": float(r),
            "synthetic_score": 1 - float(r) if detector == "ssp" else float(s),
            "logits": values.tolist(),
        }
        for o, r, s, values in zip(
            objective.cpu().double(),
            raw.cpu().double(),
            synthetic.cpu().double(),
            logits.cpu().double(),
        )
    ]


class SearchClassifier(Classifier):
    """Released classifier with one ID-independent SSP patch pool for fitness."""

    patch_identity = "common-search-pool"

    def prepare(self, rows, image_root=None):
        import torch
        from PIL import Image

        image_root = Path(image_root or self.root).resolve()
        tensors, details = [], []
        for row in rows:
            path = image_root / row["path"]
            if (
                not path.resolve().is_relative_to(image_root)
                or hashes(path)["sha256"] != row["sha256"]
            ):
                raise ValueError("Target input differs from its frozen image hash")
            with Image.open(path) as file:
                image = file.convert("RGB")
            detail = {}
            if self.detector == "ssp":
                coordinate_identity = self.patch_identity or row.get(
                    "subject", row["image_id"]
                )
                # JSON recovery represents coordinate pairs as lists. Keep the
                # effective patch pool identical before comparison/persistence.
                coordinates = [
                    list(pair)
                    for pair in patch_coordinates(
                        coordinate_identity, seed=self.config["patch_seed"]
                    )
                ]
                expected = row.get("scorer_context", {}).get("patch_coordinates")
                if (
                    expected is not None
                    and [list(pair) for pair in expected] != coordinates
                ):
                    raise ValueError("Frozen held-out SSP coordinates differ")
                image, selected = simplest_patch(image, coordinates)
                name = row["image_id"]
                if (
                    Path(name).name != name
                    or name in (".", "..")
                    or row["codec"] not in {"png", "jpeg95", "jpeg75"}
                ):
                    raise ValueError("Unsafe patch image identity")
                relative = Path("patches") / (
                    f"{name}-{row['codec']}-patch{self.config['patch_seed']}.png"
                )
                patch = self.root / relative
                patch.parent.mkdir(parents=True, exist_ok=True)
                image.save(patch)
                detail = {
                    "coordinate_identity": coordinate_identity,
                    "coordinates": coordinates,
                    "selected_index": selected,
                    "patch_path": relative.as_posix(),
                    "patch_sha256": hashes(patch)["sha256"],
                }
                atomic_json(patch.with_suffix(".json"), detail)
            tensors.append(self.transform(image))
            details.append(detail)
        return torch.stack(tensors), details

    def score(self, rows, *, image_root=None, batch=1):
        import torch

        if self.detector == "bfree" and batch != 1:
            raise ValueError("B-Free retains released serial inference")
        records = []
        for offset in range(0, len(rows), batch):
            tensors, details = self.prepare(rows[offset : offset + batch], image_root)
            with torch.inference_mode():
                logits = self.model(tensors.cuda())
            values = classifier_scores(self.detector, logits)
            records.extend(
                {**value, **detail} for value, detail in zip(values, details)
            )
        return records

    def validate_native(self, rows, benchmark_root):
        """Native scalar, repeated output and offline reload before search calls."""
        import numpy as np

        config = {**self.config, "output": str(benchmark_root)}
        # Sources share global module names; this runs in the same target-only
        # kernel, never alongside a different vendor classifier.
        reference = Classifier(
            self.detector, config, {"detectors": {self.detector: self.asset}}
        )
        adapted = self.score(rows, image_root=benchmark_root, batch=1)
        native = [
            reference.official_scalar(
                {
                    **row,
                    "image_id": self.patch_identity
                    or row.get("subject", row["image_id"]),
                }
                if self.detector == "ssp"
                else row
            )
            for row in rows
        ]
        if not np.allclose([r["raw"] for r in adapted], native, atol=1e-4, rtol=0):
            raise RuntimeError("Search classifier differs from native scalar inference")
        repeated = self.score(rows, image_root=benchmark_root, batch=1)
        if not np.allclose(
            [[r[k] for k in ("raw", "objective")] for r in repeated],
            [[r[k] for k in ("raw", "objective")] for r in adapted],
            atol=1e-4,
            rtol=0,
        ):
            raise RuntimeError("Search scalar outputs are not repeatable")
        tensor, _ = self.prepare(rows, benchmark_root)
        raw_reload = np.concatenate(
            [reference.infer(tensor[i : i + 1]) for i in range(len(rows))]
        )
        if not np.allclose([r["raw"] for r in adapted], raw_reload, atol=1e-4, rtol=0):
            raise RuntimeError("Offline classifier reload differs")
        atomic_json(
            self.root / "parity.json",
            {
                "passed": True,
                "native": native,
                "adapted": adapted,
                "repeat": repeated,
                "offline_reload_raw": raw_reload.tolist(),
            },
        )
        del reference

    def profile(self, rows, benchmark_root):
        import time

        import numpy as np
        import torch

        path = self.root / "profiles" / "classifier.json"
        if path.exists() and json.loads(path.read_text()).get("state") == "complete":
            return json.loads(path.read_text())["chosen_batch"]
        expected = json.loads((self.root / "parity.json").read_text())["adapted"]
        baseline = np.array([[r["raw"], r["objective"]] for r in expected])
        trials = []
        queries = (
            json.loads(path.read_text()).get("logical_queries_attempted", 0)
            if path.exists()
            else 0
        )
        for batch in (1,) if self.detector == "bfree" else (1, 4):
            durations, peak, valid = [], 0, True
            for _ in range(3):
                queries += len(rows)
                atomic_json(
                    path, {"state": "started", "logical_queries_attempted": queries}
                )
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                tick = time.perf_counter()
                try:
                    result = self.score(rows, image_root=benchmark_root, batch=batch)
                except torch.cuda.OutOfMemoryError:
                    valid = False
                    peak = max(
                        peak,
                        torch.cuda.max_memory_allocated(),
                        torch.cuda.max_memory_reserved(),
                    )
                    torch.cuda.empty_cache()
                    break
                torch.cuda.synchronize()
                durations.append(time.perf_counter() - tick)
                actual = np.array([[r["raw"], r["objective"]] for r in result])
                peak = max(
                    peak,
                    torch.cuda.max_memory_allocated(),
                    torch.cuda.max_memory_reserved(),
                )
                valid &= (
                    bool(np.allclose(actual, baseline, atol=1e-4, rtol=0))
                    and peak < 64 * 1024**3
                )
            trials.append(
                {
                    "batch": batch,
                    "valid": valid,
                    "seconds": float(np.median(durations)) if durations else None,
                    "peak_gpu_bytes": peak,
                }
            )
        accepted = [r for r in trials if r["valid"]]
        if not accepted:
            raise RuntimeError("No validated continuous-objective scoring batch")
        chosen = min(accepted, key=lambda r: r["seconds"])["batch"]
        atomic_json(
            path,
            {
                "state": "complete",
                "chosen_batch": chosen,
                "trials": trials,
                "logical_queries_attempted": queries,
                "scope": "preprocessing and FP32 released classifier; raw probabilities and margins atol=1e-4",
            },
        )
        return chosen


class HeldoutClassifier(SearchClassifier):
    """Native SSP coordinates per frozen subject, shared across noise/codecs."""

    patch_identity = None


def load_autoencoder_pipeline(item):
    """Use the benchmark-verified decoder class for its decoder-only snapshot."""
    import torch
    from diffusers import AutoPipelineForImage2Image, KandinskyImg2ImgPipeline

    if item["name"] == "kandinsky-2-1":
        if (item["repo"], item["revision"]) != (
            "kandinsky-community/kandinsky-2-1",
            "04b366659663bf60deacb96fd3e38f1cb39a3be4",
        ):
            raise ValueError("Unverified Kandinsky decoder revision")
        index = json.loads((Path(item["snapshot"]) / "model_index.json").read_text())
        if index["_class_name"] != "KandinskyPipeline" or index["movq"] != [
            "diffusers",
            "VQModel",
        ]:
            raise ValueError("Unexpected pinned Kandinsky decoder definition")
        factory, variant = KandinskyImg2ImgPipeline, None
    elif item["name"] in ("stable-diffusion-v1-1", "stable-diffusion-2-base"):
        factory, variant = AutoPipelineForImage2Image, "fp16"
    else:
        raise ValueError("Unexpected AEROBLADE autoencoder")
    return factory.from_pretrained(
        item["snapshot"],
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant=variant,
        local_files_only=True,
    )


class SearchAeroblade:
    """Three released AEs; reset latent sampling per image for stable fitness."""

    def __init__(self, config, assets):
        import sys

        import torch

        self.root = Path(config["output"])
        self.asset = assets["detectors"]["aeroblade"]
        sys.path.insert(0, str(Path(self.asset["source"]) / "src"))
        from aeroblade.distances import _PatchedLPIPS

        self.metric = _PatchedLPIPS(spatial=True, net="vgg").cuda().eval()
        self.components = []
        for item in self.asset["autoencoders"]:
            pipe = load_autoencoder_pipeline(item)
            pipe.enable_model_cpu_offload()
            if hasattr(pipe, "vae"):
                ae = pipe.vae
                if hasattr(pipe, "upcast_vae"):
                    pipe.upcast_vae()
            else:
                ae = pipe.movq
            ae = torch.compile(ae.to("cuda"))
            self.components.append(
                {
                    "asset": item,
                    "pipe": pipe,
                    "ae": ae,
                    "decode_dtype": next(iter(ae.post_quant_conv.parameters())).dtype,
                }
            )

    def _component(self, tensor, component):
        import torch
        from diffusers.models import VQModel
        from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
            retrieve_latents,
        )
        from torchvision.transforms.functional import to_tensor
        from torchvision.transforms.v2.functional import to_pil_image

        ae = component["ae"]
        with torch.inference_mode():
            inputs = tensor.cuda().to(ae.dtype) * 2 - 1
            latent = retrieve_latents(
                ae.encode(inputs), generator=torch.Generator().manual_seed(1)
            )
            kwargs = {"force_not_quantize": True} if isinstance(ae, VQModel) else {}
            reconstruction = ae.decode(
                latent.to(component["decode_dtype"]), return_dict=False, **kwargs
            )[0]
            image = to_pil_image((reconstruction[0] / 2 + 0.5).clamp(0, 1))
            restored = to_tensor(image).unsqueeze(0).cuda()
            _, layers = self.metric(
                tensor.cuda(), restored, retPerLayer=True, normalize=True
            )
            error = float(
                layers[1].to(device="cpu", dtype=torch.float16).mean((2, 3)).item()
            )
        return error, image

    def score(self, rows, *, image_root=None):
        import math

        from PIL import Image
        from torchvision.transforms.functional import to_tensor

        image_root = Path(image_root or self.root).resolve()
        records = []
        for row in rows:
            path = image_root / row["path"]
            if (
                not path.resolve().is_relative_to(image_root)
                or hashes(path)["sha256"] != row["sha256"]
            ):
                raise ValueError("AEROBLADE search input changed")
            with Image.open(path) as image:
                tensor = to_tensor(image.convert("RGB")).unsqueeze(0)
            errors = [
                self._component(tensor, component)[0] for component in self.components
            ]
            if not all(math.isfinite(error) and error >= 0 for error in errors):
                raise ValueError("Invalid search reconstruction error")
            raw = min(errors)
            records.append(
                {
                    "objective": -raw,
                    "raw": raw,
                    "synthetic_score": -raw,
                    "reconstruction_errors": errors,
                    "reconstruction_seed": 1,
                    "reconstruction_seed_scope": "independently reset per image",
                }
            )
        return records

    def validate_native(self, rows, benchmark_root):
        from unittest.mock import patch

        import numpy as np
        import torch
        from aeroblade.data import ImageFolder
        from aeroblade.distances import LPIPS, _compute_lpips
        from aeroblade.image import compute_reconstructions

        adapted = self.score(rows, image_root=benchmark_root)
        native = []
        for i, row in enumerate(rows):
            source = ImageFolder([Path(benchmark_root) / row["path"]])
            errors = []
            for component in self.components:
                item = component["asset"]
                output = self.root / "native" / item["name"] / f"{i:04d}"
                # Reuse an already offline-loaded official pipeline rather than
                # reloading its unused denoiser for every scalar reference.
                with (
                    torch.no_grad(),
                    patch(
                        "aeroblade.image.AutoPipelineForImage2Image.from_pretrained",
                        return_value=component["pipe"],
                    ),
                ):
                    paths = compute_reconstructions(
                        source,
                        item["snapshot"],
                        output_dir=output,
                        seed=1,
                        batch_size=1,
                        num_workers=0,
                    )
                # Official LPIPS materializes NumPy arrays; these diagnostics
                # do not need an autograd graph. Keep native formulas unchanged.
                with torch.no_grad():
                    layers = _compute_lpips.func(
                        source, ImageFolder(paths), {"net": "vgg"}, 1, 0
                    )
                metric = LPIPS(net="vgg", layer=2)
                # The release negates distances in _postprocess; retain the
                # positive reconstruction error and orient fitness separately.
                errors.append(
                    -float(metric._postprocess(layers)["lpips_vgg_2"].flatten().item())
                )
            native.append(errors)
        expected = [r["reconstruction_errors"] for r in adapted]
        if not np.allclose(expected, native, atol=1e-4, rtol=1e-3):
            raise RuntimeError(
                "Independent-seed AEROBLADE differs from official scalar reconstruction"
            )
        repeat = self.score(rows, image_root=benchmark_root)
        if not np.allclose(
            expected, [r["reconstruction_errors"] for r in repeat], atol=1e-4, rtol=1e-3
        ):
            raise RuntimeError("Search AEROBLADE scalar errors are not repeatable")
        reload = SearchAeroblade(
            {"output": str(self.root)}, {"detectors": {"aeroblade": self.asset}}
        )
        reloaded = reload.score(rows, image_root=benchmark_root)
        if not np.allclose(
            expected,
            [r["reconstruction_errors"] for r in reloaded],
            atol=1e-4,
            rtol=1e-3,
        ):
            raise RuntimeError("Offline search AEROBLADE reload differs")
        atomic_json(
            self.root / "parity.json",
            {
                "passed": True,
                "native_errors": native,
                "native_reference_pipeline_reused": True,
                "native_autograd_disabled": True,
                "decoder_loader": "pinned KandinskyImg2ImgPipeline decoder-only",
                "adapted": adapted,
                "repeat": repeat,
                "offline": reloaded,
            },
        )
        del reload
