"""Official networks in isolated stage kernels, with explicit score polarity."""

import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

from evolutionary_extensions.persistence.packaging import atomic_json, hashes

from .data import patch_coordinates, simplest_patch


def score_polarity(detector, raw):
    if detector == "ssp":
        return 1 - raw
    if detector == "aeroblade":
        return -raw
    return raw


class Classifier:
    def __init__(self, detector, config, assets):
        import torch
        from torchvision import transforms as tf

        self.detector, self.root, self.config = detector, Path(config["output"]), config
        self.asset = assets["detectors"][detector]
        source = Path(self.asset["source"])
        # Vendor packages reuse names such as models/networks/utils; every stage
        # must run in a fresh managed kernel to avoid cross-vendor import aliases.
        sys.path.insert(0, str(source / "code" if detector == "bfree" else source))
        imagenet = tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        if detector == "safe":
            self.model = importlib.import_module("models.resnet").resnet50(
                num_classes=2
            )
            state = torch.load(
                self.asset["checkpoint"], map_location="cpu", weights_only=False
            )["model"]
            self.model.load_state_dict(state, strict=True)
            self.transform = tf.Compose([tf.CenterCrop(256), tf.ToTensor()])
        elif detector == "universal_fake_detect":
            clip = importlib.import_module("models.clip.clip")
            original_load = clip.load
            # Use the released wrapper and a SHA-bound offline CLIP checkpoint.
            clip.load = lambda name, device: original_load(
                self.asset["clip_checkpoint"], device=device
            )
            try:
                self.model = importlib.import_module("models.clip_models").CLIPModel(
                    "ViT-L/14"
                )
            finally:
                clip.load = original_load
            self.model.fc.load_state_dict(
                torch.load(
                    self.asset["checkpoint"], map_location="cpu", weights_only=True
                )
            )
            self.transform = tf.Compose(
                [
                    tf.CenterCrop(224),
                    tf.ToTensor(),
                    tf.Normalize(
                        (0.48145466, 0.4578275, 0.40821073),
                        (0.26862954, 0.26130258, 0.27577711),
                    ),
                ]
            )
        elif detector == "ssp":
            module = importlib.import_module("networks.ssp")
            original = module.resnet50
            module.resnet50 = lambda pretrained: original(pretrained=False)
            try:
                self.model = module.ssp()
            finally:
                module.resnet50 = original
            state = torch.load(
                self.asset["checkpoint"], map_location="cpu", weights_only=True
            )
            self.model.load_state_dict(state, strict=True)
            self.transform = tf.Compose([tf.ToTensor(), imagenet])
        elif detector == "bfree":
            import yaml

            network = importlib.import_module("networks")
            norm = importlib.import_module("utils.normalization")
            spec = yaml.safe_load(Path(self.asset["model_config"]).read_text())
            self.model = network.load_weights(
                network.get_network(spec["arch"]), self.asset["checkpoint"]
            )
            self.transform = tf.Compose(norm.get_list_norm(spec["norm_type"]))
        else:
            raise ValueError(detector)
        self.model = self.model.float().cuda().eval()

    def prepare(self, rows):
        import torch
        from PIL import Image

        tensors, details = [], []
        for row in rows:
            path = self.root / row["path"]
            if (
                not path.resolve().is_relative_to(self.root.resolve())
                or hashes(path)["sha256"] != row["sha256"]
            ):
                raise ValueError("Scoring input differs from its frozen image identity")
            with Image.open(path) as file:
                image = file.convert("RGB")
            detail = {}
            if self.detector == "ssp":
                coordinates = patch_coordinates(
                    row["image_id"], seed=self.config["patch_seed"]
                )
                image, index = simplest_patch(image, coordinates)
                relative = Path("patches") / row["codec"] / f"{row['image_id']}.png"
                path = self.root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path)
                detail = {
                    "coordinates": coordinates,
                    "selected_index": index,
                    "patch_path": relative.as_posix(),
                    "patch_sha256": hashes(path)["sha256"],
                }
                atomic_json(path.with_suffix(".json"), detail)
            tensors.append(self.transform(image))
            details.append(detail)
        return torch.stack(tensors), details

    def infer(self, tensor):
        import torch

        with torch.inference_mode():
            output = self.model(tensor.cuda())
            if self.detector == "safe":
                raw = output.softmax(dim=1)[:, 1]
            elif self.detector == "bfree":
                raw = (
                    output[:, 0]
                    if output.shape[1] == 1
                    else output[:, 1] - output[:, 0]
                )
            else:
                raw = output.sigmoid().flatten()
        return raw.cpu().double().numpy()

    def official_scalar(self, row):
        """Separate native transformations and network calls for smoke parity."""
        import torch
        from PIL import Image

        if self.detector != "ssp":
            tensor, _ = self.prepare([row])
            if self.detector == "safe":
                from types import SimpleNamespace

                module = importlib.import_module("data.datasets")
                args = SimpleNamespace(
                    input_size=256,
                    transform_mode="crop",
                    jpeg_factor=None,
                    blur_sigma=None,
                    mask_ratio=None,
                    mask_patch_size=None,
                )
                transform = module.Get_Transforms(args)[1]
                with Image.open(self.root / row["path"]) as image:
                    native_tensor = transform(image.convert("RGB")).unsqueeze(0)
                if not torch.equal(native_tensor, tensor):
                    raise ValueError("SAFE official preprocessing differs")
            # All other classifier networks are loaded directly from pinned sources.
            return self.infer(tensor)[0]
        # SSP's released constructor unconditionally downloads ResNet weights.
        # Avoid that redundant initialization; the released SSP checkpoint replaces
        # every parameter, including the ImageNet classifier and SRM filters.
        module = importlib.import_module("networks.ssp")
        original = module.resnet50
        module.resnet50 = lambda pretrained: original(pretrained=False)
        try:
            native = module.ssp().float().cuda().eval()
        finally:
            module.resnet50 = original
        native.load_state_dict(
            torch.load(self.asset["checkpoint"], map_location="cpu", weights_only=True),
            strict=True,
        )
        patch_module = importlib.import_module("utils.patch")
        with Image.open(self.root / row["path"]) as image:
            image = image.convert("RGB")
        seed = int.from_bytes(
            hashlib.sha256(
                f"{self.config['patch_seed']}:{row['image_id']}".encode()
            ).digest()[:8],
            "big",
        ) % (2**63)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            patch = patch_module.patch_img(image, 32, 256)
        tensor = self.transform(patch).unsqueeze(0)
        with torch.inference_mode():
            value = float(native(tensor.cuda()).sigmoid().item())
        del native
        return value


class Aeroblade:
    """Native AE components reused across batches, with per-batch RNG receipts."""

    def __init__(self, config, assets):
        self.root, self.asset = Path(config["output"]), assets["detectors"]["aeroblade"]
        self.config = config
        sys.path.insert(0, str(Path(self.asset["source"]) / "src"))

    def run(self, rows, journal, progress, disk_guard):
        import base64
        import gc

        import numpy as np
        import torch
        from aeroblade.data import ImageFolder
        from aeroblade.distances import LPIPS, _compute_lpips, _PatchedLPIPS
        from aeroblade.image import compute_reconstructions
        from diffusers import AutoPipelineForImage2Image
        from diffusers.models import VQModel
        from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
            retrieve_latents,
        )
        from torchvision.transforms.functional import to_tensor
        from torchvision.transforms.v2.functional import to_pil_image

        from .runner import profile_classifier, smoke_pool

        class ReconstructionAdapter:
            def __init__(self, snapshot):
                self.pipe = AutoPipelineForImage2Image.from_pretrained(
                    snapshot,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant=None if "kandinsky-2" in snapshot else "fp16",
                    local_files_only=True,
                )
                self.pipe.enable_model_cpu_offload()
                if hasattr(self.pipe, "vae"):
                    self.ae = self.pipe.vae
                    if hasattr(self.pipe, "upcast_vae"):
                        self.pipe.upcast_vae()
                else:
                    self.ae = self.pipe.movq
                self.ae.to("cuda")
                self.ae = torch.compile(self.ae)
                self.decode_dtype = next(
                    iter(self.ae.post_quant_conv.parameters())
                ).dtype
                self.metric = _PatchedLPIPS(spatial=True, net="vgg").cuda().eval()
                self.reset()

            def reset(self):
                self.generator = torch.Generator().manual_seed(1)

            def infer(self, tensor):
                with torch.inference_mode():
                    normalized = tensor.to("cuda", dtype=self.ae.dtype) * 2 - 1
                    latent = retrieve_latents(
                        self.ae.encode(normalized), generator=self.generator
                    )
                    kwargs = (
                        {"force_not_quantize": True}
                        if isinstance(self.ae, VQModel)
                        else {}
                    )
                    decoded = self.ae.decode(
                        latent.to(self.decode_dtype), return_dict=False, **kwargs
                    )[0]
                    decoded = (decoded / 2 + 0.5).clamp(0, 1)
                    # The official detector saves PNGs and reloads 8-bit pixels.
                    self.reconstructions = [to_pil_image(image) for image in decoded]
                    reconstructed = torch.stack(
                        [to_tensor(image) for image in self.reconstructions]
                    )
                    _, layers = self.metric(
                        tensor.cuda(),
                        reconstructed.cuda(),
                        retPerLayer=True,
                        normalize=True,
                    )
                    return (
                        layers[1]
                        .to(device="cpu", dtype=torch.float16)
                        .mean((2, 3))
                        .flatten()
                        .double()
                        .numpy()
                    )

        smoke_rows = smoke_pool(rows)
        smoke_ds = ImageFolder([self.root / r["path"] for r in smoke_rows])
        smoke = torch.stack([smoke_ds[i][0] for i in range(16)])
        errors = []
        profiles = []
        timing_path = self.root / "timing/aeroblade.json"
        timing = (
            json.loads(timing_path.read_text())
            if timing_path.exists()
            else {
                "cold_load_seconds": 0.0,
                "preprocessing_seconds": 0.0,
                "inference_seconds": 0.0,
                "persistence_seconds": 0.0,
            }
        )

        def validate_saved(values):
            for value in values:
                if not np.isfinite(value["error"]) or value["error"] < 0:
                    raise ValueError("Invalid cached AEROBLADE error")
                path = self.root / value["reconstruction_path"]
                if (
                    not path.resolve().is_relative_to(self.root.resolve())
                    or hashes(path)["sha256"] != value["reconstruction_sha256"]
                ):
                    raise ValueError("Changed cached AEROBLADE reconstruction")

        for autoencoder in self.asset["autoencoders"]:
            component = self.root / "aeroblade" / autoencoder["name"]
            by_key = {}
            pending_component = False
            for codec in ("png", "jpeg95"):
                pool = [r for r in rows if r["codec"] == codec]
                frozen = component / f"{codec}-complete.json"
                if frozen.exists():
                    saved = json.loads(frozen.read_text())
                    if saved["detector_identity"] != journal.identity:
                        raise ValueError("Changed AEROBLADE component configuration")
                    values = saved["errors"]
                    if [(r["image_id"], r["sha256"]) for r in values] != [
                        (r["image_id"], r["sha256"]) for r in pool
                    ]:
                        raise ValueError("Changed AEROBLADE component input")
                    validate_saved(values)
                    by_key.update({(r["image_id"], codec): r["error"] for r in values})
                else:
                    pending_component = True
            if not pending_component:
                profiles.append(
                    json.loads(
                        (
                            self.root
                            / "profiles"
                            / f"aeroblade-{autoencoder['name']}.json"
                        ).read_text()
                    )
                )
                errors.append(by_key)
                continue
            disk_guard()
            tick = time.perf_counter()
            adapter = ReconstructionAdapter(autoencoder["snapshot"])
            torch.cuda.synchronize()
            timing["cold_load_seconds"] += time.perf_counter() - tick
            name = "aeroblade-" + autoencoder["name"]
            batch, baseline = profile_classifier(
                adapter, smoke, self.root, name, batches=(1, 4), atol=1e-4, rtol=1e-3
            )
            profile = json.loads((self.root / "profiles" / f"{name}.json").read_text())
            profiles.append(profile)
            batching = component / "batching.json"
            if batching.exists():
                frozen_batch = json.loads(batching.read_text())
                if frozen_batch["detector_identity"] != journal.identity:
                    raise ValueError("Changed AEROBLADE batching configuration")
                batch = frozen_batch["batch"]
                if not any(
                    p["batch"] == batch and p["valid"] for p in profile["candidates"]
                ):
                    raise RuntimeError(
                        "Frozen recovery batch no longer passes validation"
                    )
            else:
                atomic_json(
                    batching, {"detector_identity": journal.identity, "batch": batch}
                )
            # Compare with the release's full reconstruction + LPIPS pipeline.
            native_folder = self.root / "smoke" / name / f"native-{time.time_ns()}"
            paths = compute_reconstructions(
                smoke_ds,
                repo_id=autoencoder["snapshot"],
                output_dir=native_folder,
                seed=1,
                batch_size=1,
                num_workers=0,
            )
            native_metric = LPIPS(net="vgg", layer=2, batch_size=1, num_workers=0)
            native = native_metric._postprocess(
                _compute_lpips.func(smoke_ds, ImageFolder(paths), {"net": "vgg"}, 1, 0)
            )
            native = -native["lpips_vgg_2"].flatten().double().numpy()
            if not np.allclose(native, baseline, atol=1e-4, rtol=1e-3):
                raise RuntimeError(
                    "Official AEROBLADE reconstruction/error parity failed"
                )
            del native_metric, adapter
            gc.collect()
            torch._dynamo.reset()
            torch.cuda.empty_cache()
            adapter = ReconstructionAdapter(autoencoder["snapshot"])
            adapter.reset()
            repeated = np.concatenate(
                [adapter.infer(smoke[i : i + 1]) for i in range(16)]
            )
            if not np.allclose(repeated, baseline, atol=1e-4, rtol=1e-3):
                raise RuntimeError("Offline AEROBLADE model reload parity failed")
            atomic_json(
                self.root / "parity" / f"{name}.json",
                {
                    "passed": True,
                    "native_errors": native.tolist(),
                    "scalar_errors": baseline.tolist(),
                    "offline_errors": repeated.tolist(),
                },
            )
            for codec in ("png", "jpeg95"):
                pool = [r for r in rows if r["codec"] == codec]
                if (component / f"{codec}-complete.json").exists():
                    continue
                ds = ImageFolder([self.root / r["path"] for r in pool])
                adapter.reset()
                values = []
                for offset in range(0, len(pool), batch):
                    group = pool[offset : offset + batch]
                    receipt = component / codec / f"batch-{offset:04d}.json"
                    expected = [(r["image_id"], r["sha256"]) for r in group]
                    if receipt.exists():
                        saved = json.loads(receipt.read_text())
                        if (
                            saved["detector_identity"] != journal.identity
                            or [(r["image_id"], r["sha256"]) for r in saved["errors"]]
                            != expected
                        ):
                            raise ValueError("AEROBLADE recovery configuration differs")
                        validate_saved(saved["errors"])
                        state = torch.frombuffer(
                            bytearray(base64.b64decode(saved["rng_after"])),
                            dtype=torch.uint8,
                        ).clone()
                        adapter.generator.set_state(state)
                        values.extend(saved["errors"])
                        continue
                    disk_guard()
                    tick = time.perf_counter()
                    inputs = torch.stack(
                        [
                            ds[i][0]
                            for i in range(offset, min(offset + batch, len(pool)))
                        ]
                    )
                    timing["preprocessing_seconds"] += time.perf_counter() - tick
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    output = adapter.infer(inputs)
                    torch.cuda.synchronize()
                    timing["inference_seconds"] += time.perf_counter() - tick
                    if not np.isfinite(output).all() or (output < 0).any():
                        raise ValueError("Invalid AEROBLADE error")
                    tick = time.perf_counter()
                    batch_values = []
                    for row, error, image in zip(
                        group, output, adapter.reconstructions
                    ):
                        path = (
                            component
                            / "reconstructions"
                            / codec
                            / f"{row['image_id']}.png"
                        )
                        path.parent.mkdir(parents=True, exist_ok=True)
                        image.save(path)
                        batch_values.append(
                            {
                                "image_id": row["image_id"],
                                "sha256": row["sha256"],
                                "error": float(error),
                                "reconstruction_path": str(path.relative_to(self.root)),
                                "reconstruction_sha256": hashes(path)["sha256"],
                            }
                        )
                    atomic_json(
                        receipt,
                        {
                            "detector_identity": journal.identity,
                            "errors": batch_values,
                            "rng_after": base64.b64encode(
                                adapter.generator.get_state().numpy().tobytes()
                            ).decode(),
                        },
                    )
                    values.extend(batch_values)
                    timing["persistence_seconds"] += time.perf_counter() - tick
                    atomic_json(timing_path, timing)
                    progress(
                        autoencoder=autoencoder["name"],
                        codec=codec,
                        component_completed=len(values),
                        component_total=len(pool),
                    )
                atomic_json(
                    component / f"{codec}-complete.json",
                    {"detector_identity": journal.identity, "errors": values},
                )
                by_key.update({(r["image_id"], codec): r["error"] for r in values})
            errors.append(by_key)
            del adapter
            gc.collect()
            torch._dynamo.reset()
            torch.cuda.empty_cache()
        for row in rows:
            if journal.existing(row) is None:
                values = [e[row["image_id"], row["codec"]] for e in errors]
                raw = min(values)
                journal.write(row, raw, -raw, reconstruction_errors=values)
        if profiles:
            candidates = []
            for batch in (1, 4):
                selected = [
                    next(r for r in p["candidates"] if r["batch"] == batch)
                    for p in profiles
                ]
                duration = sum(
                    r["seconds"] for r in selected if r["seconds"] is not None
                )
                candidates.append(
                    {
                        "batch": batch,
                        "valid": all(r["valid"] for r in selected),
                        "seconds": duration if duration > 0 else None,
                        "images_per_second": 16 / duration if duration > 0 else None,
                        "peak_bytes": max(r["peak_bytes"] for r in selected),
                    }
                )
            atomic_json(
                self.root / "profiles/aeroblade.json",
                {
                    "candidates": candidates,
                    "chosen_component_batches": [p["chosen_batch"] for p in profiles],
                    "profile_scope": "All three reconstruction + LPIPS passes; components loaded sequentially",
                },
            )
        atomic_json(self.root / "timing/aeroblade.json", timing)
