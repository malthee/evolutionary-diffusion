"""Shared held-out pixels from frozen finalists and unselected controls."""

import json
import shutil
import time
from pathlib import Path

from evolutionary_extensions.experiments.detector_evaluation.config import (
    MODEL_REVISION,
    digest,
)
from evolutionary_extensions.persistence.packaging import atomic_json, hashes

from .protocol import heldout_plan, plan_targets


def safe_subject(subject):
    name = subject["subject"]
    if not isinstance(name, str) or Path(name).name != name or name in (".", "..", ""):
        raise ValueError("Unsafe held-out subject identity")
    return name


def checked_path(root, relative, sha):
    root = Path(root).resolve()
    path = root / relative
    if not path.resolve().is_relative_to(root) or hashes(path)["sha256"] != sha:
        raise ValueError("Frozen held-out source checksum differs")
    return path


def prepare_inputs(search_root, benchmark_root, benchmark_rows):
    """CPU preparation freezes the actual FP16 tensors before new rendering."""
    import torch
    from safetensors.torch import load_file, save_file

    search_root, benchmark_root = Path(search_root), Path(benchmark_root)
    plan = json.loads((search_root / "heldout_plan.json").read_text())
    targets = plan_targets(plan)
    finalists = {
        name: json.loads((search_root / name / "finalists.json").read_text())
        for name in targets
    }
    plan = json.loads((search_root / "heldout_plan.json").read_text())
    if plan != heldout_plan(finalists, benchmark_rows):
        raise ValueError("Finalists or unselected controls changed after plan freeze")
    for target in targets:
        complete = json.loads(
            (search_root / target / "search_complete.json").read_text()
        )
        if complete.get("status") != "complete" or complete.get("candidates") != 6400:
            raise ValueError(
                "All selected optimizer budgets must finish before held-out rendering"
            )
    root = search_root / "heldout"
    root.mkdir(exist_ok=True)
    baseline_path = benchmark_root / "embeddings.safetensors"
    baseline_sha = hashes(baseline_path)["sha256"]
    if (
        json.loads((benchmark_root / "generation.json").read_text())["embedding_sha256"]
        != baseline_sha
    ):
        raise ValueError("Benchmark control embeddings changed")
    baseline = load_file(str(baseline_path))
    by_id = {r["image_id"]: r for r in benchmark_rows if r["codec"] == "png"}
    tensors, sources, loaded = [], [], {}
    for slot, subject in enumerate(plan["subjects"]):
        safe_subject(subject)
        target = subject["target"]
        if target is None:
            image = by_id[subject["subject"]]
            image_path = checked_path(
                benchmark_root, image["path"], subject["noise0_sha256"]
            )
            index = subject["embedding_index"]
            values = {k: v[index : index + 1] for k, v in baseline.items()}
            source = {
                "benchmark_embedding_sha256": baseline_sha,
                "embedding_index": index,
            }
        else:
            index = subject["candidate_index"]
            selected = next(
                r for r in finalists[target]["records"] if r["index"] == index
            )
            image = selected["image"]
            if (
                image["candidate_index"] != index
                or image["image_id"] != subject["subject"]
            ):
                raise ValueError("Finalist candidate identity differs")
            folder = search_root / target
            image_path = checked_path(folder, image["path"], subject["noise0_sha256"])
            batch = json.loads(
                (folder / "generation" / f"batch-{index // 4 * 4:05d}.json").read_text()
            )
            if batch["images"][image["embedding_slot"]] != image:
                raise ValueError("Finalist image differs from its generation receipt")
            embedding_path = checked_path(
                folder, batch["embedding_path"], batch["embedding_sha256"]
            )
            if str(embedding_path) not in loaded:
                loaded[str(embedding_path)] = load_file(str(embedding_path))
            values = {
                k: v[image["embedding_slot"] : image["embedding_slot"] + 1]
                for k, v in loaded[str(embedding_path)].items()
            }
            source = {
                "target": target,
                "candidate_index": index,
                "embedding_sha256": batch["embedding_sha256"],
                "embedding_slot": image["embedding_slot"],
            }
        if (
            set(values) != {"prompt_embeds", "pooled_prompt_embeds"}
            or values["prompt_embeds"].shape != (1, 77, 2048)
            or values["pooled_prompt_embeds"].shape != (1, 1280)
            or any(
                v.dtype != torch.float16 or not torch.isfinite(v).all()
                for v in values.values()
            )
        ):
            raise ValueError("Held-out actual FP16 embedding is invalid")
        tensors.append(values)
        relative = Path("noise0") / f"{subject['subject']}.png"
        destination = root / relative
        destination.parent.mkdir(exist_ok=True)
        if destination.exists():
            checked_path(root, relative, subject["noise0_sha256"])
        else:
            temporary = destination.with_suffix(".tmp")
            shutil.copyfile(image_path, temporary)
            temporary.replace(destination)
            checked_path(root, relative, subject["noise0_sha256"])
        sources.append(
            {
                **subject,
                "noise0_path": relative.as_posix(),
                "embedding_slot": slot,
                "embedding_source": source,
            }
        )
    merged = {k: torch.cat([t[k] for t in tensors]) for k in tensors[0]}
    path = root / "embeddings.safetensors"
    if path.exists():
        saved = load_file(str(path))
        if set(saved) != set(merged) or any(
            not torch.equal(saved[k], merged[k]) for k in saved
        ):
            raise ValueError("Frozen held-out tensor pool changed")
    else:
        temporary = path.with_suffix(".tmp")
        save_file(merged, str(temporary))
        temporary.replace(path)
    recipe = {
        "plan_sha256": digest(plan),
        "subjects": sources,
        "embedding_sha256": hashes(path)["sha256"],
        "noise0_policy": "Reuse exact archived PNG bytes; no selected-image regeneration",
        "generation": {
            "model_revision": MODEL_REVISION,
            "steps": 1,
            "guidance": 0,
            "precision": "FP16",
            "size": [512, 512],
            "physical_batch": 4,
            "diffusion_seeds": [1, 2, 3],
        },
    }
    frozen = root / "inputs.json"
    if frozen.exists() and json.loads(frozen.read_text()) != recipe:
        raise ValueError("Frozen held-out source pool changed")
    atomic_json(frozen, recipe)
    return root, recipe


def write_variants(image, root, subject, seed, *, png_source=None):
    """All JPEG variants derive directly from the same canonical RGB pixels."""
    from PIL import Image

    root = Path(root)
    safe_subject(subject)
    if type(seed) is not int or seed not in (0, 1, 2, 3):
        raise ValueError("Unknown held-out diffusion seed")
    if image.mode != "RGB" or image.size != (512, 512):
        raise ValueError("Held-out images must be RGB 512x512")
    if png_source is not None:
        with Image.open(png_source) as source:
            if (
                source.size != image.size
                or source.convert("RGB").tobytes() != image.tobytes()
            ):
                raise ValueError("Noise-zero PNG differs from JPEG source pixels")
    rows = []
    for codec in ("png", "jpeg95", "jpeg75"):
        extension = "png" if codec == "png" else "jpg"
        relative = (
            Path("images") / codec / f"{subject['subject']}-noise{seed}.{extension}"
        )
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        if codec == "png" and png_source is not None:
            shutil.copyfile(png_source, temporary)
        elif codec == "png":
            image.save(temporary, format="PNG")
        else:
            image.save(temporary, format="JPEG", quality=int(codec[4:]), subsampling=0)
        temporary.replace(path)
        with Image.open(path) as saved:
            if saved.size != (512, 512) or saved.mode != "RGB":
                raise ValueError("Exported held-out pixels have invalid dimensions")
        rows.append(
            {
                "subject": subject["subject"],
                "image_id": f"{subject['subject']}-noise{seed}",
                "group": subject["group"],
                "target": subject["target"],
                "diffusion_seed": seed,
                "codec": codec,
                "path": relative.as_posix(),
                "sha256": hashes(path)["sha256"],
                "embedding_slot": subject["embedding_slot"],
            }
        )
    return rows


def render_variants(
    root, recipe, render, *, guard=lambda: None, progress=lambda **k: None
):
    """Render callback receives four subjects with one independently reset seed.

    Receipt recovery never renders completed batches. Partial final batches
    retain physical size four and explicitly count their repeated padding.
    """
    from PIL import Image

    root = Path(root)
    checked_path(root, "embeddings.safetensors", recipe["embedding_sha256"])
    subjects = recipe["subjects"]
    identity = digest(recipe)
    records = []
    ledger = root / "generation_attempts"
    attempts = sorted(ledger.glob("*.json"))
    if [p.name for p in attempts] != [f"{i:06d}.json" for i in range(len(attempts))]:
        raise ValueError("Held-out rendering attempt ledger is not contiguous")
    next_attempt = len(attempts)
    for path in attempts:
        record = json.loads(path.read_text())
        if record["identity"] != identity or record["state"] not in {
            "started",
            "complete",
        }:
            raise ValueError("Held-out rendering attempt provenance differs")
    for seed in (0, 1, 2, 3):
        for offset in range(0, len(subjects), 4):
            group = subjects[offset : offset + 4]
            receipt = root / "generation" / f"seed{seed}-batch{offset:04d}.json"
            names = [s["subject"] for s in group]
            if receipt.exists():
                saved = json.loads(receipt.read_text())
                if (
                    saved["identity"] != identity
                    or saved["subjects"] != names
                    or saved["seed"] != seed
                ):
                    raise ValueError("Held-out rendering receipt changed")
                rows = saved["rows"]
                if [(r["subject"], r["codec"]) for r in rows] != [
                    (name, codec)
                    for name in names
                    for codec in ("png", "jpeg95", "jpeg75")
                ]:
                    raise ValueError("Held-out rendered image order differs")
                for row in rows:
                    checked_path(root, row["path"], row["sha256"])
                records.extend(rows)
                continue
            guard()
            tick = time.perf_counter()
            intent = None
            if seed == 0:
                outputs = []
                for subject in group:
                    source = checked_path(
                        root, subject["noise0_path"], subject["noise0_sha256"]
                    )
                    with Image.open(source) as file:
                        outputs.append(file.convert("RGB"))
            else:
                padded = group + [group[-1]] * (4 - len(group))
                intent = ledger / f"{next_attempt:06d}.json"
                atomic_json(
                    intent,
                    {
                        "identity": identity,
                        "subjects": names,
                        "seed": seed,
                        "physical_images": 4,
                        "distinct_images": len(group),
                        "padding_images": 4 - len(group),
                        "state": "started",
                        "started_unix": time.time(),
                    },
                )
                next_attempt += 1
                outputs = render(padded, seed)
                if len(outputs) != 4:
                    raise ValueError("Held-out generation output count differs")
                outputs = outputs[: len(group)]
            rows = []
            for subject, image in zip(group, outputs):
                rows.extend(
                    write_variants(
                        image,
                        root,
                        subject,
                        seed,
                        png_source=root / subject["noise0_path"] if seed == 0 else None,
                    )
                )
            saved = {
                "identity": identity,
                "subjects": names,
                "seed": seed,
                "rows": rows,
                "seconds": time.perf_counter() - tick,
            }
            atomic_json(receipt, saved)
            if intent is not None:
                atomic_json(
                    intent,
                    {
                        **json.loads(intent.read_text()),
                        "state": "complete",
                        "receipt": str(receipt.relative_to(root)),
                        "finished_unix": time.time(),
                    },
                )
            records.extend(rows)
            progress(
                stage="heldout_generation",
                variants=len(records),
                total=len(subjects) * 12,
            )
    atomic_json(root / "images.json", records)
    return records


class NoiseRenderer:
    """One offline generator reused for all three held-out noise seeds."""

    def __init__(self, root, recipe, assets):
        from safetensors.torch import load_file

        from .generator_validation import load_generator

        if assets["generation"]["revision"] != MODEL_REVISION:
            raise ValueError("Held-out generator revision differs from the benchmark")
        self.tensors = load_file(
            str(
                checked_path(root, "embeddings.safetensors", recipe["embedding_sha256"])
            )
        )
        self.pipe = load_generator(assets)

    def __call__(self, subjects, seed):
        import torch

        from evolutionary_prompt_embedding.argument_types import PooledPromptEmbedData
        from evolutionary_prompt_embedding.image_creation import (
            SDXLPromptEmbeddingImageCreator,
        )

        if seed not in (1, 2, 3) or len(subjects) != 4:
            raise ValueError(
                "Require one held-out diffusion seed and physical batch four"
            )
        arguments = [
            PooledPromptEmbedData(
                self.tensors["prompt_embeds"][
                    s["embedding_slot"] : s["embedding_slot"] + 1
                ],
                self.tensors["pooled_prompt_embeds"][
                    s["embedding_slot"] : s["embedding_slot"] + 1
                ],
            )
            for s in subjects
        ]
        creator = SDXLPromptEmbeddingImageCreator(
            1, 1, fixed_noise_seeds=[seed], pipeline=self.pipe
        )
        outputs = creator.create_solutions(arguments)
        torch.cuda.synchronize()
        if (
            max(torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
            >= 64 * 1024**3
        ):
            raise RuntimeError("Held-out generation exceeds 64 GiB peak memory")
        return [output.result.images[0] for output in outputs]
