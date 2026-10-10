"""One fixed shared synthetic image pool, using the existing embedding creator."""

import json
import time
from pathlib import Path

from evolutionary_extensions.persistence.packaging import atomic_json, hashes

from .data import read_csv, write_variants


def generate(config):
    import torch
    from diffusers import DiffusionPipeline
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file

    from evolutionary_extensions.experiments.prompt_embedding.initialization import (
        initial_arguments,
        load_bounds,
    )
    from evolutionary_prompt_embedding.argument_types import PooledPromptEmbedData
    from evolutionary_prompt_embedding.image_creation import (
        SDXLPromptEmbeddingImageCreator,
    )

    from .runner import assets_for, cuda_preflight, guard_disk, status

    root = Path(config["output"])
    cuda_preflight(root)
    asset_lock = assets_for(config, "generate")
    started = time.perf_counter()
    snapshot = snapshot_download(
        "stabilityai/sdxl-turbo",
        revision=config["model_revision"],
        cache_dir=asset_lock["hf_cache"],
        local_files_only=True,
    )
    pipeline = DiffusionPipeline.from_pretrained(
        snapshot,
        variant="fp16",
        torch_dtype=torch.float16,
        use_safetensors=True,
        local_files_only=True,
    ).to("cuda")
    pipeline.set_progress_bar_config(disable=True)
    creator = SDXLPromptEmbeddingImageCreator(
        1, 1, fixed_noise_seeds=[config["diffusion_seed"]], pipeline=pipeline
    )
    torch.cuda.synchronize()
    timing = {
        "cold_load_seconds": time.perf_counter() - started,
        "generation_seconds": 0.0,
        "exports_seconds": 0.0,
    }
    bounds, bounds_manifest = load_bounds()
    if (root / "bounds.json").exists() and json.loads(
        (root / "bounds.json").read_text()
    ) != bounds_manifest:
        raise ValueError("Frozen coordinate bounds provenance differs")
    atomic_json(root / "bounds.json", bounds_manifest)
    bounds_path = root / "coordinate_bounds.safetensors"
    if bounds_path.exists():
        saved_bounds = load_file(str(bounds_path))
        if set(saved_bounds) != set(bounds) or any(
            not torch.equal(saved_bounds[k], bounds[k].cpu()) for k in bounds
        ):
            raise ValueError("Archived coordinate bounds differ")
    else:
        save_file({key: value.cpu() for key, value in bounds.items()}, str(bounds_path))
    selection = json.loads((root / "selection.json").read_text())
    prompts = read_csv(config["prompts_csv"])
    # A normalized mapping is prepared from the release's unchanged descriptions.
    by_source = {r["image_id"]: r["prompt"] for r in prompts}
    if len(by_source) != len(prompts):
        raise ValueError("Duplicate Synthbuster prompt identities")
    photos = [r for r in selection if r["group"] == "real"]
    n = config["test_count"]
    torch.manual_seed(config["sample_seed"])
    random_arguments = initial_arguments(bounds, n)
    frozen = root / "embeddings.safetensors"
    if frozen.exists():
        tensors = load_file(str(frozen))
    else:
        prompted = [
            creator.arguments_from_prompt(by_source[row["File"]]) for row in photos
        ]
        arguments = random_arguments + prompted
        tensors = {
            "prompt_embeds": torch.cat([a.prompt_embeds.cpu() for a in arguments]),
            "pooled_prompt_embeds": torch.cat(
                [a.pooled_prompt_embeds.cpu() for a in arguments]
            ),
        }
        save_file(tensors, str(frozen))
    if tensors["prompt_embeds"].shape != (2 * n, 77, 2048) or tensors[
        "pooled_prompt_embeds"
    ].shape != (2 * n, 1280):
        raise ValueError("Frozen embedding shape differs")
    if not all(torch.isfinite(t).all() for t in tensors.values()):
        raise ValueError("Non-finite frozen embeddings")
    for key in tensors:
        expected = torch.cat([getattr(a, key).cpu() for a in random_arguments])
        if not torch.equal(tensors[key][:n], expected):
            raise ValueError(
                "Frozen random embeddings differ from the seeded bounds recipe"
            )
    metadata = [
        {
            "image_id": f"random-{i:04d}",
            "group": "random",
            "pair_id": f"random-{i:04d}",
            "prompt": None,
        }
        for i in range(n)
    ] + [
        {
            "image_id": f"prompted-{r['File']}",
            "group": "prompted",
            "pair_id": r["File"],
            "prompt": by_source[r["File"]],
        }
        for r in photos
    ]
    records = []
    for offset in range(0, 2 * n, 4):
        guard_disk(config)
        group = metadata[offset : offset + 4]
        receipt = root / "generation" / f"batch-{offset:04d}.json"
        if receipt.exists():
            batch = json.loads(receipt.read_text())
            if {(r["image_id"], r["codec"]) for r in batch} != {
                (m["image_id"], c) for m in group for c in ("png", "jpeg95")
            }:
                raise ValueError("Generation receipt order/identity differs")
            for row in batch:
                if hashes(root / row["path"])["sha256"] != row["sha256"]:
                    raise ValueError("Generated image changed")
            records.extend(batch)
            continue
        arguments = [
            PooledPromptEmbedData(
                tensors["prompt_embeds"][i : i + 1],
                tensors["pooled_prompt_embeds"][i : i + 1],
            )
            for i in range(offset, min(offset + 4, 2 * n))
        ]
        # Fixed physical shape also applies when recovering a final partial batch.
        padded = arguments + [arguments[-1]] * (4 - len(arguments))
        torch.cuda.synchronize()
        tick = time.perf_counter()
        outputs = creator.create_solutions(padded)[: len(arguments)]
        torch.cuda.synchronize()
        timing["generation_seconds"] += time.perf_counter() - tick
        tick = time.perf_counter()
        batch = []
        for i, (solution, item) in enumerate(zip(outputs, group)):
            batch.extend(
                write_variants(
                    solution.result.images[0],
                    root,
                    item["image_id"],
                    item["group"],
                    pair_id=item["pair_id"],
                    prompt=item["prompt"],
                    embedding_index=offset + i,
                    diffusion_seed=config["diffusion_seed"],
                )
            )
        atomic_json(receipt, batch)
        timing["exports_seconds"] += time.perf_counter() - tick
        records.extend(batch)
        status(root, "generation", generated_images=min(offset + 4, 2 * n))
    real = json.loads((root / "real_images.json").read_text())
    atomic_json(root / "images.json", real + records)
    atomic_json(
        root / "generation.json",
        {
            "embedding_sha256": hashes(frozen)["sha256"],
            "images": 2 * n,
            "render_batch": 4,
            "padding_images": (4 - 2 * n % 4) % 4,
        },
    )
    atomic_json(root / "timing/generation.json", timing)
