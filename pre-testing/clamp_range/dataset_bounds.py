"""Reproducible, text-only SDXL bounds experiments; no image generation."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CACHE = ROOT.parents[1] / ".cache" / "sdxl_bounds"
MODEL_ID = "stabilityai/sdxl-turbo"
MODEL_REVISION = "71153311d3dbb46851df1931d3ca6e939de83304"
SEED = 42
DATASETS = (
    "parti",
    "genai_bench",
    "t2i_compbench",
    "diffusiondb",
    "drawbench",
    "geneval",
)
SOURCES = {
    "parti": (
        "nateraw/parti-prompts",
        "944b156abfdad7627c3221b5ec4f6a6fb060a197",
        "apache-2.0",
    ),
    "genai_bench": (
        "BaiqiL/GenAI-Bench-1600",
        "e51e8a8070fadfbfe1f160436bf14e7664e87247",
        "apache-2.0 (declared on BaiqiL/GenAI-Bench)",
    ),
    "t2i_compbench": (
        "NinaKarine/t2i-compbench",
        "18f3f268acaf39c89f8a3afcc2b471243042c927",
        "mit",
    ),
    "diffusiondb": (
        "poloclub/diffusiondb",
        "fb620fbe49fa4420e0734bd9c0df11f51176b61f",
        "cc0-1.0",
    ),
    "drawbench": (
        "sayakpaul/drawbench",
        "ad794cba698c253ffdb6d18eff3fc87b004c1135",
        "apache-2.0",
    ),
    "geneval": (
        "djghosh13/geneval",
        "af4902f24d3ca90ebbb446dd9891a59e0f82725f",
        "mit (repository)",
    ),
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def fetch(dataset, filename):
    repo, revision, _ = SOURCES[dataset]
    path = CACHE / dataset / revision / filename
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        quoted = urllib.parse.quote(filename)
        if dataset == "geneval":
            url = f"https://raw.githubusercontent.com/{repo}/{revision}/{quoted}"
        else:
            url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{quoted}"
        temporary = path.with_suffix(path.suffix + ".tmp")
        with (
            urllib.request.urlopen(url, timeout=60) as response,
            temporary.open("wb") as output,
        ):
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(path)
    return path


def raw_records(dataset):
    if dataset in ("parti", "drawbench"):
        name = (
            "PartiPrompts.tsv"
            if dataset == "parti"
            else "DrawBench Prompts - Sheet1.csv"
        )
        with fetch(dataset, name).open(newline="") as handle:
            reader = csv.DictReader(
                handle, delimiter="\t" if dataset == "parti" else ","
            )
            for i, row in enumerate(reader):
                yield {
                    "id": str(i),
                    "prompt": row["Prompt" if dataset == "parti" else "Prompts"],
                    "category": row["Category"],
                }
    elif dataset == "geneval":
        for i, line in enumerate(
            fetch(dataset, "prompts/evaluation_metadata.jsonl").read_text().splitlines()
        ):
            row = json.loads(line)
            yield {"id": str(i), "prompt": row["prompt"], "category": row["tag"]}
    elif dataset == "genai_bench":
        prompts = json.loads(fetch(dataset, "genai_image.json").read_text())
        skills = json.loads(fetch(dataset, "genai_skills.json").read_text())
        tags = {}
        for tag, indices in skills.items():
            for index in indices:
                tags.setdefault(str(index).zfill(5), []).append(tag)
        for key, row in sorted(prompts.items()):
            yield {
                "id": key,
                "prompt": row["prompt"],
                "category": ",".join(tags.get(key, [])),
            }
    else:
        import pyarrow.parquet as pq

        if dataset == "diffusiondb":
            files = [("metadata.parquet", "user_prompt")]
            columns = ["prompt", "image_name"]
        else:
            categories = (
                "color",
                "shape",
                "texture",
                "spatial",
                "non_spatial",
                "complex",
            )
            files = []
            for category in categories:
                for split in ("train", "val"):
                    prefix = "spatial_" if category == "non_spatial" else ""
                    files.append(
                        (
                            f"{category}_{split}/{prefix}{split}-00000-of-00001.parquet",
                            category,
                        )
                    )
            columns = ["text"]
        for filename, category in files:
            row_id = 0
            for batch in pq.ParquetFile(fetch(dataset, filename)).iter_batches(
                columns=columns
            ):
                data = batch.to_pydict()
                for i, prompt in enumerate(data[columns[0]]):
                    identifier = (
                        data["image_name"][i]
                        if dataset == "diffusiondb"
                        else f"{filename}:{row_id}"
                    )
                    yield {"id": identifier, "prompt": prompt, "category": category}
                    row_id += 1


def clean_records(records):
    """Deduplicate exact text, retaining original text and first source identity."""
    seen, cleaned = set(), []
    counts = {"raw_rows": 0, "blank_rows": 0, "duplicate_rows": 0}
    for row in records:
        counts["raw_rows"] += 1
        prompt = row["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            counts["blank_rows"] += 1
        elif prompt in seen:
            counts["duplicate_rows"] += 1
        else:
            seen.add(prompt)
            cleaned.append(row)
    counts["unique_prompts"] = len(cleaned)
    if not cleaned:
        raise ValueError("Dataset contains no usable prompts")
    return cleaned, counts


def select_records(records, count, seed=SEED):
    """Uniform random ordering; prefixes give reproducible nested samples."""
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    return [records[i] for i in indices[: min(count, len(records))]]


class Bounds:
    def __init__(self):
        self.minimum = None
        self.maximum = None
        self.count = 0

    def update(self, tensor):
        tensor = tensor.detach().to(device="cpu", dtype=torch.float32)
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("Embedding contains non-finite values")
        if self.minimum is None:
            self.minimum, self.maximum = tensor.clone(), tensor.clone()
        else:
            if tensor.shape != self.minimum.shape:
                raise ValueError("Embedding shape changed during evaluation")
            torch.minimum(self.minimum, tensor, out=self.minimum)
            torch.maximum(self.maximum, tensor, out=self.maximum)
        self.count += 1

    def summary(self):
        if not self.count:
            raise ValueError("Cannot summarize empty bounds")
        span = self.maximum - self.minimum
        return {
            "shape": list(span.shape),
            "global_min": self.minimum.min().item(),
            "global_max": self.maximum.max().item(),
            "span_sum": span.sum().item(),
            "span_mean": span.mean().item(),
            "constant_coordinates": int((span == 0).sum()),
        }

    def tensors(self):
        if not self.count:
            raise ValueError("Cannot save empty bounds")
        return {"min": self.minimum, "max": self.maximum}


class TextEncoder:
    """The same text-encoder outputs as SDXLPromptEmbeddingImageCreator."""

    def __init__(self, device=None):
        from transformers import (
            CLIPTextModel,
            CLIPTextModelWithProjection,
            CLIPTokenizer,
        )

        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizers = [
            CLIPTokenizer.from_pretrained(
                MODEL_ID, subfolder=folder, revision=MODEL_REVISION
            )
            for folder in ("tokenizer", "tokenizer_2")
        ]
        self.encoders = [
            cls.from_pretrained(
                MODEL_ID,
                subfolder=folder,
                revision=MODEL_REVISION,
                variant="fp16",
                dtype=torch.float16,
            )
            .eval()
            .to(self.device)
            for cls, folder in (
                (CLIPTextModel, "text_encoder"),
                (CLIPTextModelWithProjection, "text_encoder_2"),
            )
        ]

    def synchronize(self):
        if self.device == "mps":
            torch.mps.synchronize()

    @torch.no_grad()
    def encode(self, prompt):
        hidden, truncated, lengths = [], [], []
        for tokenizer, encoder in zip(self.tokenizers, self.encoders):
            full_ids = tokenizer(prompt, truncation=False, verbose=False).input_ids
            lengths.append(len(full_ids))
            truncated.append(len(full_ids) > tokenizer.model_max_length)
            ids = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(self.device)
            output = encoder(ids, output_hidden_states=True)
            hidden.append(output.hidden_states[-2])
        token = torch.cat(hidden, dim=-1).cpu()
        pooled = output[0].cpu()
        self.synchronize()
        return token, pooled, truncated, lengths

    def verify_creator_parity(self):
        # Avoid loading the unused VAE/UNet while calling the production method.
        import contextlib
        import io

        from evolutionary_prompt_embedding.image_creation import (
            SDXLPromptEmbeddingImageCreator,
        )

        pipeline = SimpleNamespace(
            tokenizer=self.tokenizers[0],
            tokenizer_2=self.tokenizers[1],
            text_encoder=self.encoders[0],
            text_encoder_2=self.encoders[1],
            device=torch.device(self.device),
        )
        receiver = SimpleNamespace(_pipeline=pipeline)
        errors = []
        for prompt in (
            "a red car",
            "A painting of a church beside a river at sunset.",
            "a horse and a blue car " * 30,
        ):
            token, pooled, _, _ = self.encode(prompt)
            with contextlib.redirect_stdout(io.StringIO()):
                original = SDXLPromptEmbeddingImageCreator.arguments_from_prompt(
                    receiver, prompt
                )
            self.synchronize()
            errors.append(
                {
                    "token_max_abs_error": (
                        token.float() - original.prompt_embeds.cpu().float()
                    )
                    .abs()
                    .max()
                    .item(),
                    "pooled_max_abs_error": (
                        pooled.float() - original.pooled_prompt_embeds.cpu().float()
                    )
                    .abs()
                    .max()
                    .item(),
                }
            )
            torch.testing.assert_close(
                token, original.prompt_embeds.cpu(), rtol=1e-3, atol=1e-3
            )
            torch.testing.assert_close(
                pooled, original.pooled_prompt_embeds.cpu(), rtol=1e-3, atol=1e-3
            )
        return errors


def planned_count(available, elapsed, timings, budget_seconds=3000):
    if not timings or any(not math.isfinite(value) or value <= 0 for value in timings):
        raise ValueError("Pilot timings must be finite and positive")
    conservative_seconds = float(np.percentile(timings, 95)) * 1.25
    return min(
        available, max(0, math.floor((budget_seconds - elapsed) / conservative_seconds))
    )


def save_checkpoint(directory, bounds, metadata, growth):
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "bounds.pt.tmp"
    torch.save({key: value.tensors() for key, value in bounds.items()}, temporary)
    temporary.replace(directory / "bounds.pt")
    write_json(
        directory / "summary.json",
        {
            **metadata,
            "embeddings": {key: value.summary() for key, value in bounds.items()},
            "growth": growth,
        },
    )


def load_result(dataset):
    directory = RESULTS / dataset
    summary = json.loads((directory / "summary.json").read_text())
    tensors = torch.load(directory / "bounds.pt", map_location="cpu", weights_only=True)
    return summary, tensors


def run_dataset(dataset, *, working_seconds=3000, stop_seconds=3300, device=None):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    started = time.monotonic()
    directory = RESULTS / dataset
    directory.mkdir(parents=True, exist_ok=True)
    records, counts = clean_records(raw_records(dataset))
    parti_prompts = {row["prompt"] for row in raw_records("parti")}
    encoder = TextEncoder(device)
    parity = encoder.verify_creator_parity()
    probe = select_records(records, min(35, len(records)))
    timings = []
    for i, row in enumerate(probe):
        tick = time.monotonic()
        encoder.encode(row["prompt"])
        if i >= 3:
            timings.append(time.monotonic() - tick)
    count = planned_count(
        len(records), time.monotonic() - started, timings, working_seconds
    )
    if not count:
        raise TimeoutError("No encoding budget remains after preparation")
    selected = select_records(records, count)
    del records
    selected_path = directory / "selected_prompts.jsonl"
    with selected_path.open("w") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest_hash = hashlib.sha256(selected_path.read_bytes()).hexdigest()
    repo, revision, license_name = SOURCES[dataset]
    metadata = {
        "dataset": dataset,
        "source": repo,
        "source_revision": revision,
        "declared_license": license_name,
        **counts,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "device": encoder.device,
        "encoding_dtype": "float16",
        "aggregation_dtype": "float32",
        "seed": SEED,
        "planned_prompts": count,
        "encoded_prompts": 0,
        "status": "running",
        "selected_prompts_sha256": manifest_hash,
        "pilot_mean_seconds": float(np.mean(timings)),
        "pilot_p95_seconds": float(np.percentile(timings, 95)),
        "parity": parity,
        "working_seconds": working_seconds,
        "stop_seconds": stop_seconds,
        "truncation_counts": [0, 0],
        "max_token_lengths": [0, 0],
        "parti_overlap_selected": sum(
            row["prompt"] in parti_prompts for row in selected
        ),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "diffusers", "datasets", "numpy")
        },
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(
        f"{dataset}: {encoder.device}, {count}/{counts['unique_prompts']} prompts; pilot {metadata['pilot_mean_seconds']:.4f}s/prompt",
        flush=True,
    )
    bounds = {"token": Bounds(), "pooled": Bounds()}
    matched = {"token": Bounds(), "pooled": Bounds()}
    cache = {}
    if dataset == "parti":
        for key, shape in (
            ("token", (count, 1, 77, 2048)),
            ("pooled", (count, 1, 1280)),
        ):
            cache[key] = np.lib.format.open_memmap(
                directory / f"embeddings_{key}.npy",
                mode="w+",
                dtype=np.float16,
                shape=shape,
            )
    growth = []
    try:
        for i, row in enumerate(selected):
            if time.monotonic() - started >= stop_seconds:
                metadata["status"] = "partial_timeout"
                break
            token, pooled, truncated, lengths = encoder.encode(row["prompt"])
            for key, tensor in (("token", token), ("pooled", pooled)):
                bounds[key].update(tensor)
                if i < 1632:
                    matched[key].update(tensor)
                if key in cache:
                    cache[key][i] = tensor.numpy()
            metadata["encoded_prompts"] = i + 1
            metadata["truncation_counts"] = [
                a + int(b) for a, b in zip(metadata["truncation_counts"], truncated)
            ]
            metadata["max_token_lengths"] = [
                max(a, b) for a, b in zip(metadata["max_token_lengths"], lengths)
            ]
            if (i + 1) % 100 == 0 or i + 1 == count:
                metadata["elapsed_seconds"] = time.monotonic() - started
                growth.append(
                    {
                        "n": i + 1,
                        **{key: value.summary() for key, value in bounds.items()},
                    }
                )
                save_checkpoint(directory, bounds, metadata, growth)
                print(
                    f"{dataset}: {i + 1}/{count}, {metadata['elapsed_seconds'] / 60:.1f} minutes",
                    flush=True,
                )
        else:
            metadata["status"] = (
                "full" if count == counts["unique_prompts"] else "sampled"
            )
    except Exception as error:
        metadata["status"] = "failed"
        metadata["error"] = repr(error)
        raise
    finally:
        for array in cache.values():
            array.flush()
        metadata["elapsed_seconds"] = time.monotonic() - started
        if all(value.count for value in bounds.values()):
            save_checkpoint(directory, bounds, metadata, growth)
            torch.save(
                {key: value.tensors() for key, value in matched.items()},
                directory / "matched_bounds.pt",
            )
        else:
            write_json(directory / "summary.json", metadata)
    return metadata


def compare_bounds(candidate, baseline, mask=None):
    low, high = candidate["min"].float(), candidate["max"].float()
    base_low, base_high = baseline["min"].float(), baseline["max"].float()
    if low.shape != base_low.shape or high.shape != base_high.shape:
        raise ValueError("Cannot compare incompatible embedding shapes")
    if mask is not None:
        low, high, base_low, base_high = (
            x[mask] for x in (low, high, base_low, base_high)
        )
    span, base_span = high - low, base_high - base_low
    delta_low, delta_high, delta_span = (
        low - base_low,
        high - base_high,
        span - base_span,
    )
    denominator = base_span.sum().item()
    return {
        "coordinates": low.numel(),
        "global_min": low.min().item(),
        "global_max": high.max().item(),
        "span_sum": span.sum().item(),
        "parti_span_sum": denominator,
        "span_change_percent": 100 * (span.sum().item() / denominator - 1)
        if denominator
        else None,
        "lower_bound_mean_abs_change": delta_low.abs().mean().item(),
        "upper_bound_mean_abs_change": delta_high.abs().mean().item(),
        "span_mean_abs_change": delta_span.abs().mean().item(),
        "below_parti_percent": 100 * (delta_low < 0).float().mean().item(),
        "above_parti_percent": 100 * (delta_high > 0).float().mean().item(),
        "either_outside_percent": 100
        * ((delta_low < 0) | (delta_high > 0)).float().mean().item(),
        "largest_lower_extension": torch.clamp(-delta_low, min=0).max().item(),
        "largest_upper_extension": torch.clamp(delta_high, min=0).max().item(),
    }


def parti_matched_bounds(n):
    summary, _ = load_result("parti")
    if n > summary["encoded_prompts"]:
        raise ValueError(
            "Parti baseline has fewer encoded prompts than the matched comparison"
        )
    result = {}
    for key in ("token", "pooled"):
        array = np.load(RESULTS / "parti" / f"embeddings_{key}.npy", mmap_mode="r")
        # Parti is already in the seeded random order used for nested samples.
        selected = np.asarray(array[:n], dtype=np.float32)
        result[key] = {
            "min": torch.from_numpy(selected.min(axis=0)),
            "max": torch.from_numpy(selected.max(axis=0)),
        }
    return result


def build_comparison():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    baseline_summary, baseline = load_result("parti")
    if baseline_summary["status"] != "full":
        raise ValueError("A fully evaluated Parti baseline is required")
    rows, manifest = [], {}
    for dataset in DATASETS:
        summary, tensors = load_result(dataset)
        if summary["status"] not in ("full", "sampled", "partial_timeout"):
            raise ValueError(f"{dataset} has no successful evaluation")
        for field in (
            "model_revision",
            "device",
            "encoding_dtype",
            "aggregation_dtype",
            "versions",
        ):
            if summary[field] != baseline_summary[field]:
                raise ValueError(f"Incompatible baseline: {dataset} differs in {field}")
        n = min(1632, summary["encoded_prompts"])
        matched_base = parti_matched_bounds(n)
        matched_candidate = torch.load(
            RESULTS / dataset / "matched_bounds.pt",
            weights_only=True,
            map_location="cpu",
        )
        manifest[dataset] = summary
        differences = {}
        for mode, current, reference in (
            ("all", tensors, baseline),
            ("matched", matched_candidate, matched_base),
        ):
            differences[mode] = {
                key: {
                    "lower": current[key]["min"] - reference[key]["min"],
                    "upper": current[key]["max"] - reference[key]["max"],
                    "span": (current[key]["max"] - current[key]["min"])
                    - (reference[key]["max"] - reference[key]["min"]),
                }
                for key in ("token", "pooled")
            }
            for key in ("token", "pooled"):
                rows.append(
                    {
                        "dataset": dataset,
                        "comparison": mode,
                        "embedding": key,
                        "region": "all",
                        "n": summary["encoded_prompts"] if mode == "all" else n,
                        "parti_n": baseline_summary["encoded_prompts"]
                        if mode == "all"
                        else n,
                        **compare_bounds(current[key], reference[key]),
                    }
                )
                variable = baseline[key]["max"] > baseline[key]["min"]
                rows.append(
                    {
                        "dataset": dataset,
                        "comparison": mode,
                        "embedding": key,
                        "region": "parti_variable",
                        "n": summary["encoded_prompts"] if mode == "all" else n,
                        "parti_n": baseline_summary["encoded_prompts"]
                        if mode == "all"
                        else n,
                        **compare_bounds(current[key], reference[key], variable),
                    }
                )
                if key == "token":
                    for name, section in (
                        ("encoder_1", slice(0, 768)),
                        ("encoder_2", slice(768, 2048)),
                    ):
                        a = {k: v[..., section] for k, v in current[key].items()}
                        b = {k: v[..., section] for k, v in reference[key].items()}
                        rows.append(
                            {
                                "dataset": dataset,
                                "comparison": mode,
                                "embedding": key,
                                "region": name,
                                "n": summary["encoded_prompts"] if mode == "all" else n,
                                "parti_n": baseline_summary["encoded_prompts"]
                                if mode == "all"
                                else n,
                                **compare_bounds(a, b),
                            }
                        )
        directory = RESULTS / dataset
        torch.save(differences, directory / "bounds_differences.pt")
        fig, axes = plt.subplots(3, 1, figsize=(14, 9), constrained_layout=True)
        for ax, (title, values) in zip(
            axes,
            (
                (
                    "Lower bound minus Parti",
                    tensors["token"]["min"] - baseline["token"]["min"],
                ),
                (
                    "Upper bound minus Parti",
                    tensors["token"]["max"] - baseline["token"]["max"],
                ),
                (
                    "Span minus Parti",
                    (tensors["token"]["max"] - tensors["token"]["min"])
                    - (baseline["token"]["max"] - baseline["token"]["min"]),
                ),
            ),
        ):
            limit = max(float(values.abs().max()), 1e-6)
            plot = ax.imshow(
                values[0].numpy(),
                aspect="auto",
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
            )
            ax.axvline(767.5, color="black", linewidth=0.8)
            ax.set(
                title=title,
                xlabel="Embedding dimension (encoder boundary at 768)",
                ylabel="Token position",
            )
            fig.colorbar(plot, ax=ax)
        fig.suptitle(f"{dataset}: full/budgeted bounds versus all Parti prompts")
        fig.savefig(directory / "bounds_difference.png", dpi=120)
        plt.close(fig)
        fig, axes = plt.subplots(3, 1, figsize=(14, 7), constrained_layout=True)
        for ax, (name, values) in zip(axes, differences["all"]["pooled"].items()):
            ax.plot(values[0].numpy(), linewidth=0.6)
            ax.axhline(0, color="black", linewidth=0.6)
            ax.set(
                title=f"Pooled {name} difference versus Parti",
                xlabel="Embedding coordinate",
                ylabel="Difference",
            )
        fig.suptitle(f"{dataset}: pooled bounds at every coordinate")
        fig.savefig(directory / "pooled_bounds_difference.png", dpi=120)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        for ax, key in zip(axes, ("token", "pooled")):
            growth = summary["growth"]
            ax.plot([x["n"] for x in growth], [x[key]["span_sum"] for x in growth])
            ax.axhline(
                baseline_summary["embeddings"][key]["span_sum"],
                linestyle="--",
                color="black",
                label="Full Parti",
            )
            ax.set(
                title=key,
                xlabel="Encoded unique prompts",
                ylabel="Sum of coordinate spans",
            )
            ax.legend()
        fig.savefig(directory / "bounds_growth.png", dpi=120)
        plt.close(fig)
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "comparison.csv", index=False)
    write_json(RESULTS / "comparison.json", rows)
    write_json(RESULTS / "run_manifest.json", manifest)
    legacy = {}
    for key, suffix in (("token", ""), ("pooled", "_pooled")):
        packaged = {
            bound: torch.load(
                ROOT.parents[1]
                / "evolutionary_prompt_embedding"
                / "tensors"
                / f"sdxl_turbo_{bound}_tensor{suffix}.pt",
                map_location="cpu",
                weights_only=True,
            ).float()
            for bound in ("min", "max")
        }
        legacy[key] = compare_bounds(baseline[key], packaged)
    write_json(RESULTS / "parti_packaged_drift.json", legacy)
    return frame, manifest, legacy


def write_report(frame, manifest, legacy):
    """Write a durable, numerical readout from completed evaluation artifacts."""
    lines = [
        "# SDXL-Turbo dataset bounds comparison",
        "",
        "## Experiment",
        "",
        f"Model: `{MODEL_ID}` at `{MODEL_REVISION}`. Text-only encoding; float16 model outputs, float32 reductions. Sampling seed: {SEED}. All results use the same device and package versions.",
        "",
        "These are observed coordinate-wise intervals, not guarantees of the embedding space's full extent. Larger samples have more opportunities to widen extrema. Equal-count comparisons use one deterministic random subset, without uncertainty estimates.",
        "",
        "## Dataset coverage and timing",
        "",
        "| Dataset | Status | Raw rows | Unique | Encoded | Minutes | Device | Truncated tokenizer 1 / 2 | Parti overlap selected |",
        "|---|---|---:|---:|---:|---:|---|---|---:|",
    ]
    for dataset, row in manifest.items():
        n = row["encoded_prompts"]
        truncation = " / ".join(
            f"{x} ({100 * x / n:.2f}%)" for x in row["truncation_counts"]
        )
        lines.append(
            f"| {dataset} | {row['status']} | {row['raw_rows']:,} | {row['unique_prompts']:,} | {n:,} | {row['elapsed_seconds'] / 60:.2f} | {row['device']} | {truncation} | {row['parti_overlap_selected']:,} |"
        )
    for mode, title in (
        ("all", "Full or budgeted samples versus all 1,632 Parti prompts"),
        ("matched", "Equal-count samples versus Parti"),
    ):
        lines += [
            "",
            f"## {title}",
            "",
            "| Dataset | Embedding | Prompts / Parti | Min | Max | Sum of spans | Span change | Positions outside Parti |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
        selected = frame[(frame.comparison == mode) & (frame.region == "all")]
        for row in selected.to_dict("records"):
            change = (
                "undefined"
                if row["span_change_percent"] is None
                else f"{row['span_change_percent']:+.2f}%"
            )
            lines.append(
                f"| {row['dataset']} | {row['embedding']} | {row['n']} / {row['parti_n']} | {row['global_min']:.6g} | {row['global_max']:.6g} | {row['span_sum']:.3f} | {change} | {row['either_outside_percent']:.2f}% |"
            )
    lines += ["", "## Observed differences", ""]
    for dataset in DATASETS[1:]:
        selected = frame[(frame.dataset == dataset) & (frame.region == "all")]
        values = {
            (r["comparison"], r["embedding"]): r for r in selected.to_dict("records")
        }
        token, pooled = values[("all", "token")], values[("all", "pooled")]
        token_matched, pooled_matched = (
            values[("matched", "token")],
            values[("matched", "pooled")],
        )
        lines.append(
            f"- **{dataset}**: token span sum changes by {token['span_change_percent']:+.2f}% and pooled span sum by {pooled['span_change_percent']:+.2f}% versus full Parti. At equal counts the changes are {token_matched['span_change_percent']:+.2f}% and {pooled_matched['span_change_percent']:+.2f}%. The full/budgeted dataset extends outside Parti at {token['either_outside_percent']:.2f}% of token positions and {pooled['either_outside_percent']:.2f}% of pooled positions."
        )
    lines += [
        "",
        "## Constant coordinates and encoder blocks",
        "",
        "The original packaged Parti token bounds have 2,048 constant coordinates, the first token's complete embedding row. Their extreme values dominate overall min/max. The CSV includes all coordinates, positions variable in the fresh full Parti baseline, and SDXL's separate 768- and 1,280-dimensional encoder blocks. Constant positions remain included in the main tables.",
        "",
        "## Fresh Parti versus packaged tensors",
        "",
        "The packaged baseline was produced in an older software environment. The primary dataset comparisons use fresh Parti, so environment drift is not mistaken for a dataset effect.",
        "",
    ]
    for key, row in legacy.items():
        lines.append(
            f"- {key}: lower-bound mean absolute difference {row['lower_bound_mean_abs_change']:.6g}; upper-bound difference {row['upper_bound_mean_abs_change']:.6g}; summed span change {row['span_change_percent']:+.4f}%."
        )
    lines += ["", "## Sources and reproducibility", ""]
    for dataset, row in manifest.items():
        repo = row["source"]
        url = (
            f"https://github.com/{repo}"
            if dataset == "geneval"
            else f"https://huggingface.co/datasets/{repo}"
        )
        lines.append(
            f"- [{dataset}]({url}), revision `{row['source_revision']}`; declared license: {row['declared_license']}. Sample SHA-256: `{row['selected_prompts_sha256']}`."
        )
    lines += [
        "",
        "Exact selected prompt IDs and text are saved locally in each dataset's `selected_prompts.jsonl`. The Parti embedding cache is local and excluded from Git. The manifest records package versions, throughput, parity checks, deadlines and truncation counts. Re-running a dataset notebook replaces that dataset's artifacts.",
        "",
        "## Interpretation limits",
        "",
        "- Token padding and the 77-token truncation behavior match the existing production representation. Long prompts are therefore comparisons of their encoded prefixes, not their entire text.",
        "- DiffusionDB includes real user prompting styles and potentially multiple languages; no language or content filter is applied. T2I-CompBench uses all original-category train/validation prompt rows because this study does not train a model.",
        "- Duplicates are removed within each dataset. Exact overlap with Parti is reported; overlap is retained to evaluate each dataset as distributed.",
        "- A wider interval does not establish better semantic coverage, image quality, or a better evolutionary clamp. No production clamp tensors were changed.",
        "",
    ]
    (RESULTS / "REPORT.md").write_text("\n".join(lines))
