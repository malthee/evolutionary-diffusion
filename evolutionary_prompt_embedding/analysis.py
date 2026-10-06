"""Offline analysis adapters. Importing persistence never imports this module."""

import hashlib
import json
import tempfile
from dataclasses import asdict, dataclass, field
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from evolutionary_prompt_embedding.archive import _atomic_json

REPRESENTATIONS = ("token", "pooled", "combined_avg", "combined_append")


def load_records(archives, generations=None, islands=None, filters=None):
    """Return metadata only. Filters are column -> allowed values, applied pre-fit."""
    rows = []
    signature = None
    for archive in archives:
        spec = archive.manifest["tensor_spec"]
        identity = archive.manifest["run_metadata"].get("model", {})
        current = (spec, identity)
        if signature is not None and current != signature:
            raise ValueError(
                "Archives have incompatible tensor specifications or model identities"
            )
        signature = current
        for record in archive.iter_records():
            if generations is not None and record["generation"] not in generations:
                continue
            if islands is not None and record["island_id"] not in islands:
                continue
            row = dict(record)
            metadata = row.pop("metadata")
            # Keep reserved archive fields authoritative.
            row.update({k: v for k, v in metadata.items() if k not in row})
            row["metadata_json"] = json.dumps(metadata, sort_keys=True)
            fitness = row.pop("fitness")
            row["fitness_json"] = json.dumps(fitness)
            if isinstance(fitness, list):
                row["fitness"] = float(np.mean(fitness)) if fitness else None
                names = archive.manifest["run_metadata"].get("objective_names", [])
                for i, value in enumerate(fitness):
                    row[f"objective_{i}"] = value
                    if i < len(names):
                        row[f"objective_{i}_name"] = names[i]
            else:
                row["fitness"] = fitness
            row["archive_dir"] = str(archive.run_dir)
            if filters and any(
                row.get(key) not in allowed for key, allowed in filters.items()
            ):
                continue
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No committed records match the selection")
    if frame.record_id.duplicated().any():
        raise ValueError("Duplicate records: an archive was selected twice")
    return frame


def available_representations(archives):
    has_pooled = all(
        a.manifest["tensor_spec"]
        and "pooled_prompt_embeds" in a.manifest["tensor_spec"]
        for a in archives
    )
    return REPRESENTATIONS if has_pooled else ("token",)


def _representation(tensors, representation):
    token = tensors["prompt_embeds"].float().numpy()
    n = len(token)
    if representation == "token":
        return token.reshape(n, -1)
    if "pooled_prompt_embeds" not in tensors:
        raise ValueError("This representation requires pooled embeddings")
    pooled = tensors["pooled_prompt_embeds"].float().numpy().reshape(n, -1)
    if representation == "pooled":
        return pooled
    if representation == "combined_avg":
        # Candidate axis first; token axis is penultimate in the original tensor.
        if token.ndim < 3:
            raise ValueError("Token averaging requires a token and feature axis")
        token = token.mean(axis=-2)
    elif representation != "combined_append":
        raise ValueError(f"Unknown representation: {representation}")
    return np.concatenate([token.reshape(n, -1), pooled], axis=1)


@dataclass(frozen=True)
class ProjectionConfig:
    representation: str = "combined_append"
    algorithm: str = "umap"
    dimensions: int = 2
    pca_components: int = 64
    batch_size: int = 256
    seed: int = 42
    parameters: dict = field(default_factory=dict)


def _matrix_batches(archives, records, config):
    ids = set(records.record_id)
    for archive in archives:
        for batch, tensors in archive.iter_batches(config.batch_size, ids):
            matrix = _representation(tensors, config.representation)
            if not np.isfinite(matrix).all():
                raise ValueError("Non-finite embedding values cannot be projected")
            yield [r["record_id"] for r in batch], matrix


def _fit_batches(batches, minimum):
    """Hold the final batch so every IncrementalPCA partial_fit has >=k rows."""
    pending = None
    for _, matrix in batches:
        if pending is None:
            pending = matrix
        elif len(pending) >= minimum:
            # Retain up to one additional batch in case the final one is short.
            combined = np.concatenate([pending, matrix])
            if len(combined) >= 2 * minimum:
                yield combined[:-minimum]
                pending = combined[-minimum:]
            else:
                pending = combined
        else:
            pending = np.concatenate([pending, matrix])
    if pending is not None:
        yield pending


def compute_projection(archives, records, config=None, cache_dir=None, force=False):
    """Project every selected record. Only PCA scores are kept for the full run."""
    from sklearn.decomposition import IncrementalPCA
    from sklearn.neighbors import NearestNeighbors

    config = config or ProjectionConfig()
    if config.representation not in available_representations(archives):
        raise ValueError("Representation is unavailable for the selected archives")
    if config.algorithm not in ("pca", "umap", "tsne") or config.dimensions not in (
        2,
        3,
    ):
        raise ValueError("Choose pca/umap/tsne and 2 or 3 dimensions")
    if config.batch_size < 1 or config.pca_components < 1:
        raise ValueError("Batch size and PCA components must be positive")
    if len(records) < config.dimensions + 1:
        raise ValueError(
            f"{config.dimensions}D requires at least {config.dimensions + 1} records"
        )
    if config.algorithm in ("umap", "tsne") and len(records) < 4:
        raise ValueError("UMAP/t-SNE require at least four records; use PCA")
    packages = ["numpy", "pandas", "pyarrow", "scikit-learn", "safetensors", "torch"]
    if config.algorithm == "umap":
        packages += ["umap-learn", "numba", "pynndescent"]
    descriptor = {
        "config": asdict(config),
        "versions": {p: version(p) for p in packages},
        "analysis_version": 1,
        "records": records.record_id.tolist(),
        "archives": [a.manifest for a in archives],
    }
    key = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
    cache = Path(cache_dir or archives[0].run_dir / "analysis_cache")
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{key}.parquet"
    if target.exists() and not force:
        cached = pd.read_parquet(target)
        # Archives and their caches can be copied from Colab or moved locally.
        # Locations are runtime metadata, not part of the immutable projection.
        cached["archive_dir"] = cached.record_id.map(
            records.set_index("record_id").archive_dir
        )
        return cached
    _, first = next(_matrix_batches(archives, records, config))
    components = min(config.pca_components, len(records) - 1, first.shape[1])
    if components < config.dimensions:
        raise ValueError(
            "Too few input dimensions/PCA components for the requested projection"
        )
    del first
    pca = IncrementalPCA(n_components=components, batch_size=config.batch_size)
    for matrix in _fit_batches(_matrix_batches(archives, records, config), components):
        pca.partial_fit(matrix)
    scores = np.empty((len(records), components), dtype=np.float32)
    offsets = {rid: i for i, rid in enumerate(records.record_id)}
    seen = set()
    for ids, matrix in _matrix_batches(archives, records, config):
        indices = [offsets[rid] for rid in ids]
        scores[indices] = pca.transform(matrix)
        seen.update(ids)
    if seen != set(offsets):
        raise ValueError("Selected records are missing from the archives")
    if config.algorithm == "pca":
        if config.parameters:
            raise ValueError("PCA accepts no additional parameters")
        coordinates = scores[:, : config.dimensions]
    elif config.algorithm == "umap":
        from umap import UMAP

        params = {
            "n_neighbors": min(15, len(records) - 1),
            "min_dist": 0.1,
            "metric": "euclidean",
        }
        params.update(config.parameters)
        coordinates = UMAP(
            n_components=config.dimensions,
            random_state=config.seed,
            init="random",
            **params,
        ).fit_transform(scores)
    else:
        from sklearn.manifold import TSNE

        params = {
            "perplexity": min(30.0, (len(records) - 1) / 3),
            "init": "random",
            "learning_rate": "auto",
        }
        params.update(config.parameters)
        coordinates = TSNE(
            n_components=config.dimensions, random_state=config.seed, **params
        ).fit_transform(scores)
    count = min(11, len(records))
    nn = NearestNeighbors(n_neighbors=count).fit(scores)
    distances, indices = nn.kneighbors(scores)
    result = records.copy()
    for axis, values in zip(("x", "y", "z"), coordinates.T):
        result[axis] = values
    result["neighbors"] = [
        [records.iloc[int(j)].record_id for j in neighbors if j != i][:10]
        for i, neighbors in enumerate(indices)
    ]
    result["neighbor_distances"] = [
        [float(d) for j, d in zip(neighbors, ds) if j != i][:10]
        for i, (neighbors, ds) in enumerate(zip(indices, distances))
    ]
    result["neighbor_space"] = (
        f"Euclidean distance in {components}-component PCA analysis space"
    )
    result["projection_key"] = key
    with tempfile.NamedTemporaryFile(
        dir=cache, suffix=".parquet", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        result.to_parquet(temporary, index=False)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    _atomic_json(cache / f"{key}.json", descriptor)
    return result


def inspect_record(archives, record_id):
    """Return original CPU tensors alongside metadata; no implicit projection."""
    for archive in archives:
        try:
            return archive.read_record(record_id)
        except KeyError:
            continue
    raise KeyError(record_id)


def create_demo_archive():
    """Small model-free example covering images, missing files and island records."""
    import torch
    from PIL import Image

    from evolutionary.evolution_base import SolutionCandidate
    from evolutionary_prompt_embedding.archive import EmbeddingArchiveWriter
    from evolutionary_prompt_embedding.argument_types import PooledPromptEmbedData

    run_dir = Path(tempfile.mkdtemp(prefix="embedding-demo-"))
    writer = EmbeddingArchiveWriter(
        run_dir,
        {
            "model": {"id": "synthetic", "revision": "1"},
            "seeds": {"torch": 42},
            "configuration": {"demo": True},
        },
    )
    generator = torch.Generator().manual_seed(42)
    for generation in range(2):
        for island in range(2):
            population, image_paths, metadata = [], [], []
            for slot in range(6):
                args = PooledPromptEmbedData(
                    torch.randn(1, 4, 8, generator=generator),
                    torch.randn(1, 4, generator=generator),
                )
                candidate = SolutionCandidate(args, None)
                candidate.fitness = float(slot + generation / 10 + island / 100)
                population.append(candidate)
                paths = []
                if slot % 2:
                    for image_index in range(2 if slot == 3 else 1):
                        path = (
                            run_dir
                            / f"image-{generation}-{island}-{slot}-{image_index}.png"
                        )
                        Image.new(
                            "RGB", (64, 64), (slot * 40, generation * 100, island * 100)
                        ).save(path)
                        paths.append(path)
                if slot == 4:
                    paths.append(run_dir / "missing.png")
                image_paths.append(paths)
                metadata.append(
                    {
                        "prompt": f"Synthetic candidate {slot}",
                        "category": f"Island {island}",
                    }
                )
            writer.write_generation(
                population, generation, island, image_paths, metadata
            )
    print("Synthetic archive:", run_dir)
    return run_dir
