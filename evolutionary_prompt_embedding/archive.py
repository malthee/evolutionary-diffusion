"""Lossless, sharded embedding snapshots, independent of visualization tools.

One writer at a time per run directory. Writers keep paths/configuration only,
so callbacks holding them remain pickleable. A manifest entry is the commit
point: readers ignore temporary files and unreferenced completed files.
"""

import hashlib
import json
import os
import uuid
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SCHEMA_VERSION = 1
DEFAULT_SHARD_BYTES = 256 * 1024 * 1024


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _json_text(value):
    return json.dumps(value, default=_json_value, allow_nan=False, sort_keys=True)


def _atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        stream.write(_json_text(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def create_run_directory(parent="embedding_runs", prefix="run"):
    """Create a unique run root shared by images, embeddings and the manifest."""
    path = Path(parent).expanduser().resolve() / f"{prefix}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _tensor_spec(arguments):
    tensors = {"prompt_embeds": arguments.prompt_embeds}
    pooled = getattr(arguments, "pooled_prompt_embeds", None)
    if pooled is not None:
        tensors["pooled_prompt_embeds"] = pooled
    for tensor in tensors.values():
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            raise ValueError("Embeddings must be dense torch tensors")
        if not tensor.is_floating_point() or tensor.numel() == 0:
            raise ValueError("Embeddings must be nonempty floating-point tensors")
    spec = {
        key: {"shape": list(t.shape), "dtype": str(t.dtype)}
        for key, t in tensors.items()
    }
    return tensors, spec


class EmbeddingArchiveWriter:
    def __init__(self, run_dir, run_metadata=None, max_shard_bytes=DEFAULT_SHARD_BYTES):
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.run_dir / "manifest.json"
        if self.manifest_path.exists():
            manifest = EmbeddingArchiveReader(self.run_dir).manifest
            if (
                run_metadata is not None
                and json.loads(_json_text(run_metadata)) != manifest["run_metadata"]
            ):
                raise ValueError("Existing run metadata differs")
        else:
            if max_shard_bytes <= 4096 or max_shard_bytes > DEFAULT_SHARD_BYTES:
                raise ValueError("Shard limit must be >4096 and <=256 MiB")
            _atomic_json(
                self.manifest_path,
                {
                    "format": "evolutionary-embedding-archive",
                    "schema_version": SCHEMA_VERSION,
                    "run_id": uuid.uuid4().hex,
                    "run_metadata": run_metadata or {},
                    "max_shard_bytes": max_shard_bytes,
                    "tensor_spec": None,
                    "snapshots": [],
                    "shards": [],
                },
            )

    def check_snapshot_available(self, generation, island_id=None, *, snapshot_id=None):
        """Preflight before saving accompanying images to avoid overwriting them."""
        manifest = EmbeddingArchiveReader(self.run_dir).manifest
        if any(
            s["generation"] == generation
            and s["island_id"] == island_id
            and s.get("snapshot_id") == snapshot_id
            for s in manifest["snapshots"]
        ):
            raise ValueError(
                f"Duplicate snapshot: generation={generation}, island={island_id}"
            )

    def write_generation(
        self,
        population,
        generation,
        island_id=None,
        image_paths=None,
        metadata=None,
        *,
        snapshot_id=None,
    ):
        """Persist a snapshot; optional metadata is one JSON mapping per candidate.

        Image paths must be inside run_dir. Missing images are permitted. Once
        any snapshot is started it cannot be overwritten, including after failure.
        Earlier committed shards remain readable if a later shard fails.
        """
        if (
            not isinstance(generation, int)
            or isinstance(generation, bool)
            or generation < 0
        ):
            raise ValueError("Generation must be a nonnegative integer")
        if island_id is not None and (
            not isinstance(island_id, int) or isinstance(island_id, bool)
        ):
            raise ValueError("Island ID must be an integer or None")
        count = len(population)
        images = [[] for _ in population] if image_paths is None else image_paths
        labels = [{} for _ in population] if metadata is None else metadata
        if len(images) != count or len(labels) != count:
            raise ValueError("Images/metadata must have one entry per candidate")
        manifest = EmbeddingArchiveReader(self.run_dir).manifest
        if any(
            s["generation"] == generation
            and s["island_id"] == island_id
            and s.get("snapshot_id") == snapshot_id
            for s in manifest["snapshots"]
        ):
            raise ValueError(
                f"Duplicate snapshot: generation={generation}, island={island_id}"
            )
        if snapshot_id is not None and (
            not isinstance(snapshot_id, str)
            or not snapshot_id
            or len(snapshot_id) > 128
        ):
            raise ValueError(
                "snapshot_id must be a nonempty string of at most 128 characters"
            )
        records = []
        spec = None
        record_bytes = 0
        for index, candidate in enumerate(population):
            tensors, candidate_spec = _tensor_spec(candidate.arguments)
            if spec is None:
                spec = candidate_spec
                record_bytes = sum(
                    t.numel() * t.element_size() for t in tensors.values()
                )
            if candidate_spec != spec or (
                manifest["tensor_spec"] is not None and spec != manifest["tensor_spec"]
            ):
                raise ValueError(
                    "Incompatible tensor shapes, dtypes or pooled availability in run"
                )
            if not isinstance(labels[index], dict):
                raise TypeError("Candidate metadata must be a mapping")
            relative_images = []
            for image in images[index]:
                path = Path(image).expanduser()
                path = (
                    path.resolve()
                    if path.is_absolute()
                    else (self.run_dir / path).resolve()
                )
                try:
                    relative_images.append(path.relative_to(self.run_dir).as_posix())
                except ValueError as exc:
                    raise ValueError(
                        "Images must reside inside the run directory"
                    ) from exc
            record = {
                "record_id": f"{manifest['run_id']}:{island_id}:{generation}:{index}"
                + (f":{snapshot_id}" if snapshot_id is not None else ""),
                "run_id": manifest["run_id"],
                "generation": generation,
                "candidate_slot": index,
                "island_id": island_id,
                "fitness": candidate.fitness,
                "image_paths": relative_images,
                "metadata": labels[index],
            }
            # Validate before committing any files; retain full numeric precision.
            records.append(json.loads(_json_text(record)))
        capacity = max(1, (manifest["max_shard_bytes"] - 4096) // max(1, record_bytes))
        if record_bytes + 4096 > manifest["max_shard_bytes"]:
            raise ValueError("A candidate exceeds the shard limit")
        manifest["tensor_spec"] = spec or manifest["tensor_spec"]
        snapshot = {
            "generation": generation,
            "island_id": island_id,
            "expected_count": count,
            "committed_count": 0,
            "status": "writing",
        }
        if snapshot_id is not None:
            snapshot["snapshot_id"] = snapshot_id
        manifest["snapshots"].append(snapshot)
        _atomic_json(self.manifest_path, manifest)
        folder = self.run_dir / "embeddings"
        folder.mkdir(exist_ok=True)
        for start in range(0, count, capacity):
            stop = min(count, start + capacity)
            first, _ = _tensor_spec(population[start].arguments)
            # Preallocate once; avoid retaining candidate copies plus a stacked copy.
            buffers = {
                key: torch.empty(
                    (stop - start, *tensor.shape), dtype=tensor.dtype, device="cpu"
                )
                for key, tensor in first.items()
            }
            for row, candidate in enumerate(population[start:stop]):
                tensors, _ = _tensor_spec(candidate.arguments)
                for key, tensor in tensors.items():
                    buffers[key][row].copy_(tensor.detach())
            name = f"g{generation}-i{island_id}-{uuid.uuid4().hex}"
            tensor_path = folder / f"{name}.safetensors"
            temporary = tensor_path.with_suffix(".tmp")
            save_file(buffers, temporary)
            del buffers
            if temporary.stat().st_size > manifest["max_shard_bytes"]:
                raise ValueError("Serialized shard exceeds the limit")
            os.replace(temporary, tensor_path)
            record_path = folder / f"{name}.jsonl"
            temporary = record_path.with_suffix(".tmp")
            with open(temporary, "w", encoding="utf-8") as stream:
                for row, record in enumerate(records[start:stop]):
                    record.update(
                        tensor_file=tensor_path.relative_to(self.run_dir).as_posix(),
                        tensor_row=row,
                    )
                    stream.write(_json_text(record) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, record_path)
            manifest["shards"].append(
                {
                    "tensors": tensor_path.relative_to(self.run_dir).as_posix(),
                    "records": record_path.relative_to(self.run_dir).as_posix(),
                    "count": stop - start,
                    "tensor_sha256": sha256_file(tensor_path),
                    "records_sha256": sha256_file(record_path),
                }
            )
            snapshot["committed_count"] = stop
            _atomic_json(self.manifest_path, manifest)
        snapshot["status"] = "complete"
        _atomic_json(self.manifest_path, manifest)


class EmbeddingArchiveReader:
    def __init__(self, run_dir, verify=True):
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.manifest = json.loads(
            (self.run_dir / "manifest.json").read_text(encoding="utf-8")
        )
        if (
            self.manifest.get("format") != "evolutionary-embedding-archive"
            or self.manifest.get("schema_version") != SCHEMA_VERSION
        ):
            raise ValueError("Unsupported embedding archive format/version")
        self.verify = verify

    def resolve_path(self, relative):
        path = (self.run_dir / relative).resolve()
        if not path.is_relative_to(self.run_dir):
            raise ValueError("Archive path escapes the run directory")
        return path

    def iter_records(self):
        for shard in self.manifest["shards"]:
            path = self.resolve_path(shard["records"])
            if self.verify and sha256_file(path) != shard["records_sha256"]:
                raise ValueError(f"Record checksum mismatch: {path}")
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            if len(records) != shard["count"]:
                raise ValueError("Shard record count differs")
            yield from records

    def iter_batches(self, batch_size=256, record_ids=None):
        """Yield (records, tensor mapping) on CPU, never loading a whole archive."""
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        selected = None if record_ids is None else set(record_ids)
        records = list(self.iter_records())
        by_file = {}
        for record in records:
            if selected is None or record["record_id"] in selected:
                by_file.setdefault(record["tensor_file"], []).append(record)
        known = {shard["tensors"]: shard for shard in self.manifest["shards"]}
        for filename, rows in by_file.items():
            if filename not in known:
                raise ValueError("Record refers to an uncommitted tensor shard")
            path = self.resolve_path(filename)
            if self.verify and sha256_file(path) != known[filename]["tensor_sha256"]:
                raise ValueError(f"Tensor checksum mismatch: {path}")
            with safe_open(path, framework="pt", device="cpu") as handle:
                spec = self.manifest["tensor_spec"]
                if set(handle.keys()) != set(spec):
                    raise ValueError("Tensor keys differ from the manifest")
                for key in handle.keys():  # noqa: SIM118 -- safe_open is not iterable
                    if handle.get_slice(key).get_shape() != [
                        known[filename]["count"],
                        *spec[key]["shape"],
                    ]:
                        raise ValueError("Tensor shape differs from the manifest")
                for offset in range(0, len(rows), batch_size):
                    batch = rows[offset : offset + batch_size]
                    if any(
                        not isinstance(r["tensor_row"], int)
                        or not 0 <= r["tensor_row"] < known[filename]["count"]
                        for r in batch
                    ):
                        raise ValueError("Invalid tensor row reference")
                    tensors = {}
                    for key in handle.keys():  # noqa: SIM118 -- safe_open is not a dict/iterable
                        tensor_slice = handle.get_slice(key)
                        tensors[key] = torch.cat(
                            [
                                tensor_slice[r["tensor_row"] : r["tensor_row"] + 1]
                                for r in batch
                            ]
                        )
                        if str(tensors[key].dtype) != spec[key]["dtype"]:
                            raise ValueError("Tensor dtype differs from the manifest")
                    yield batch, tensors

    def read_record(self, record_id):
        for records, tensors in self.iter_batches(batch_size=1, record_ids=[record_id]):
            return records[0], {key: tensor[0] for key, tensor in tensors.items()}
        raise KeyError(record_id)
