"""Owned optimizer checkpoints and incremental, hash-bound candidate evidence."""

import hashlib
import json
import math
import os
import time
from pathlib import Path

from evolutionary_extensions.experiments.detector_evaluation.config import digest
from evolutionary_extensions.persistence.packaging import atomic_json, hashes

from .optimizer import SearchOptimizer


class SearchStore:
    """One writer per target; score recovery never changes a sampled population."""

    def __init__(self, root, recipe):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.identity = digest(recipe)
        self.budget = recipe["budget"]
        self.target = recipe.get("target")
        if type(self.budget) is not int or self.budget <= 0:
            raise ValueError("Require a positive candidate budget")
        frozen = self.root / "recipe.json"
        expected = {"identity": self.identity, "recipe": recipe}
        if frozen.exists() and json.loads(frozen.read_text()) != expected:
            raise ValueError("Search recipe changed; use another output directory")
        atomic_json(frozen, expected)
        paths = sorted((self.root / "queries").glob("*.json"))
        if [p.name for p in paths] != [f"{i:06d}.json" for i in range(len(paths))]:
            raise ValueError("Physical query ledger is not contiguous")
        self._next_query = len(paths)
        self._counts = {
            kind: {
                "attempted_queries": 0,
                "component_calls": 0,
                "unfinished_queries": 0,
            }
            for kind in ("fitness", "smoke", "calibration", "heldout")
        }
        for path in paths:
            record = json.loads(path.read_text())
            if record["identity"] != self.identity or record["state"] not in {
                "started",
                "complete",
            }:
                raise ValueError("Physical query ledger identity/state differs")
            count = self._counts[record["kind"]]
            count["attempted_queries"] += record["logical_queries"]
            count["component_calls"] += record["component_calls"]
            if record["state"] != "complete":
                count["unfinished_queries"] += record["logical_queries"]

    def _index(self, index):
        if type(index) is not int or not 0 <= index < self.budget:
            raise ValueError("Candidate index is outside the frozen budget")
        return f"{index:05d}"

    def checkpoint(self, optimizer):
        if optimizer.budget != self.budget:
            raise ValueError("Optimizer budget differs from the frozen recipe")
        content = optimizer.state()
        sha = hashlib.sha256(content).hexdigest()
        path = self.root / "checkpoints" / f"{sha}.pickle"
        path.parent.mkdir(exist_ok=True)
        if not path.exists():
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        # Publishing this pointer is the transaction commit. If interrupted
        # before it, the preceding pending population and its scores are replayed.
        pointer = self.root / "checkpoint.json"
        previous = (
            json.loads(pointer.read_text())["sha256"] if pointer.exists() else None
        )
        atomic_json(
            pointer,
            {
                "identity": self.identity,
                "sha256": sha,
                "completed": optimizer.completed,
                "pending": optimizer.pending is not None,
            },
        )
        # Full-dimensional pending populations are large. Keep the current
        # transaction and its preceding checkpoint; every actual FP16 candidate
        # and generation score remains archived independently for replay.
        for stale in path.parent.glob("*.pickle"):
            if stale.stem not in {sha, previous}:
                stale.unlink()

    def restore(self):
        pointer = self.root / "checkpoint.json"
        if not pointer.exists():
            return None
        saved = json.loads(pointer.read_text())
        sha = saved["sha256"]
        if (
            saved["identity"] != self.identity
            or len(sha) != 64
            or any(c not in "0123456789abcdef" for c in sha)
        ):
            raise ValueError("Optimizer checkpoint identity differs")
        optimizer = SearchOptimizer.restore_owned_state(
            (self.root / "checkpoints" / f"{sha}.pickle").read_bytes(), sha
        )
        if (
            optimizer.completed != saved["completed"]
            or optimizer.budget != self.budget
            or (optimizer.pending is not None) != saved["pending"]
        ):
            raise ValueError("Optimizer checkpoint counters differ")
        return optimizer

    def _validate_image(self, image):
        path = self.root / image["path"]
        if (
            not path.resolve().is_relative_to(self.root)
            or hashes(path)["sha256"] != image["sha256"]
        ):
            raise ValueError("Candidate pixels changed")

    def score(self, index, image):
        self._validate_image(image)
        path = self.root / "scores" / f"{self._index(index)}.json"
        if not path.exists():
            return None
        saved = json.loads(path.read_text())
        if (
            saved["identity"] != self.identity
            or saved["index"] != index
            or saved["image"] != image
        ):
            raise ValueError("Candidate score identity differs")
        if not all(
            math.isfinite(saved[k]) for k in ("objective", "raw", "synthetic_score")
        ):
            raise ValueError("Non-finite candidate score")
        if "patch_path" in saved:
            self._validate_image(
                {"path": saved["patch_path"], "sha256": saved["patch_sha256"]}
            )
        return saved

    def begin_queries(self, indices, images, *, kind="fitness", detector_calls=1):
        if (
            kind not in {"fitness", "smoke", "calibration", "heldout"}
            or type(detector_calls) is not int
            or detector_calls < 1
            or len(indices) != len(images)
            or len(set(indices)) != len(indices)
            or not indices
        ):
            raise ValueError("Invalid physical query intent")
        for index, image in zip(indices, images):
            self._index(index)
            self._validate_image(image)
            if kind == "fitness" and self.score(index, image) is not None:
                raise ValueError("Completed candidate must not be queried again")
        ledger = self.root / "queries"
        ledger.mkdir(exist_ok=True)
        number = self._next_query
        path = ledger / f"{number:06d}.json"
        if path.exists():
            raise ValueError("Physical query ledger is not contiguous")
        atomic_json(
            path,
            {
                "identity": self.identity,
                "indices": indices,
                "images": images,
                "kind": kind,
                "logical_queries": len(indices),
                "component_calls": len(indices) * detector_calls,
                "state": "started",
                "started_unix": time.time(),
            },
        )
        self._next_query += 1
        self._counts[kind]["attempted_queries"] += len(indices)
        self._counts[kind]["component_calls"] += len(indices) * detector_calls
        self._counts[kind]["unfinished_queries"] += len(indices)
        return path

    def finish_queries(self, intent, records):
        intent = Path(intent)
        if intent.parent.resolve() != (self.root / "queries").resolve():
            raise ValueError("Query intent belongs to another store")
        saved = json.loads(intent.read_text())
        if saved["identity"] != self.identity or len(records) != len(saved["indices"]):
            raise ValueError("Query completion identity/order differs")
        # Validate the entire batch before publishing any score receipt.
        for record in records:
            if not all(
                math.isfinite(record[k])
                for k in ("objective", "raw", "synthetic_score")
            ):
                raise ValueError("Non-finite candidate score")
        for index, image, record in zip(saved["indices"], saved["images"], records):
            self._validate_image(image)
            if saved["kind"] == "fitness":
                previous = self.score(index, image)
                value = {
                    **record,
                    "identity": self.identity,
                    "index": index,
                    "image": image,
                }
                if previous is not None and previous != value:
                    raise ValueError("Refuse to overwrite a completed candidate score")
                atomic_json(self.root / "scores" / f"{self._index(index)}.json", value)
        atomic_json(
            intent,
            {
                **saved,
                "state": "complete",
                "records": records,
                "finished_unix": time.time(),
            },
        )
        if saved["state"] != "complete":
            self._counts[saved["kind"]]["unfinished_queries"] -= saved[
                "logical_queries"
            ]

    def query_counts(self):
        return {kind: dict(counts) for kind, counts in self._counts.items()}
