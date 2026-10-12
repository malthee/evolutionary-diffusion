"""Incremental held-out measurements, separate from optimizer feedback."""

import json
import math
import time
from pathlib import Path

from evolutionary_extensions.experiments.detector_evaluation.adapters import (
    score_polarity,
)
from evolutionary_extensions.experiments.detector_evaluation.config import (
    DETECTORS,
    digest,
)
from evolutionary_extensions.persistence.packaging import atomic_json, hashes


def validate_value(detector, value):
    raw, synthetic = value["raw"], value["synthetic_score"]
    if not all(math.isfinite(v) for v in (raw, synthetic)):
        raise ValueError("Non-finite held-out score")
    if not math.isclose(synthetic, score_polarity(detector, raw), abs_tol=1e-12):
        raise ValueError("Held-out score polarity differs")
    if detector in {"safe", "universal_fake_detect", "ssp"} and not 0 <= raw <= 1:
        raise ValueError("Held-out probability outside [0,1]")
    if detector == "aeroblade":
        errors = value["reconstruction_errors"]
        if (
            len(errors) != 3
            or not all(math.isfinite(v) and v >= 0 for v in errors)
            or raw != min(errors)
        ):
            raise ValueError("Held-out autoencoder errors differ")


class HeldoutJournal:
    """A score key binds pixels, codec and the effective scoring procedure.

    SSP's effective procedure includes the actual candidate coordinates, not
    merely the seed: per-subject patch pools can differ for identical pixels.
    The subject/seed factors remain in each measurement even when its numeric
    score is reused from an identical physical inference key.
    """

    def __init__(self, root, detector, configuration, rows):
        if detector not in DETECTORS:
            raise ValueError("Unknown held-out detector")
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.detector, self.configuration = detector, configuration
        self.identity = digest({"detector": detector, "configuration": configuration})
        self.directory = self.root / "measurements" / detector / self.identity
        self.rows = {}
        for row in rows:
            key = digest(row)
            if key in self.rows:
                raise ValueError("Duplicate held-out measurement identity")
            self.validate_image(row)
            self.rows[key] = row
        recipe = {
            "identity": self.identity,
            "detector": detector,
            "configuration": configuration,
            "rows": rows,
        }
        path = self.directory / "recipe.json"
        if path.exists() and json.loads(path.read_text()) != recipe:
            raise ValueError("Frozen held-out recipe changed")
        atomic_json(path, recipe)
        self.cache = {}
        for row in rows:
            saved = self.existing(row)
            if saved is not None:
                self.cache[self.score_key(row)] = saved
        self.ledger = self.directory / "queries"
        attempts = sorted(self.ledger.glob("*.json"))
        if [p.name for p in attempts] != [
            f"{i:06d}.json" for i in range(len(attempts))
        ]:
            raise ValueError("Held-out query ledger is not contiguous")
        self.next_attempt = len(attempts)
        self.attempted = self.component_calls = self.unfinished = 0
        for path in attempts:
            record = json.loads(path.read_text())
            if record["identity"] != self.identity or record["state"] not in {
                "started",
                "complete",
            }:
                raise ValueError("Held-out query ledger provenance differs")
            self.attempted += record["logical_queries"]
            self.component_calls += record["component_calls"]
            if record["state"] != "complete":
                self.unfinished += record["logical_queries"]

    def validate_image(self, row):
        path = self.root / row["path"]
        if (
            not path.resolve().is_relative_to(self.root)
            or hashes(path)["sha256"] != row["sha256"]
        ):
            raise ValueError("Held-out image changed")

    def score_key(self, row):
        context = row.get("scorer_context", {})
        if self.detector == "ssp" and (
            len(context.get("patch_coordinates", [])) != 64
            or any(
                len(c) != 2 or any(type(v) is not int or not 0 <= v <= 480 for v in c)
                for c in context["patch_coordinates"]
            )
        ):
            raise ValueError("SSP reuse requires all 64 actual patch coordinates")
        return digest(
            {
                "sha256": row["sha256"],
                "codec": row["codec"],
                "detector_identity": self.identity,
                "context": context,
            }
        )

    def path(self, row):
        if digest(row) not in self.rows:
            raise ValueError("Measurement is outside the frozen held-out plan")
        return self.directory / "scores" / f"{digest(row)}.json"

    def existing(self, row):
        self.validate_image(row)
        path = self.path(row)
        if not path.exists():
            return None
        record = json.loads(path.read_text())
        if (
            record["identity"] != self.identity
            or record["row"] != row
            or record["score_key"] != self.score_key(row)
        ):
            raise ValueError("Held-out score provenance differs")
        self.validate_result(row, record["value"])
        return record

    def validate_result(self, row, value):
        validate_value(self.detector, value)
        if self.detector == "ssp" and (
            value.get("coordinates") != row["scorer_context"]["patch_coordinates"]
            or type(value.get("selected_index")) is not int
            or not 0 <= value["selected_index"] < 64
        ):
            raise ValueError("Held-out selected patch provenance differs")
        if "patch_path" in value:
            self.validate_image(
                {
                    "path": value["patch_path"],
                    "sha256": value["patch_sha256"],
                }
            )

    def write(self, row, value, *, reused_from=None):
        self.validate_result(row, value)
        if self.existing(row) is not None:
            raise ValueError("Refuse to overwrite a completed held-out measurement")
        record = {
            "identity": self.identity,
            "score_key": self.score_key(row),
            "row": row,
            "value": value,
            "reused_from": reused_from,
        }
        atomic_json(self.path(row), record)
        self.cache[record["score_key"]] = record
        return record

    def begin(self, rows):
        if not rows:
            raise ValueError("Cannot record an empty physical inference batch")
        if any(self.existing(row) is not None for row in rows):
            raise ValueError("Completed held-out scores must not be queried again")
        keys = [self.score_key(r) for r in rows]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate physical keys within an inference batch")
        count = len(rows)
        components = count * (3 if self.detector == "aeroblade" else 1)
        path = self.ledger / f"{self.next_attempt:06d}.json"
        atomic_json(
            path,
            {
                "identity": self.identity,
                "rows": rows,
                "score_keys": keys,
                "logical_queries": count,
                "component_calls": components,
                "state": "started",
                "started_unix": time.time(),
            },
        )
        self.next_attempt += 1
        self.attempted += count
        self.component_calls += components
        self.unfinished += count
        return path

    def finish(self, path, values):
        if Path(path).resolve().parent != self.ledger.resolve():
            raise ValueError("Inference intent is outside the owned ledger")
        intent = json.loads(Path(path).read_text())
        if intent["identity"] != self.identity or intent["state"] != "started":
            raise ValueError("Held-out inference intent changed")
        if len(values) != len(intent["rows"]):
            raise ValueError("Held-out batch output count differs")
        # Validate the entire batch before publishing any completed score. A
        # crash while publishing preserves its prefix; recovery queries only
        # unfinished entries and keeps the original intent as uncertain.
        for row, value in zip(intent["rows"], values):
            self.validate_result(row, value)
        records = [self.write(row, value) for row, value in zip(intent["rows"], values)]
        atomic_json(path, {**intent, "state": "complete", "finished_unix": time.time()})
        self.unfinished -= intent["logical_queries"]
        return records

    def counts(self):
        return {
            "attempted_queries": self.attempted,
            "component_calls": self.component_calls,
            "unfinished_queries": self.unfinished,
            "interrupted_device_completion": "unknown within unfinished intents",
        }


def score_matrix(
    journal, score, *, batch=1, guard=lambda: None, progress=lambda **k: None
):
    """Callback receives only pending physical identities, in frozen row order."""
    if type(batch) is not int or batch < 1:
        raise ValueError("Require a validated positive scoring batch")
    pending = []

    def flush():
        if not pending:
            return
        guard()
        intent = journal.begin(pending)
        journal.finish(intent, score(pending))
        pending.clear()
        progress(**journal.counts())

    for row in journal.rows.values():
        if journal.existing(row) is not None:
            continue
        cached = journal.cache.get(journal.score_key(row))
        if cached is not None:
            journal.write(
                row,
                cached["value"],
                reused_from={
                    "measurement": digest(cached["row"]),
                    "score_key": cached["score_key"],
                },
            )
            continue
        # Flush before an alias so the canonical physical score is committed
        # before reuse; aliases never enter the inference callback.
        if any(journal.score_key(r) == journal.score_key(row) for r in pending):
            flush()
            cached = journal.cache[journal.score_key(row)]
            journal.write(
                row,
                cached["value"],
                reused_from={
                    "measurement": digest(cached["row"]),
                    "score_key": cached["score_key"],
                },
            )
        else:
            pending.append(row)
            if len(pending) == batch:
                flush()
    flush()
    records = [journal.existing(row) for row in journal.rows.values()]
    if any(record is None for record in records):
        raise ValueError("Held-out matrix remains incomplete")
    return records
