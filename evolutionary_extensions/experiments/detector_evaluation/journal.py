"""Atomic per-image scores make recovery independent of notebook lifetime."""

import json
import math
from pathlib import Path

from evolutionary_extensions.persistence.packaging import atomic_json, hashes

from .config import digest


class ScoreJournal:
    def __init__(self, root, detector, identity):
        self.root = Path(root)
        self.detector = detector
        self.identity = digest(identity)
        self.directory = self.root / "scores" / detector

    def path(self, row):
        return self.directory / f"{row['image_id']}-{row['codec']}.json"

    def existing(self, row):
        path = self.path(row)
        if not path.exists():
            return None
        record = json.loads(path.read_text())
        if (
            record["detector_identity"] != self.identity
            or record["sha256"] != row["sha256"]
            or record["image_id"] != row["image_id"]
            or record["codec"] != row["codec"]
        ):
            raise ValueError("Existing score provenance differs; use a new output root")
        if not math.isfinite(record["synthetic_score"]) or not math.isfinite(
            record["raw_score"]
        ):
            raise ValueError("Non-finite persisted score")
        if hashes(self.root / row["path"])["sha256"] != row["sha256"]:
            raise ValueError("Scored image has changed")
        return record

    def write(self, row, raw_score, synthetic_score, **extra):
        if not all(math.isfinite(float(s)) for s in (raw_score, synthetic_score)):
            raise ValueError("Detector returned a non-finite score")
        if hashes(self.root / row["path"])["sha256"] != row["sha256"]:
            raise ValueError("Image checksum differs")
        if self.existing(row) is not None:
            raise ValueError("Refuse to overwrite a completed score")
        record = {
            **row,
            "detector": self.detector,
            "detector_identity": self.identity,
            "raw_score": float(raw_score),
            "synthetic_score": float(synthetic_score),
            **extra,
        }
        atomic_json(self.path(row), record)
        return record


def validate_complete(rows, scores, detectors):
    from .adapters import score_polarity

    keys = {(r["image_id"], r["codec"]) for r in rows}
    if len(keys) != len(rows):
        raise ValueError("Duplicate input image/codec keys")
    expected = {(d, *k) for d in detectors for k in keys}
    actual = [(r["detector"], r["image_id"], r["codec"]) for r in scores]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("Missing, duplicate or unexpected detector scores")
    by_key = {(r["image_id"], r["codec"]): r for r in rows}
    for detector in detectors:
        if (
            len({r["detector_identity"] for r in scores if r["detector"] == detector})
            != 1
        ):
            raise ValueError("Mixed detector configurations or environments")
    for score in scores:
        row = by_key[(score["image_id"], score["codec"])]
        if score["sha256"] != row["sha256"] or score["group"] != row["group"]:
            raise ValueError("Score/image provenance mismatch")
        if not all(math.isfinite(score[k]) for k in ("raw_score", "synthetic_score")):
            raise ValueError("Non-finite score")
        if not math.isclose(
            score["synthetic_score"],
            score_polarity(score["detector"], score["raw_score"]),
            abs_tol=1e-12,
        ):
            raise ValueError("Detector score polarity differs")
        if (
            score["detector"] in ("safe", "universal_fake_detect", "ssp")
            and not 0 <= score["raw_score"] <= 1
        ):
            raise ValueError("Classifier probability outside [0,1]")
        if score["detector"] == "aeroblade" and score["raw_score"] < 0:
            raise ValueError("Negative reconstruction error")
    return len(expected)
