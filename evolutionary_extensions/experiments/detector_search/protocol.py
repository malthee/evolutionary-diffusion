"""Freeze held-out factors and query identities before looking at their scores."""

import json
from pathlib import Path

from evolutionary_extensions.experiments.detector_evaluation.config import DETECTORS
from evolutionary_extensions.persistence.packaging import atomic_json


def search_targets(config):
    """A frozen nonempty subset controls optimization, independent of evaluators."""
    value = config.get("targets", list(DETECTORS))
    if (
        not isinstance(value, (list, tuple))
        or not value
        or len(set(value)) != len(value)
        or any(t not in DETECTORS for t in value)
    ):
        raise ValueError("Require distinct known optimization targets")
    return tuple(t for t in DETECTORS if t in value)


def plan_targets(plan):
    return search_targets({"targets": plan.get("targets", list(DETECTORS))})


def heldout_plan(finalists, benchmark_images):
    targets = search_targets({"targets": list(finalists)})
    subjects = []
    for target in targets:
        records = finalists[target]["records"]
        if not 1 <= len(records) <= 20 or len(
            {r["image"]["sha256"] for r in records}
        ) != len(records):
            raise ValueError("Require up to 20 distinct frozen image hashes per target")
        subjects.extend(
            {
                "subject": f"{target}-{r['index']:05d}",
                "group": "optimized",
                "target": target,
                "candidate_index": r["index"],
                "noise0_sha256": r["image"]["sha256"],
            }
            for r in records
        )
    for group in ("random", "prompted"):
        # IDs and ordering predate detector scores; no fitness selection here.
        controls = sorted(
            [
                r
                for r in benchmark_images
                if r["group"] == group and r["codec"] == "png"
            ],
            key=lambda r: r["image_id"],
        )[:20]
        if len(controls) != 20 or len({r["image_id"] for r in controls}) != 20:
            raise ValueError("Require 20 unselected benchmark controls per group")
        subjects.extend(
            {
                "subject": r["image_id"],
                "group": group,
                "target": None,
                "embedding_index": r["embedding_index"],
                "noise0_sha256": r["sha256"],
            }
            for r in controls
        )
    if len({r["subject"] for r in subjects}) != len(subjects):
        raise ValueError("Held-out subject identities collide")
    rows = [
        {**subject, "diffusion_seed": seed, "codec": codec}
        for subject in subjects
        for seed in (0, 1, 2, 3)
        for codec in ("png", "jpeg95", "jpeg75")
    ]
    result = {
        "selection": "20 fitness-ranked distinct PNG hashes per target; first 20 frozen IDs per control group",
        "subjects": subjects,
        "reduced_unique_finalist_sets": {
            target: len(finalists[target]["records"])
            for target in targets
            if len(finalists[target]["records"]) < 20
        },
        "variants": rows,
        "detectors": list(DETECTORS),
        "diffusion_seeds": [0, 1, 2, 3],
        "codecs": ["png", "jpeg95", "jpeg75"],
        "extra_ssp_patch_seeds": [43, 44, 45],
        "extra_ssp_patch_scope": "noise seed 0, all subjects and codecs",
        "logical_primary_matrix_scores": len(rows) * len(DETECTORS),
        "logical_extra_ssp_scores": len(subjects) * 3 * 3,
        "new_generated_images": len(subjects) * 3,
        "search_feedback": False,
        "reuse_policy": "Reuse only identical image hash, codec and scorer configuration; report actual fresh calls separately",
        "calibration_queries_outside_search": {
            "ssp_common_search_pool_png": 200 if "ssp" in targets else 0,
            "aeroblade_independent_reconstruction_seed_png_jpeg95": 400,
            "ssp_unseen_patch_seed_png_fpr_drift": 600,
        },
    }
    if targets != tuple(DETECTORS):
        result["targets"] = list(targets)
    return result


def freeze_heldout(path, plan):
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("Held-out plan is already frozen")
    atomic_json(path, plan)
