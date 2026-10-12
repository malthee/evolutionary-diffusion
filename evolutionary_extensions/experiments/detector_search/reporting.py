"""Read-only held-out evidence validation and reproducible CPU exports."""

import argparse
import csv
import json
from pathlib import Path

from evolutionary_extensions.experiments.detector_evaluation.config import (
    DETECTORS,
    digest,
)
from evolutionary_extensions.experiments.detector_evaluation.data import (
    patch_coordinates,
)
from evolutionary_extensions.persistence.packaging import atomic_json, hashes

from .analysis import robustness_statistics
from .heldout import validate_value
from .heldout_rendering import checked_path
from .heldout_runner import validate_variants
from .protocol import plan_targets


def read_json(path):
    return json.loads(Path(path).read_text())


def load_measurements(search_root):
    """Require a complete frozen factorial; never load models or create queries."""
    search_root = Path(search_root).resolve()
    root = search_root / "heldout"
    plan = read_json(search_root / "heldout_plan.json")
    complete = read_json(search_root / "heldout_inference_complete.json")
    targets = plan_targets(plan)
    budget_total = 6400 * len(targets)
    expected = plan["logical_primary_matrix_scores"] + plan["logical_extra_ssp_scores"]
    if (
        complete.get("status") != "inference_complete"
        or complete.get("optimizer_candidates") != budget_total
        or complete.get("heldout_matrix_scores") != expected
        or complete.get("ssp_extra_calibration_scores") != 600
        or plan.get("search_feedback") is not False
        or plan.get("detectors") != list(DETECTORS)
    ):
        raise ValueError("Require complete held-out inference for the frozen plan")
    subjects = plan["subjects"]
    if (
        len({s["subject"] for s in subjects}) != len(subjects)
        or any(
            not 1 <= sum(s["target"] == target for s in subjects) <= 20
            for target in targets
        )
        or any(
            sum(s["group"] == group for s in subjects) != 20
            for group in ("random", "prompted")
        )
    ):
        raise ValueError("Frozen subject selection differs from the protocol")
    images = read_json(root / "images.json")
    count = validate_variants(images, plan)
    if (
        count != len(subjects) * 12
        or plan["logical_primary_matrix_scores"] != count * 5
        or plan["logical_extra_ssp_scores"] != len(subjects) * 9
    ):
        raise ValueError("Frozen factorial counts differ")
    generation = read_json(root / "generation_complete.json")
    if (
        generation.get("status") != "complete"
        or generation.get("variants") != count
        or generation.get("plan_sha256") != digest(plan)
    ):
        raise ValueError("Generation completion does not bind the frozen plan")
    verified_files, records, summaries, sources = set(), [], [], []

    def verify(relative, sha):
        key = (relative, sha)
        if key not in verified_files:
            checked_path(root, relative, sha)
            verified_files.add(key)

    for target in DETECTORS:
        if target in targets:
            search_complete = read_json(search_root / target / "search_complete.json")
            if (
                search_complete.get("status") != "complete"
                or search_complete.get("candidates") != 6400
            ):
                raise ValueError(
                    "Require each selected completed 6,400-candidate search"
                )
        marker_path = root / f"{target}_complete.json"
        marker = read_json(marker_path)
        pools = marker["summaries"]
        seeds = (42, 43, 44, 45) if target == "ssp" else (None,)
        if (
            marker.get("status") != "complete"
            or marker.get("detector") != target
            or [s["patch_seed"] for s in pools] != list(seeds)
        ):
            raise ValueError("Missing detector or patch-pool completion")
        for summary in pools:
            seed = summary["patch_seed"]
            configuration = read_json(
                root / "scorers" / target / f"patch{seed or 42}" / "recipe.json"
            )
            identity = digest({"detector": target, "configuration": configuration})
            directory = root / "measurements" / target / identity
            recipe_path = directory / "recipe.json"
            recipe = read_json(recipe_path)
            if (
                summary["identity"] != identity
                or recipe["identity"] != identity
                or recipe["detector"] != target
                or recipe["configuration"] != configuration
                or configuration.get("ssp_patch_seed") != seed
                or summary.get("comparison") != "strict >"
            ):
                raise ValueError(
                    "Held-out scorer identity or threshold comparison changed"
                )
            rows = recipe["rows"]
            matrix = [r for r in rows if r["group"] != "calibration"]
            calibration = [r for r in rows if r["group"] == "calibration"]
            selected = (
                images
                if seed in (None, 42)
                else [r for r in images if r["diffusion_seed"] == 0]
            )
            expected_rows = [
                {
                    **r,
                    "scorer_context": {
                        "patch_coordinates": [
                            list(c) for c in patch_coordinates(r["subject"], seed=seed)
                        ]
                    },
                }
                if target == "ssp"
                else r
                for r in selected
            ]
            expected_cal = 200 if seed in (43, 44, 45) else 0
            if (
                matrix != expected_rows
                or len(calibration) != expected_cal
                or summary["measurements"] != len(rows)
                or summary["matrix_measurements"] != len(matrix)
                or summary["calibration_measurements"] != len(calibration)
                or len({digest(r) for r in rows}) != len(rows)
            ):
                raise ValueError("Score recipe differs from the frozen factorial")
            if calibration and (
                len({r["subject"] for r in calibration}) != 200
                or any(
                    r["codec"] != "png"
                    or r["diffusion_seed"] is not None
                    or r["scorer_context"]["patch_coordinates"]
                    != [list(c) for c in patch_coordinates(r["subject"], seed=seed)]
                    for r in calibration
                )
            ):
                raise ValueError("SSP calibration pool differs")
            if {p.stem for p in (directory / "scores").glob("*.json")} != {
                digest(r) for r in rows
            }:
                raise ValueError("Missing, duplicate or extra score receipts")
            intents = sorted((directory / "queries").glob("*.json"))
            if [p.name for p in intents] != [
                f"{i:06d}.json" for i in range(len(intents))
            ]:
                raise ValueError("Physical query ledger is not contiguous")
            attempted = components = unfinished = 0
            row_identities = {digest(r) for r in rows}
            for intent_path in intents:
                intent = read_json(intent_path)
                n = len(intent["rows"])
                keys = [
                    digest(
                        {
                            "sha256": r["sha256"],
                            "codec": r["codec"],
                            "detector_identity": identity,
                            "context": r.get("scorer_context", {}),
                        }
                    )
                    for r in intent["rows"]
                ]
                if (
                    n == 0
                    or intent["identity"] != identity
                    or intent["state"] not in ("started", "complete")
                    or intent["logical_queries"] != n
                    or intent["component_calls"]
                    != n * (3 if target == "aeroblade" else 1)
                    or any(digest(r) not in row_identities for r in intent["rows"])
                    or intent["score_keys"] != keys
                    or len(set(keys)) != len(keys)
                ):
                    raise ValueError("Physical query ledger provenance differs")
                attempted += n
                components += intent["component_calls"]
                unfinished += n if intent["state"] == "started" else 0
            if any(
                summary["queries"][key] != value
                for key, value in {
                    "attempted_queries": attempted,
                    "component_calls": components,
                    "unfinished_queries": unfinished,
                }.items()
            ):
                raise ValueError("Physical query counts differ from the ledger")
            if (
                attempted
                and read_json(
                    root / "scorers" / target / f"patch{seed or 42}" / "parity.json"
                ).get("passed")
                is not True
            ):
                raise ValueError("Native/repeat/offline parity remains unresolved")
            for row in rows:
                verify(row["path"], row["sha256"])
                path = directory / "scores" / f"{digest(row)}.json"
                record = read_json(path)
                key = digest(
                    {
                        "sha256": row["sha256"],
                        "codec": row["codec"],
                        "detector_identity": identity,
                        "context": row.get("scorer_context", {}),
                    }
                )
                if (
                    record["identity"] != identity
                    or record["row"] != row
                    or record["score_key"] != key
                ):
                    raise ValueError("Completed score receipt provenance differs")
                value = record["value"]
                validate_value(target, value)
                if target == "ssp" and (
                    value.get("coordinates")
                    != row["scorer_context"]["patch_coordinates"]
                    or type(value.get("selected_index")) is not int
                    or not 0 <= value["selected_index"] < 64
                    or "patch_path" not in value
                ):
                    raise ValueError("Selected SSP patch provenance differs")
                if "patch_path" in value:
                    verify(value["patch_path"], value["patch_sha256"])
                reused = record.get("reused_from")
                if reused and reused.get("phase") == "search":
                    source = checked_path(
                        search_root, reused["score_path"], reused["score_sha256"]
                    )
                    checked_path(
                        search_root,
                        Path(reused["score_path"]).parents[1] / "recipe.json",
                        reused["recipe_sha256"],
                    )
                    descriptor = checked_path(
                        search_root,
                        Path(reused["score_path"]).parents[1]
                        / "scoring-procedure.json",
                        reused["procedure_sha256"],
                    )
                    if (
                        read_json(descriptor)["procedure"]
                        != configuration["effective_procedure"]
                    ):
                        raise ValueError("Reused scoring procedure differs")
                    original = read_json(source)
                    if any(
                        original.get(k) != v
                        for k, v in value.items()
                        if k != "patch_path"
                    ):
                        raise ValueError("Reused search score differs from its source")
                elif reused:
                    original = read_json(
                        directory / "scores" / f"{reused['measurement']}.json"
                    )
                    if (
                        reused["score_key"] != key
                        or original["score_key"] != key
                        or original["value"] != value
                    ):
                        raise ValueError("Reused alias differs from its physical score")
                records.append(
                    {
                        **record,
                        "detector": target,
                        "patch_seed": seed,
                        "threshold_png": summary["threshold_png"],
                        "secondary_threshold_jpeg95": summary[
                            "secondary_threshold_jpeg95"
                        ],
                        "receipt_path": str(path.relative_to(search_root)),
                        "receipt_sha256": hashes(path)["sha256"],
                    }
                )
            summaries.append({"detector": target, **summary})
            sources.append(
                {
                    "path": str(recipe_path.relative_to(search_root)),
                    "sha256": hashes(recipe_path)["sha256"],
                }
            )
        sources.append(
            {
                "path": str(marker_path.relative_to(search_root)),
                "sha256": hashes(marker_path)["sha256"],
            }
        )
    if sum(r["row"]["group"] != "calibration" for r in records) != expected:
        raise ValueError("Completed held-out matrix score count differs")
    return records, {
        "plan_sha256": digest(plan),
        "matrix_scores": expected,
        "calibration_scores": 600,
        "optimizer_candidates": budget_total,
        "targets": list(targets),
        "summaries": summaries,
        "source_receipts": sources,
    }


def export_reports(search_root):
    records, evidence = load_measurements(search_root)
    statistics = robustness_statistics(records)
    root = Path(search_root).resolve() / "reports"
    atomic_json(root / "scores.json", records)
    atomic_json(root / "robustness.json", {**statistics, "evidence": evidence})
    fields = (
        "subject",
        "group",
        "target",
        "diffusion_seed",
        "codec",
        "detector",
        "patch_seed",
        "raw",
        "synthetic_score",
        "threshold_png",
        "detected",
        "path",
        "sha256",
        "patch_path",
        "receipt_path",
        "receipt_sha256",
        "reused_from",
    )
    with (root / "scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row, value = record["row"], record["value"]
            item = {k: row.get(k) for k in fields}
            item.update(
                {
                    k: record.get(k)
                    for k in (
                        "detector",
                        "patch_seed",
                        "threshold_png",
                        "receipt_path",
                        "receipt_sha256",
                    )
                }
            )
            item.update(
                raw=value["raw"],
                synthetic_score=value["synthetic_score"],
                detected=value["synthetic_score"] > record["threshold_png"],
                patch_path=value.get("patch_path"),
                reused_from=json.dumps(record.get("reused_from"), sort_keys=True),
            )
            writer.writerow(item)
    for key in (
        "rates",
        "paired_differences",
        "unseen_noise_averages",
        "ssp_patch_fpr_drift",
        "source_target_transfer",
    ):
        atomic_json(root / f"{key}.json", statistics[key])
    lines = [
        "# Detector search and held-out robustness",
        "",
        f"{evidence['optimizer_candidates']:,} fresh optimizer candidates; completed budgets of 6,400 for the frozen selected targets.",
        f"{evidence['matrix_scores']} frozen matrix score identities and 600 additional SSP calibration measurements.",
        "",
        "A not-detected decision means synthetic score ≤ the unchanged PNG calibration threshold. It is conditional on this detector, selected image set and scoring procedure.",
        "",
        "Intervals resample subjects, keeping their detector/codec/noise/patch variants together. Finalists are fitness-selected; controls are unselected. Independent optimizer-restart uncertainty is not measured.",
        "",
        "## Noise-0 PNG source-to-target transfer",
        "",
        "| Optimized against | Evaluated with | Not detected | Rate | 95% Wilson interval |",
        "|---|---|---:|---:|---|",
    ]
    for row in statistics["source_target_transfer"]:
        low, high = row["not_detected_ci95"]
        lines.append(
            f"| {row['cohort'].split('/')[1]} | {row['detector']} | {row['not_detected']}/{row['n_subjects']} | {row['not_detected_rate']:.1%} | {low:.1%}–{high:.1%} |"
        )
    lines += [
        "",
        "## Exports",
        "",
        "[Raw scores CSV](scores.csv), [score receipts JSON](scores.json), [paired robustness statistics](robustness.json).",
        "",
        "Interactive gallery, figures, archive verification and visual audit remain separate completion requirements. These exports alone do not mark the overall experiment complete.",
        "",
    ]
    (root / "README.md").write_text("\n".join(lines))
    # Other export stages overwrite their own receipts on recovery. Binding those
    # here would leave stale hashes after figures, gallery or curves are rebuilt.
    numeric_files = (
        "scores.json",
        "scores.csv",
        "robustness.json",
        "rates.json",
        "paired_differences.json",
        "unseen_noise_averages.json",
        "ssp_patch_fpr_drift.json",
        "source_target_transfer.json",
        "README.md",
    )
    manifest = {name: hashes(root / name)["sha256"] for name in numeric_files}
    atomic_json(
        root / "export_manifest.json",
        {
            "status": "numeric_exports_complete",
            "files": manifest,
            "evidence": evidence,
            "visual_audit_pending": True,
        },
    )
    return statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    export_reports(parser.parse_args().root)


if __name__ == "__main__":
    main()
