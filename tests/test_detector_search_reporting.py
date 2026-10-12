import csv
import json

import pytest
from PIL import Image

from evolutionary_extensions.experiments.detector_evaluation.config import (
    DETECTORS,
    digest,
)
from evolutionary_extensions.experiments.detector_evaluation.data import (
    patch_coordinates,
)
from evolutionary_extensions.experiments.detector_search.figures import export_figures
from evolutionary_extensions.experiments.detector_search.gallery import build_gallery
from evolutionary_extensions.experiments.detector_search.heldout import (
    HeldoutJournal,
    score_matrix,
)
from evolutionary_extensions.experiments.detector_search.reporting import (
    export_reports,
    load_measurements,
)
from evolutionary_extensions.persistence.packaging import atomic_json, hashes


@pytest.fixture
def completed(tmp_path):
    root = tmp_path / "heldout"
    root.mkdir()
    image = root / "pixels.png"
    Image.new("RGB", (512, 512), (31, 63, 127)).save(image)
    sha = hashes(image)["sha256"]
    patch = root / "patch.png"
    Image.new("RGB", (32, 32), (31, 63, 127)).save(patch)
    patch_sha = hashes(patch)["sha256"]
    subjects = [
        {
            "subject": target,
            "group": "optimized",
            "target": target,
            "noise0_sha256": sha,
        }
        for target in DETECTORS
    ] + [
        {
            "subject": f"{group}-{i}",
            "group": group,
            "target": None,
            "noise0_sha256": sha,
        }
        for group in ("random", "prompted")
        for i in range(20)
    ]
    rows = [
        {
            **s,
            "diffusion_seed": noise,
            "codec": codec,
            "path": "pixels.png",
            "sha256": sha,
        }
        for s in subjects
        for noise in range(4)
        for codec in ("png", "jpeg95", "jpeg75")
    ]
    plan = {
        "subjects": subjects,
        "variants": rows,
        "detectors": list(DETECTORS),
        "search_feedback": False,
        "logical_primary_matrix_scores": len(rows) * 5,
        "logical_extra_ssp_scores": len(subjects) * 9,
    }
    atomic_json(tmp_path / "heldout_plan.json", plan)
    atomic_json(root / "images.json", rows)
    atomic_json(
        root / "generation_complete.json",
        {"status": "complete", "variants": len(rows), "plan_sha256": digest(plan)},
    )
    atomic_json(
        tmp_path / "heldout_inference_complete.json",
        {
            "status": "inference_complete",
            "optimizer_candidates": 32000,
            "heldout_matrix_scores": len(rows) * 5 + len(subjects) * 9,
            "ssp_extra_calibration_scores": 600,
        },
    )
    for target in DETECTORS:
        atomic_json(
            tmp_path / target / "search_complete.json",
            {"status": "complete", "candidates": 6400},
        )
        summaries = []
        for seed in (42, 43, 44, 45) if target == "ssp" else (None,):
            selected = (
                rows
                if seed in (None, 42)
                else [r for r in rows if r["diffusion_seed"] == 0]
            )
            measured = [
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
            if seed in (43, 44, 45):
                measured += [
                    {
                        "subject": f"cal-{i}",
                        "group": "calibration",
                        "target": None,
                        "codec": "png",
                        "diffusion_seed": None,
                        "path": "pixels.png",
                        "sha256": sha,
                        "scorer_context": {
                            "patch_coordinates": [
                                list(c)
                                for c in patch_coordinates(f"cal-{i}", seed=seed)
                            ]
                        },
                    }
                    for i in range(200)
                ]
            config = {"ssp_patch_seed": seed, "batch": 1}
            folder = root / "scorers" / target / f"patch{seed or 42}"
            atomic_json(folder / "recipe.json", config)
            atomic_json(folder / "parity.json", {"passed": True})
            journal = HeldoutJournal(root, target, config, measured)

            def score(pool, target=target):
                result = []
                for row in pool:
                    raw = 0.4
                    value = {
                        "raw": raw,
                        "synthetic_score": 1 - raw
                        if target == "ssp"
                        else -raw
                        if target == "aeroblade"
                        else raw,
                    }
                    if target == "aeroblade":
                        value["reconstruction_errors"] = [raw, raw + 0.1, raw + 0.2]
                    if target == "ssp":
                        value.update(
                            coordinates=row["scorer_context"]["patch_coordinates"],
                            selected_index=0,
                            patch_path="patch.png",
                            patch_sha256=patch_sha,
                        )
                    result.append(value)
                return result

            score_matrix(journal, score, batch=4)
            summaries.append(
                {
                    "identity": journal.identity,
                    "patch_seed": seed,
                    "measurements": len(measured),
                    "matrix_measurements": len(selected),
                    "calibration_measurements": 200 if seed in (43, 44, 45) else 0,
                    "queries": journal.counts(),
                    "threshold_png": -0.5 if target == "aeroblade" else 0.5,
                    "secondary_threshold_jpeg95": -0.5
                    if target == "aeroblade"
                    else 0.5,
                    "comparison": "strict >",
                }
            )
        atomic_json(
            root / f"{target}_complete.json",
            {"status": "complete", "detector": target, "summaries": summaries},
        )
    return tmp_path


def test_exports_recalculate_decisions_and_do_not_modify_inference(
    completed, monkeypatch
):
    before = {str(p): hashes(p)["sha256"] for p in completed.rglob("*") if p.is_file()}
    records, evidence = load_measurements(completed)
    assert len(records) == 3705 and evidence["matrix_scores"] == 3105
    statistics = export_reports(completed)
    assert statistics["bootstrap_samples"] == 2000
    assert all(hashes(p)["sha256"] == sha for p, sha in before.items())
    with (completed / "reports/scores.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(records)
    assert all(
        (row["detected"] == "True")
        == (float(row["synthetic_score"]) > float(row["threshold_png"]))
        for row in rows
    )
    manifest = json.loads((completed / "reports/export_manifest.json").read_text())
    assert (
        manifest["status"] == "numeric_exports_complete"
        and manifest["visual_audit_pending"]
    )
    assert all(
        hashes(completed / "reports" / name)["sha256"] == sha
        for name, sha in manifest["files"].items()
    )
    sources = export_figures(completed)
    assert sources["source-target-transfer"] == statistics["source_target_transfer"]
    assert sources["ssp-patch-fpr-drift"] == statistics["ssp_patch_fpr_drift"]
    figures = json.loads((completed / "reports/figures/manifest.json").read_text())
    assert len(figures["files"]) == 18 and figures["visual_audit_pending"]
    assert all(
        hashes(completed / "reports/figures" / name)["sha256"] == sha
        for name, sha in figures["files"].items()
    )
    monkeypatch.setattr(
        "evolutionary_extensions.experiments.detector_search.gallery.calibration_references",
        lambda *_: (
            {d: [-0.5 if d == "aeroblade" else 0.5] * 200 for d in DETECTORS},
            {"cpu_test_reference": True},
        ),
    )
    payload = build_gallery(completed, completed / "unused-test-benchmark")
    assert len(payload["rows"]) == 540
    gallery = json.loads((completed / "reports/gallery_manifest.json").read_text())
    assert gallery["visual_audit_pending"] and gallery["rows"] == 540
    assert (
        hashes(completed / "reports/gallery.html")["sha256"] == gallery["html_sha256"]
    )
    for name, sha in gallery["thumbnail_hashes"].items():
        with Image.open(completed / "reports/thumbnails" / name) as thumbnail:
            assert thumbnail.size == (128, 128)
        assert hashes(completed / "reports/thumbnails" / name)["sha256"] == sha
    # Retrying numeric exports must not bind mutable gallery/curve exports.
    (completed / "reports/optimizer_curves.svg").write_text("previous curve")
    export_reports(completed)
    manifest = json.loads((completed / "reports/export_manifest.json").read_text())
    assert len(manifest["files"]) == 9
    assert "gallery_manifest.json" not in manifest["files"]
    assert "optimizer_curves.svg" not in manifest["files"]
    (completed / "reports/optimizer_curves.svg").write_text("rebuilt curve")
    assert all(
        hashes(completed / "reports" / name)["sha256"] == sha
        for name, sha in manifest["files"].items()
    )
    assert all(hashes(p)["sha256"] == sha for p, sha in before.items())


def test_missing_scores_changed_pixels_and_ledger_counts_are_rejected(completed):
    marker = completed / "heldout/safe_complete.json"
    original = json.loads(marker.read_text())
    changed = json.loads(marker.read_text())
    changed["summaries"][0]["queries"]["attempted_queries"] += 1
    atomic_json(marker, changed)
    with pytest.raises(ValueError, match="query counts"):
        load_measurements(completed)
    atomic_json(marker, original)
    receipt = next((completed / "heldout/measurements/safe").glob("*/scores/*.json"))
    saved = receipt.read_text()
    receipt.unlink()
    with pytest.raises(ValueError, match="score receipts"):
        load_measurements(completed)
    receipt.write_text(saved)
    (completed / "heldout/pixels.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_measurements(completed)


def test_transfer_figures_preserve_two_selected_targets_and_zero_wilson_boundary(
    completed,
):
    from evolutionary_extensions.experiments.detector_evaluation.analysis import wilson

    export_reports(completed)
    path = completed / "reports/robustness.json"
    data = json.loads(path.read_text())
    selected = ["universal_fake_detect", "bfree"]
    data["evidence"]["targets"] = selected
    for key in (
        "rates",
        "paired_differences",
        "unseen_noise_averages",
        "source_target_transfer",
    ):
        data[key] = [
            r
            for r in data[key]
            if not r["cohort"].startswith("optimized/")
            or r["cohort"].split("/")[1] in selected
        ]
    for r in data["ssp_patch_fpr_drift"]:
        r.update(
            false_positives=0, false_positive_rate=0, false_positive_ci95=wilson(0, 200)
        )
    atomic_json(path, data)
    manifest_path = completed / "reports/export_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["robustness.json"] = hashes(path)["sha256"]
    atomic_json(manifest_path, manifest)
    sources = export_figures(completed)
    assert len(sources["source-target-transfer"]) == 10
    assert {r["cohort"] for r in sources["source-target-transfer"]} == {
        "optimized/" + t for t in selected
    }
    assert all(
        r["false_positive_ci95"] == wilson(0, 200)
        for r in sources["ssp-patch-fpr-drift"]
    )
    assert (
        len(
            json.loads((completed / "reports/figures/manifest.json").read_text())[
                "files"
            ]
        )
        == 18
    )
