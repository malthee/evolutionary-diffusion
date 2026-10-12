"""Scientific invariants and interrupted-run recovery for the detector benchmark."""

import copy
import hashlib
import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from evolutionary_extensions.experiments.detector_evaluation.adapters import (
    score_polarity,
)
from evolutionary_extensions.experiments.detector_evaluation.analysis import (
    analyze,
    percentile,
    ranking_metrics,
    threshold_at_fpr,
    wilson,
)
from evolutionary_extensions.experiments.detector_evaluation.assets import extract
from evolutionary_extensions.experiments.detector_evaluation.config import (
    CODECS,
    DETECTORS,
)
from evolutionary_extensions.experiments.detector_evaluation.data import (
    canonical_photo,
    patch_coordinates,
    select_photos,
    simplest_patch,
    write_variants,
)
from evolutionary_extensions.experiments.detector_evaluation.journal import (
    ScoreJournal,
    validate_complete,
)
from evolutionary_extensions.persistence.packaging import atomic_json


def test_seeded_disjoint_selection_is_independent_of_csv_order():
    rows = [{"File": f"r{i:04d}"} for i in range(1000)]
    selected = select_photos(rows)
    assert selected == select_photos(list(reversed(rows)))
    real = {r["File"] for r in selected if r["group"] == "real"}
    calibration = {r["File"] for r in selected if r["group"] == "calibration"}
    assert len(real) == 100 and len(calibration) == 200 and not real & calibration
    with pytest.raises(ValueError):
        select_photos(rows + [rows[0]])
    with pytest.raises(ValueError):
        select_photos([{"File": "../photo"}])


def test_codec_twins_and_canonical_crop(tmp_path):
    image = Image.fromarray(
        np.tile(np.arange(768, dtype=np.uint8)[None, :, None], (512, 1, 3))
    )
    source = tmp_path / "photo.tif"
    image.save(source)
    canonical = canonical_photo(source)
    assert canonical.size == (512, 512)
    assert canonical.getpixel((0, 0)) == image.getpixel((128, 0))
    rows = write_variants(canonical, tmp_path, "photo", "real")
    assert [r["codec"] for r in rows] == list(CODECS)
    assert rows[0]["sha256"] != rows[1]["sha256"]
    with Image.open(tmp_path / rows[0]["path"]) as decoded:
        assert np.array_equal(np.asarray(decoded), np.asarray(canonical))
    from PIL.JpegImagePlugin import get_sampling

    with Image.open(tmp_path / rows[1]["path"]) as decoded:
        assert get_sampling(decoded) == 0


def test_ssp_coordinates_match_randomcrop_draw_order_without_changing_rng():
    import torch
    from torchvision.transforms import RandomCrop

    state = torch.random.get_rng_state().clone()
    coordinates = patch_coordinates("r0001")
    assert torch.equal(state, torch.random.get_rng_state())
    assert coordinates == patch_coordinates("r0001")
    assert coordinates != patch_coordinates("r0002")
    assert len(coordinates) == 64
    key = hashlib.sha256(b"42:r0001").digest()
    with torch.random.fork_rng():
        torch.manual_seed(int.from_bytes(key[:8], "big") % (2**63))
        for left, top in coordinates:
            y, x, h, w = RandomCrop.get_params(Image.new("RGB", (512, 512)), (32, 32))
            assert (x, y, w, h) == (left, top, 32, 32)


def test_ssp_selects_simple_patch_with_stable_ties():
    a = np.random.default_rng(5).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    a[:32, 32:] = 100
    patch, index = simplest_patch(Image.fromarray(a), [(0, 0), (32, 0), (32, 0)])
    assert index == 1 and np.asarray(patch).min() == 100


def test_score_directions_keep_human_probability_and_error():
    assert score_polarity("ssp", 0.9) == pytest.approx(0.1)
    assert score_polarity("aeroblade", 0.2) == -0.2
    assert score_polarity("bfree", -2.3) == -2.3


def test_full_diffusiondb_sampling_is_reproducible_and_inside_coordinate_bounds():
    import torch

    from evolutionary_extensions.experiments.prompt_embedding.initialization import (
        initial_arguments,
        load_bounds,
    )

    bounds, _ = load_bounds()
    with torch.random.fork_rng():
        torch.manual_seed(42)
        first = initial_arguments(bounds, 4)
        torch.manual_seed(42)
        second = initial_arguments(bounds, 4)
    for a, b in zip(first, second):
        for key, field in (
            ("token", "prompt_embeds"),
            ("pooled", "pooled_prompt_embeds"),
        ):
            values = getattr(a, field)
            assert values.dtype == torch.float16
            assert torch.equal(values, getattr(b, field))
            assert torch.isfinite(values).all()
            assert (values.float() >= bounds[f"{key}_min"]).all()
            assert (values.float() <= bounds[f"{key}_max"]).all()


def test_threshold_conservatively_handles_ties_and_fpr_denominator():
    calibration = np.arange(200.0)
    threshold = threshold_at_fpr(calibration)
    assert threshold == 189 and (calibration > threshold).sum() == 10
    assert threshold_at_fpr(np.ones(200)) == 1
    tied = np.r_[np.zeros(185), np.ones(15)]
    assert (tied > threshold_at_fpr(tied)).sum() == 0
    assert percentile(calibration, 99) == 50
    assert wilson(0, 100)[1] == pytest.approx(0.0369935, abs=1e-6)


def test_ranking_metrics_handle_perfect_inverted_and_constant_scores():
    assert ranking_metrics([0, 1], [2, 3])["auroc"] == 1
    assert ranking_metrics([0, 1], [2, 3])["ap"] == 1
    assert ranking_metrics([2, 3], [0, 1])["auroc"] == 0
    tied = ranking_metrics([1, 1], [1, 1])
    assert tied["auroc"] == 0.5 and tied["ap"] == 0.5


def test_score_journal_recovers_completed_entries_and_rejects_changes(tmp_path):
    rows = write_variants(
        Image.new("RGB", (512, 512)), tmp_path, "r1", "real", pair_id="r1"
    )
    journal = ScoreJournal(tmp_path, "ssp", {"checkpoint": "abc"})
    journal.write(rows[0], 0.9, 0.1)
    assert journal.existing(rows[0])["raw_score"] == 0.9
    assert journal.existing(rows[1]) is None
    with pytest.raises(ValueError):
        journal.write(rows[0], 0.9, 0.1)
    with pytest.raises(ValueError):
        ScoreJournal(tmp_path, "ssp", {"checkpoint": "changed"}).existing(rows[0])
    with pytest.raises(ValueError):
        journal.write(rows[1], float("nan"), 0.1)
    (tmp_path / rows[0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError):
        journal.existing(rows[0])


def test_complete_cross_product_requires_exactly_5000_finite_entries():
    rows = [
        {"image_id": str(i), "codec": c, "sha256": "abc", "group": "real"}
        for i in range(500)
        for c in CODECS
    ]
    scores = [
        {
            **r,
            "detector": d,
            "detector_identity": "test",
            "raw_score": 0.3,
            "synthetic_score": score_polarity(d, 0.3),
        }
        for d in DETECTORS
        for r in rows
    ]
    assert validate_complete(rows, scores, DETECTORS) == 5000
    for invalid in (scores[:-1], scores + [scores[0]]):
        with pytest.raises(ValueError):
            validate_complete(rows, invalid, DETECTORS)
    scores[0]["synthetic_score"] = float("inf")
    with pytest.raises(ValueError):
        validate_complete(rows, scores, DETECTORS)


def test_archive_rejects_traversal_and_accepts_dropbox_root_directory(tmp_path):
    archive = tmp_path / "input.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("/", b"")
        z.writestr("sd4/checkpoint.txt", b"weight")
    extract(archive, tmp_path / "model")
    assert (tmp_path / "model/sd4/checkpoint.txt").read_bytes() == b"weight"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../outside", b"unsafe")
    with pytest.raises(ValueError):
        extract(archive, tmp_path / "model")


def fixture_evaluation(root):
    rows = []
    for group in ("calibration", "real", "random", "prompted"):
        for i in range(4):
            pair = f"r{i:03d}" if group in ("real", "prompted") else f"{group}-{i}"
            rows.extend(
                write_variants(
                    Image.new("RGB", (512, 512), (i * 40, 20, 30)),
                    root,
                    f"{group}-{i}",
                    group,
                    pair_id=pair,
                )
            )
    atomic_json(root / "images.json", rows)
    for detector in DETECTORS:
        journal = ScoreJournal(root, detector, {"test": "fixed"})
        for row in rows:
            value = int(row["image_id"].split("-")[-1]) / 10 + (
                0.6 if row["group"] in ("random", "prompted") else 0
            )
            raw = (
                1 - value
                if detector == "ssp"
                else 0.1 * (1 - value)
                if detector == "aeroblade"
                else value
            )
            details = {}
            if detector == "ssp":
                coordinates = patch_coordinates(row["image_id"])
                with Image.open(root / row["path"]) as image:
                    patch, selected_index = simplest_patch(image, coordinates)
                path = root / "patches" / row["codec"] / f"{row['image_id']}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                patch.save(path)
                details = {
                    "coordinates": coordinates,
                    "selected_index": selected_index,
                    "patch_path": str(path.relative_to(root)),
                }
            journal.write(row, raw, score_polarity(detector, raw), **details)
    return {
        "output": str(root),
        "test_count": 4,
        "calibration_count": 4,
        "bootstrap_samples": 25,
        "sample_seed": 42,
    }


def test_paired_bootstrap_and_reanalysis_are_repeatable(tmp_path):
    config = fixture_evaluation(tmp_path)
    first = analyze(config)
    assert first == analyze(config)
    assert first["primary_score_count"] == 160
    assert all(
        d["auroc_difference_ci95"] == [0.0, 0.0] for d in first["paired_differences"]
    )
    assert all(
        m["confusion"] == {"tn": 4, "fp": 0, "fn": 0, "tp": 4} for m in first["metrics"]
    )
    modified = copy.deepcopy(config)
    modified["test_count"] = 5
    with pytest.raises(ValueError):
        analyze(modified)


def test_batch_profile_rejects_changed_image_order(tmp_path, monkeypatch):
    import torch

    from evolutionary_extensions.experiments.detector_evaluation.runner import (
        profile_classifier,
    )

    for function in ("reset_peak_memory_stats", "synchronize"):
        monkeypatch.setattr(torch.cuda, function, lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 1024)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 1024)

    class ReorderedBatch:
        def infer(self, tensor):
            values = tensor.numpy().astype(float)
            return values[::-1] if len(values) > 1 else values

    chosen, _ = profile_classifier(ReorderedBatch(), torch.arange(16), tmp_path, "safe")
    result = json.loads((tmp_path / "profiles/safe.json").read_text())
    assert chosen == 1
    assert [r["valid"] for r in result["candidates"]] == [True, False, False, False]


def test_report_preserves_interval_endpoints_even_outside_point_estimate():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from evolutionary_extensions.experiments.detector_evaluation.report import (
        plot_interval,
    )

    figure, ax = plt.subplots()
    try:
        # Wilson at zero has a positive floating-point lower endpoint. An interval
        # can also exclude a point estimate; neither case justifies changing it.
        for index, (point, bounds) in enumerate(
            ((0.0, wilson(0, 100)), (0.1, [0.2, 0.3]))
        ):
            plot_interval(ax, point, index, bounds, "blue")
            assert ax.collections[index].get_segments()[0].tolist() == [
                [bounds[0], index],
                [bounds[1], index],
            ]
            assert ax.lines[2 * index].get_xdata().tolist() == bounds
            assert ax.lines[2 * index + 1].get_xdata().tolist() == [point]
        figure.canvas.draw()
    finally:
        plt.close(figure)


def test_cpu_report_exports_full_matrix_and_preserves_score_evidence(tmp_path):
    from evolutionary_extensions.experiments.detector_evaluation.report import (
        export_report,
    )

    config = fixture_evaluation(tmp_path)
    data = analyze(config)
    target = export_report(config)
    assert json.loads((target / "scores.json").read_text()) == data["scores"]
    reviewed = json.loads((target / "reviewed-data.json").read_text())
    matrix = reviewed["queries"]["matrix"]["rows"]
    assert len(matrix) == 24
    for row in matrix:
        assert set(row["detectors"]) == set(DETECTORS)
        assert (target / row["thumbnail"]).is_file()
        assert (target / row["image_url"]).is_file()
        assert (tmp_path / row["detectors"]["ssp"]["patch_path"]).is_file()
    assert (target / "figures/rates-random-jpeg95.svg").is_file()
    assert (target / "figures/roc-prompted-png.pdf").is_file()
