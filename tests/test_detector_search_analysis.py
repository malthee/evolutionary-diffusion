import copy

import pytest

from evolutionary_extensions.experiments.detector_search.analysis import (
    robustness_statistics,
)


def measurements():
    records = []
    for detector in ("safe", "ssp"):
        for subject in ("a", "b"):
            for noise in range(4):
                for codec in ("png", "jpeg95", "jpeg75"):
                    score = 0.4 if (subject == "a") == (noise == 0) else 0.6
                    if codec == "jpeg75":
                        score = 0.7
                    records.append(
                        {
                            "detector": detector,
                            "patch_seed": 42 if detector == "ssp" else None,
                            "threshold_png": 0.5,
                            "row": {
                                "subject": subject,
                                "group": "optimized",
                                "target": "safe",
                                "diffusion_seed": noise,
                                "codec": codec,
                            },
                            "value": {
                                "raw": 1 - score if detector == "ssp" else score,
                                "synthetic_score": score,
                            },
                        }
                    )
    base = [
        r for r in records if r["detector"] == "ssp" and r["row"]["diffusion_seed"] == 0
    ]
    for patch in (43, 44, 45):
        for original in base:
            record = copy.deepcopy(original)
            record["patch_seed"] = patch
            if patch == 43:
                record["value"] = {"raw": 0.2, "synthetic_score": 0.8}
            if patch == 45:
                record["value"] = {"raw": 0.5, "synthetic_score": 0.5}
            records.append(record)
        for subject in ("real-a", "real-b"):
            records.append(
                {
                    "detector": "ssp",
                    "patch_seed": patch,
                    "threshold_png": 0.5,
                    "row": {
                        "subject": subject,
                        "group": "calibration",
                        "target": None,
                        "diffusion_seed": None,
                        "codec": "png",
                    },
                    "value": {"raw": 0.2, "synthetic_score": 0.8},
                }
            )
    return records


def test_paired_identical_codecs_have_zero_difference_uncertainty():
    result = robustness_statistics(measurements(), detectors=("safe", "ssp"))
    paired = [
        r
        for r in result["paired_differences"]
        if r["kind"] == "compressed minus PNG" and r["codec"] == "jpeg95"
    ]
    assert len(paired) == 8
    assert all(
        r["difference_ci95"] == [0, 0] and r["not_detected_rate_difference"] == 0
        for r in paired
    )
    assert result["bootstrap_samples"] == 2000
    reversed_input = list(reversed(measurements()))
    assert robustness_statistics(reversed_input, detectors=("safe", "ssp")) == result


def test_unseen_noise_intervals_resample_subjects_and_preserve_rate_ties():
    result = robustness_statistics(measurements(), detectors=("safe", "ssp"))
    pool = next(
        r
        for r in result["unseen_noise_averages"]
        if r["detector"] == "safe" and r["codec"] == "png"
    )
    assert pool["n_subjects"] == 2
    assert pool["mean_not_detected_rate"] == pool["all_three_rate"] == 0.5
    assert pool["subject_bootstrap_ci95"] == [0, 1]
    tied = [r for r in result["rates"] if r["patch_seed"] == 45]
    assert len(tied) == 3 and all(r["not_detected_rate"] == 1 for r in tied)
    assert len(result["source_target_transfer"]) == 2
    assert all(
        r["threshold_recalibrated"] is False and r["false_positive_rate"] == 1
        for r in result["ssp_patch_fpr_drift"]
    )


def test_missing_duplicate_unpaired_and_nonfinite_statistics_fail():
    records = measurements()
    with pytest.raises(ValueError, match="Duplicate"):
        robustness_statistics(records + records[:1], detectors=("safe", "ssp"))
    with pytest.raises(ValueError, match="denominator"):
        robustness_statistics(records[1:], detectors=("safe", "ssp"))
    missing = [r for r in records if r["patch_seed"] != 44]
    with pytest.raises(ValueError, match="unseen SSP"):
        robustness_statistics(missing, detectors=("safe", "ssp"))
    changed = copy.deepcopy(records)
    changed[0]["value"]["raw"] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        robustness_statistics(changed, detectors=("safe", "ssp"))


def test_independent_cohorts_are_not_artificially_content_paired():
    records = [r for r in measurements() if r["row"]["group"] != "calibration"]
    other = copy.deepcopy(records)
    for row in other:
        row["row"].update(
            group="random", target=None, subject="control-" + row["row"]["subject"]
        )
    result = robustness_statistics(records + other, detectors=("safe", "ssp"))
    assert {r["cohort"] for r in result["rates"]} == {"optimized/safe", "random"}
    assert all(r["n_paired_subjects"] == 2 for r in result["paired_differences"])
    assert all(
        r["cohort"] == "optimized/safe" for r in result["source_target_transfer"]
    )


def test_png_threshold_and_real_patch_calibration_pairing_stay_frozen():
    records = measurements()
    changed = copy.deepcopy(records)
    next(r for r in changed if r["row"]["codec"] == "jpeg95")["threshold_png"] = 0.7
    with pytest.raises(ValueError, match="threshold must remain"):
        robustness_statistics(changed, detectors=("safe", "ssp"))
    changed = copy.deepcopy(records)
    next(r for r in changed if r["row"]["group"] == "calibration")["row"]["subject"] = (
        "another photograph"
    )
    with pytest.raises(ValueError, match="not paired"):
        robustness_statistics(changed, detectors=("safe", "ssp"))
