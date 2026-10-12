"""Conditional robustness estimates with paired, subject-level resampling."""

import hashlib
import itertools
import math

import numpy as np

from evolutionary_extensions.experiments.detector_evaluation.analysis import wilson
from evolutionary_extensions.experiments.detector_evaluation.config import DETECTORS

from .heldout import validate_value


def cohort(row):
    if row["group"] == "optimized" and row["target"] in DETECTORS:
        return "optimized/" + row["target"]
    if row["group"] in {"random", "prompted"} and row["target"] is None:
        return row["group"]
    raise ValueError("Invalid held-out cohort")


def robustness_statistics(
    records, *, bootstrap_samples=2000, seed=43, detectors=DETECTORS
):
    """Wilson intervals use one fixed condition and one image per subject.

    Bootstrap draws are shared across all detectors, codecs, diffusion seeds
    and SSP patch pools within a cohort. Different target cohorts have distinct
    draws: their optimization trajectories do not create matched content pairs.
    Intervals describe the frozen selected sets, not independent optimizer runs.
    """
    if type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise ValueError("Require a positive bootstrap count")
    buckets, calibration, identities, frozen_thresholds = {}, [], set(), {}
    for record in records:
        row, value = record["row"], record["value"]
        detector, patch_seed = record["detector"], record["patch_seed"]
        threshold = record["threshold_png"]
        if detector not in detectors or not math.isfinite(threshold):
            raise ValueError("Unknown detector or non-finite frozen threshold")
        if frozen_thresholds.setdefault(detector, threshold) != threshold:
            raise ValueError(
                "PNG threshold must remain unchanged across held-out factors"
            )
        validate_value(detector, value)
        identity = (
            row["subject"],
            detector,
            patch_seed,
            row["diffusion_seed"],
            row["codec"],
        )
        if identity in identities:
            raise ValueError("Duplicate held-out statistical identity")
        identities.add(identity)
        if row["group"] == "calibration":
            if (
                detector != "ssp"
                or patch_seed not in (43, 44, 45)
                or row["codec"] != "png"
            ):
                raise ValueError("Unexpected held-out calibration procedure")
            calibration.append(record)
            continue
        if (
            row["diffusion_seed"] not in (0, 1, 2, 3)
            or row["codec"] not in ("png", "jpeg95", "jpeg75")
            or patch_seed not in ((42, 43, 44, 45) if detector == "ssp" else (None,))
            or (patch_seed in (43, 44, 45) and row["diffusion_seed"] != 0)
        ):
            raise ValueError("Unexpected held-out factor")
        key = (
            cohort(row),
            detector,
            patch_seed or 42,
            row["diffusion_seed"],
            row["codec"],
        )
        buckets.setdefault(key, []).append(record)
    cohorts = sorted({key[0] for key in buckets})
    if not cohorts:
        raise ValueError("Missing held-out statistical evidence")
    subjects, draws, arrays, rates = {}, {}, {}, []
    for name in cohorts:
        pools = [pool for key, pool in buckets.items() if key[0] == name]
        subjects[name] = sorted({r["row"]["subject"] for pool in pools for r in pool})
        count = len(subjects[name])
        salt = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")
        draws[name] = np.random.default_rng([seed, salt]).integers(
            0, count, (bootstrap_samples, count)
        )
        for detector, noise, codec in itertools.product(
            detectors, range(4), ("png", "jpeg95", "jpeg75")
        ):
            expected = (name, detector, 42, noise, codec)
            if expected not in buckets:
                raise ValueError("Missing paired detector/codec/noise condition")
        if "ssp" in detectors:
            for patch, codec in itertools.product(
                (43, 44, 45), ("png", "jpeg95", "jpeg75")
            ):
                if (name, "ssp", patch, 0, codec) not in buckets:
                    raise ValueError("Missing unseen SSP patch condition")
    for key, pool in sorted(buckets.items()):
        name, detector, patch, noise, codec = key
        pool.sort(key=lambda r: r["row"]["subject"])
        if [r["row"]["subject"] for r in pool] != subjects[name]:
            raise ValueError("Unpaired subject denominator differs")
        if len({r["threshold_png"] for r in pool}) != 1:
            raise ValueError("Mixed frozen thresholds within a condition")
        passed = np.array(
            [r["value"]["synthetic_score"] <= r["threshold_png"] for r in pool],
            dtype=float,
        )
        arrays[key] = passed
        count, successes = len(passed), int(passed.sum())
        rates.append(
            {
                "cohort": name,
                "detector": detector,
                "patch_seed": patch if detector == "ssp" else None,
                "diffusion_seed": noise,
                "codec": codec,
                "n_subjects": count,
                "not_detected": successes,
                "detected": count - successes,
                "not_detected_rate": successes / count,
                "not_detected_ci95": wilson(successes, count),
                "threshold_png": pool[0]["threshold_png"],
                "comparison": "synthetic_score <= frozen PNG threshold",
            }
        )

    differences = []

    def compare(current, baseline, kind):
        delta = arrays[current] - arrays[baseline]
        name, detector, patch, noise, codec = current
        differences.append(
            {
                "cohort": name,
                "detector": detector,
                "kind": kind,
                "patch_seed": patch if detector == "ssp" else None,
                "diffusion_seed": noise,
                "codec": codec,
                "baseline_patch_seed": baseline[2] if detector == "ssp" else None,
                "baseline_diffusion_seed": baseline[3],
                "baseline_codec": baseline[4],
                "n_paired_subjects": len(delta),
                "not_detected_rate_difference": float(delta.mean()),
                "difference_ci95": np.quantile(
                    delta[draws[name]].mean(axis=1), [0.025, 0.975]
                ).tolist(),
            }
        )

    for key in arrays:
        name, detector, patch, noise, codec = key
        if patch != 42:
            baseline = (name, detector, 42, noise, codec)
            compare(key, baseline, "unseen patch pool minus patch42")
        elif codec != "png":
            compare(key, (name, detector, 42, noise, "png"), "compressed minus PNG")
        if patch == 42 and noise != 0:
            compare(key, (name, detector, 42, 0, codec), "unseen noise minus noise0")
    averages = []
    for name, detector, codec in itertools.product(
        cohorts, detectors, ("png", "jpeg95", "jpeg75")
    ):
        values = np.stack(
            [arrays[name, detector, 42, noise, codec] for noise in (1, 2, 3)], axis=1
        )
        subject_means = values.mean(axis=1)
        all_passed = values.all(axis=1)
        averages.append(
            {
                "cohort": name,
                "detector": detector,
                "codec": codec,
                "noise_seeds": [1, 2, 3],
                "n_subjects": len(values),
                "mean_not_detected_rate": float(subject_means.mean()),
                "subject_bootstrap_ci95": np.quantile(
                    subject_means[draws[name]].mean(axis=1), [0.025, 0.975]
                ).tolist(),
                "not_detected_at_all_three_seeds": int(all_passed.sum()),
                "all_three_rate": float(all_passed.mean()),
                "all_three_ci95": wilson(int(all_passed.sum()), len(values)),
            }
        )
    calibration_rates = []
    if calibration and {r["patch_seed"] for r in calibration} != {43, 44, 45}:
        raise ValueError("Missing SSP calibration patch pool")
    calibration_subjects = None
    for patch in sorted({r["patch_seed"] for r in calibration}):
        pool = [r for r in calibration if r["patch_seed"] == patch]
        subjects_in_pool = [r["row"]["subject"] for r in pool]
        if calibration_subjects is None:
            calibration_subjects = set(subjects_in_pool)
        elif set(subjects_in_pool) != calibration_subjects:
            raise ValueError(
                "SSP calibration photographs are not paired across patch pools"
            )
        if (
            len(set(subjects_in_pool)) != len(pool)
            or len({r["threshold_png"] for r in pool}) != 1
        ):
            raise ValueError("Calibration subject/threshold denominator differs")
        fp = sum(r["value"]["synthetic_score"] > r["threshold_png"] for r in pool)
        calibration_rates.append(
            {
                "patch_seed": patch,
                "n_calibration": len(pool),
                "false_positives": fp,
                "false_positive_rate": fp / len(pool),
                "false_positive_ci95": wilson(fp, len(pool)),
                "threshold_png": pool[0]["threshold_png"],
                "threshold_recalibrated": False,
            }
        )
    return {
        "rates": rates,
        "paired_differences": differences,
        "unseen_noise_averages": averages,
        "ssp_patch_fpr_drift": calibration_rates,
        "source_target_transfer": [
            r
            for r in rates
            if r["cohort"].startswith("optimized/")
            and r["diffusion_seed"] == 0
            and r["codec"] == "png"
            and r["patch_seed"] in (None, 42)
        ],
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "uncertainty_scope": "Conditional on frozen finalist/control sets, fixed calibration and fixed noise/patch seeds; no independent optimizer-run uncertainty",
        "resampling_unit": "subject; all detector, codec, noise and patch measurements remain paired within cohort",
    }
