"""Locked operating points and paired, image-level uncertainty estimates."""

import itertools
import json
import math
from pathlib import Path

import numpy as np

from evolutionary_extensions.persistence.packaging import atomic_json

from .config import CODECS, DETECTORS
from .journal import validate_complete


def threshold_at_fpr(scores, fpr=0.05):
    values = np.sort(np.asarray(scores, dtype=float))
    if not len(values) or not np.isfinite(values).all() or not 0 <= fpr < 1:
        raise ValueError("Require finite calibration scores and a valid FPR")
    # Strict '>' handles ties conservatively, including a constant detector.
    return float(values[len(values) - math.floor(len(values) * fpr) - 1])


def wilson(successes, count):
    if count <= 0 or not 0 <= successes <= count:
        raise ValueError("Invalid binomial counts")
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count**2)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def ranking_metrics(real, fake):
    values = np.concatenate([real, fake])
    labels = np.concatenate([np.zeros(len(real)), np.ones(len(fake))])
    order = np.argsort(-values, kind="stable")
    values, labels = values[order], labels[order]
    ends = np.r_[np.flatnonzero(np.diff(values)), len(values) - 1]
    tp = np.cumsum(labels)[ends]
    fp = 1 + ends - tp
    tpr, fpr = np.r_[0, tp / len(fake)], np.r_[0, fp / len(real)]
    precision = tp / (tp + fp)
    ap = float(np.sum(np.diff(np.r_[0, tp / len(fake)]) * precision))
    return {
        "auroc": float(np.trapezoid(tpr, fpr)),
        "ap": ap,
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
    }


def summarize(real, fake, threshold):
    fp, tp = int(np.sum(real > threshold)), int(np.sum(fake > threshold))
    nr, nf = len(real), len(fake)
    return {
        **ranking_metrics(real, fake),
        "threshold": threshold,
        "false_positive_rate": fp / nr,
        "detection_rate": tp / nf,
        "false_positive_ci95": wilson(fp, nr),
        "detection_ci95": wilson(tp, nf),
        "balanced_accuracy": ((nr - fp) / nr + tp / nf) / 2,
        "confusion": {"tn": nr - fp, "fp": fp, "fn": nf - tp, "tp": tp},
        "n_real": nr,
        "n_generated": nf,
    }


def percentile(calibration, score):
    return float(100 * np.mean(np.asarray(calibration) <= score))


def analyze(config):
    root = Path(config["output"])
    rows = json.loads((root / "images.json").read_text())
    scores = [
        json.loads(p.read_text()) for p in sorted((root / "scores").glob("*/*.json"))
    ]
    count = validate_complete(rows, scores, DETECTORS)
    expected = (
        (3 * config["test_count"] + config["calibration_count"]) * 2 * len(DETECTORS)
    )
    if count != expected:
        raise ValueError("Input pool differs from the configured sample counts")
    thresholds, calibration = {}, {}
    for detector in DETECTORS:
        for codec in CODECS:
            values = [
                r["synthetic_score"]
                for r in scores
                if (r["detector"], r["codec"], r["group"])
                == (detector, codec, "calibration")
            ]
            if len(values) != config["calibration_count"]:
                raise ValueError("Calibration denominator differs")
            calibration[detector, codec] = values
            thresholds[f"{detector}/{codec}"] = threshold_at_fpr(values)
    threshold_path = root / "thresholds.json"
    if threshold_path.exists() and json.loads(threshold_path.read_text()) != thresholds:
        raise ValueError("Frozen thresholds changed")
    atomic_json(threshold_path, thresholds)
    for record in scores:
        detector = record["detector"]
        record["calibration_percentile"] = percentile(
            calibration[detector, "png"], record["synthetic_score"]
        )
        record["predicted_fake"] = (
            record["synthetic_score"] > thresholds[f"{detector}/png"]
        )
    metrics, bootstraps = [], {}
    for detector, codec, group in itertools.product(
        DETECTORS, CODECS, ("random", "prompted")
    ):
        # Align photographs and prompted controls by their source-photo identity.
        selected = [
            r for r in scores if r["detector"] == detector and r["codec"] == codec
        ]
        real_rows = sorted(
            [r for r in selected if r["group"] == "real"], key=lambda r: r["pair_id"]
        )
        fake_rows = sorted(
            [r for r in selected if r["group"] == group], key=lambda r: r["pair_id"]
        )
        if (
            len(real_rows) != config["test_count"]
            or len(fake_rows) != config["test_count"]
        ):
            raise ValueError("Test denominator differs")
        if group == "prompted" and [r["pair_id"] for r in real_rows] != [
            r["pair_id"] for r in fake_rows
        ]:
            raise ValueError("Real/prompted pairing differs")
        real = np.array([r["synthetic_score"] for r in real_rows])
        fake = np.array([r["synthetic_score"] for r in fake_rows])
        rng = np.random.default_rng(config["sample_seed"] + 1)
        draws = []
        for _ in range(config["bootstrap_samples"]):
            ir = rng.integers(0, len(real), len(real))
            jf = ir if group == "prompted" else rng.integers(0, len(fake), len(fake))
            m = ranking_metrics(real[ir], fake[jf])
            draws.append([m["auroc"], m["ap"]])
        boot = np.asarray(draws)
        bootstraps[detector, codec, group] = boot
        for regime in ("png_locked", "codec_calibrated"):
            threshold = thresholds[
                f"{detector}/{'png' if regime == 'png_locked' else codec}"
            ]
            entry = {
                "detector": detector,
                "codec": codec,
                "group": group,
                "regime": regime,
                **summarize(real, fake, threshold),
            }
            entry["auroc_ci95"], entry["ap_ci95"] = np.quantile(
                boot, [0.025, 0.975], axis=0
            ).T.tolist()
            metrics.append(entry)
    primary = {
        (m["detector"], m["codec"], m["group"]): m
        for m in metrics
        if m["regime"] == "png_locked"
    }
    differences = []
    for a, b in itertools.combinations(DETECTORS, 2):
        for codec, group in itertools.product(CODECS, ("random", "prompted")):
            intervals = np.quantile(
                bootstraps[a, codec, group] - bootstraps[b, codec, group],
                [0.025, 0.975],
                axis=0,
            ).T
            point_a = primary[a, codec, group]
            point_b = primary[b, codec, group]
            differences.append(
                {
                    "a": a,
                    "b": b,
                    "codec": codec,
                    "group": group,
                    "auroc_difference": point_a["auroc"] - point_b["auroc"],
                    "ap_difference": point_a["ap"] - point_b["ap"],
                    "auroc_difference_ci95": intervals[0].tolist(),
                    "ap_difference_ci95": intervals[1].tolist(),
                }
            )
    codec_differences = []
    for detector, group in itertools.product(DETECTORS, ("random", "prompted")):
        ci = np.quantile(
            bootstraps[detector, "jpeg95", group] - bootstraps[detector, "png", group],
            [0.025, 0.975],
            axis=0,
        ).T
        codec_differences.append(
            {
                "detector": detector,
                "group": group,
                "comparison": "jpeg95 minus png",
                "auroc_difference_ci95": ci[0].tolist(),
                "ap_difference_ci95": ci[1].tolist(),
            }
        )
    output = {
        "primary_score_count": count,
        "thresholds": thresholds,
        "metrics": metrics,
        "paired_differences": differences,
        "codec_differences": codec_differences,
        "scores": scores,
        "uncertainty": "95% intervals; bootstrap thresholds conditional on frozen calibration; Wilson rates",
    }
    atomic_json(root / "analysis.json", output)
    return output
