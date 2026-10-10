"""Reproducible scientific exports and reviewed rows for the local Data app."""

import csv
import itertools
import json
import time
from pathlib import Path

import numpy as np

from evolutionary_extensions.persistence.packaging import atomic_json, snapshot

from .config import CODECS, DETECTORS

COLORS = ("#285b8e", "#af8324", "#c25f35", "#6b7634", "#ad6285")


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot export empty evidence")
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(v) if isinstance(v, (list, dict)) else v
                    for k, v in row.items()
                }
            )


def export_report(config):
    started = time.perf_counter()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    root = Path(config["output"])
    data = json.loads((root / "analysis.json").read_text())
    scores, metrics = data["scores"], data["metrics"]
    target = root / "reports"
    target.mkdir(parents=True, exist_ok=True)
    # Uniform exported columns retain specialized evidence as explicit JSON.
    columns = sorted(set().union(*(r.keys() for r in scores)))
    write_csv(target / "scores.csv", [{k: r.get(k) for k in columns} for r in scores])
    atomic_json(target / "scores.json", scores)
    write_csv(target / "metrics.csv", metrics)
    atomic_json(target / "paired_differences.json", data["paired_differences"])
    plots = target / "figures"
    plots.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )

    def save(figure, name):
        figure.tight_layout()
        for extension in ("png", "pdf", "svg"):
            figure.savefig(plots / f"{name}.{extension}", dpi=180, bbox_inches="tight")
        plt.close(figure)

    primary = [m for m in metrics if m["regime"] == "png_locked"]
    for codec, group in itertools.product(CODECS, ("random", "prompted")):
        selected = [m for m in primary if m["codec"] == codec and m["group"] == group]
        figure, ax = plt.subplots(figsize=(7, 5))
        for color, m in zip(COLORS, selected):
            ax.plot(
                m["roc"]["fpr"],
                m["roc"]["tpr"],
                color=color,
                label=f"{m['detector']} ({m['auroc']:.3f})",
            )
        ax.plot([0, 1], [0, 1], color=".5", linestyle="--", linewidth=1)
        ax.set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="False-positive rate",
            ylabel="Detection rate",
            title=f"Real versus {group} · {codec} · n={config['test_count']} per group",
        )
        ax.legend(title="AUROC", loc="lower right")
        save(figure, f"roc-{group}-{codec}")
        figure, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, metric, interval, title in zip(
            axes,
            ("detection_rate", "false_positive_rate"),
            ("detection_ci95", "false_positive_ci95"),
            ("Detection rate", "False-positive rate"),
        ):
            for i, (color, m) in enumerate(zip(COLORS, selected)):
                low, high = m[interval]
                ax.errorbar(
                    m[metric],
                    i,
                    xerr=[[m[metric] - low], [high - m[metric]]],
                    fmt="o",
                    color=color,
                    capsize=3,
                )
            ax.set(yticks=range(5), yticklabels=DETECTORS, xlim=(0, 1), title=title)
            ax.invert_yaxis()
        figure.suptitle(f"{group} · {codec} · PNG thresholds · 95% Wilson intervals")
        save(figure, f"rates-{group}-{codec}")
        figure, axes = plt.subplots(1, 5, figsize=(14, 3))
        for ax, m in zip(axes, selected):
            c = m["confusion"]
            values = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
            ax.imshow(values, cmap="Blues", vmin=0, vmax=config["test_count"])
            for i, j in itertools.product(range(2), repeat=2):
                ax.text(
                    j,
                    i,
                    str(values[i, j]),
                    ha="center",
                    va="center",
                    color="white"
                    if values[i, j] > config["test_count"] * 0.6
                    else "black",
                )
            ax.set(
                xticks=[0, 1],
                xticklabels=["real", "fake"],
                yticks=[0, 1],
                yticklabels=["real", "fake"],
                title=m["detector"],
                xlabel="Predicted",
            )
        axes[0].set_ylabel("Actual")
        save(figure, f"confusion-{group}-{codec}")
    matrix_rows, correlations, drift, distributions = [], [], [], []
    for codec in CODECS:
        for group in ("real", "random", "prompted"):
            subset = [s for s in scores if s["codec"] == codec and s["group"] == group]
            ids = sorted({s["image_id"] for s in subset})
            lookup = {(s["image_id"], s["detector"]): s for s in subset}
            values = np.array(
                [[lookup[i, d]["synthetic_score"] for d in DETECTORS] for i in ids]
            )
            decisions = np.array(
                [[lookup[i, d]["predicted_fake"] for d in DETECTORS] for i in ids]
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = np.corrcoef(values.T)
            disagreement = np.mean(
                decisions[:, :, None] != decisions[:, None, :], axis=0
            )
            figure, axes = plt.subplots(1, 2, figsize=(10, 4))
            for ax, mat, title, limits in zip(
                axes,
                (corr, disagreement),
                ("Score correlation (Pearson)", "Decision disagreement"),
                ((-1, 1), (0, 1)),
            ):
                image = ax.imshow(
                    mat,
                    cmap="coolwarm" if title.startswith("Score") else "Blues",
                    vmin=limits[0],
                    vmax=limits[1],
                )
                ax.set(
                    xticks=range(5),
                    xticklabels=DETECTORS,
                    yticks=range(5),
                    yticklabels=DETECTORS,
                    title=title,
                )
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
                figure.colorbar(image, ax=ax, shrink=0.7)
            figure.suptitle(f"{group} · {codec} · n={len(ids)}")
            save(figure, f"agreement-{group}-{codec}")
            for a, b in itertools.product(range(5), repeat=2):
                correlations.append(
                    {
                        "codec": codec,
                        "group": group,
                        "a": DETECTORS[a],
                        "b": DETECTORS[b],
                        "correlation": float(corr[a, b])
                        if np.isfinite(corr[a, b])
                        else None,
                        "disagreement": float(disagreement[a, b]),
                    }
                )
            for image_id in ids:
                row = lookup[image_id, DETECTORS[0]]
                thumbnail = target / "thumbnails" / codec / f"{image_id}.jpg"
                thumbnail.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(root / row["path"]) as image:
                    image.thumbnail((96, 96))
                    image.save(thumbnail, quality=85)
                matrix_rows.append(
                    {
                        "image_id": image_id,
                        "codec": codec,
                        "group": group,
                        "thumbnail": f"thumbnails/{codec}/{image_id}.jpg",
                        "image_url": f"../{row['path']}",
                        "detectors": {d: lookup[image_id, d] for d in DETECTORS},
                    }
                )
    for detector in DETECTORS:
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, codec in zip(axes, CODECS):
            for color, group in zip(
                COLORS, ("calibration", "real", "random", "prompted")
            ):
                values = [
                    s["synthetic_score"]
                    for s in scores
                    if (s["detector"], s["codec"], s["group"])
                    == (detector, codec, group)
                ]
                ax.hist(
                    values,
                    bins=20,
                    histtype="step",
                    density=True,
                    color=color,
                    label=f"{group} (n={len(values)})",
                )
                distributions.extend(
                    {"detector": detector, "codec": codec, "group": group, "score": v}
                    for v in values
                )
            ax.axvline(
                data["thresholds"][f"{detector}/png"],
                color=".3",
                linestyle="--",
                label="PNG threshold",
            )
            ax.set(title=codec, xlabel="Synthetic-oriented raw score", ylabel="Density")
            ax.legend(fontsize=8)
        figure.suptitle(detector)
        save(figure, f"distribution-{detector}")
        lookup = {
            (s["image_id"], s["codec"]): s for s in scores if s["detector"] == detector
        }
        figure, ax = plt.subplots(figsize=(7, 4))
        for i, group in enumerate(("real", "random", "prompted")):
            ids = sorted(
                {s["image_id"] for s in lookup.values() if s["group"] == group}
            )
            delta = [
                lookup[k, "jpeg95"]["synthetic_score"]
                - lookup[k, "png"]["synthetic_score"]
                for k in ids
            ]
            ax.scatter(delta, np.full(len(ids), i), color=COLORS[i], alpha=0.4, s=15)
            drift.extend(
                {"image_id": k, "detector": detector, "group": group, "delta": d}
                for k, d in zip(ids, delta)
            )
        ax.axvline(0, color=".5", linewidth=1)
        ax.set(
            yticks=range(3),
            yticklabels=["real", "random", "prompted"],
            xlabel="JPEG95 − PNG score",
            title=detector,
        )
        save(figure, f"codec-drift-{detector}")
    runtimes = []
    for detector in DETECTORS:
        path = root / "profiles" / f"{detector}.json"
        if path.exists():
            profile = json.loads(path.read_text())
            for candidate in profile["candidates"]:
                runtimes.append({"detector": detector, **candidate})
    if runtimes:
        figure, axes = plt.subplots(1, 2, figsize=(11, 4))
        for color, detector in zip(COLORS, DETECTORS):
            rows = [r for r in runtimes if r["detector"] == detector and r["valid"]]
            axes[0].plot(
                [r["batch"] for r in rows],
                [r["images_per_second"] for r in rows],
                "o-",
                color=color,
                label=detector,
            )
            axes[1].plot(
                [r["batch"] for r in rows],
                [r["peak_bytes"] / 1024**3 for r in rows],
                "o-",
                color=color,
                label=detector,
            )
        axes[0].set(
            xlabel="Batch size",
            ylabel="Images/second",
            title="Validated warm throughput",
        )
        axes[1].set(
            xlabel="Batch size", ylabel="GiB", title="Peak allocated GPU memory"
        )
        axes[0].legend()
        axes[1].legend()
        save(figure, "runtime-memory")
    sources = {
        "provider": "Local detector evaluation",
        "tables": ["scores.csv", "metrics.csv"],
        "evidenceFlow": [
            {
                "title": "Frozen scoring evidence",
                "detail": "analysis.json, images.json, recipe.json, provenance/, thresholds.json",
            }
        ],
    }
    query_rows = {
        "matrix": matrix_rows,
        "metrics": metrics,
        "agreement": correlations,
        "drift": drift,
        "distributions": distributions,
        "runtime": runtimes,
        "recipe": [
            {
                "test_count": config["test_count"],
                "calibration_count": config["calibration_count"],
                "bootstrap_samples": config["bootstrap_samples"],
            }
        ],
    }
    app_data = {
        "title": "Image detector evaluation",
        "surface": "report",
        "generatedAt": __import__("datetime")
        .datetime.now(__import__("datetime").timezone.utc)
        .isoformat(),
        "filters": [],
        "queries": {
            k: {
                "rows": v,
                "source": sources,
                "methods": [
                    {
                        "language": "Python",
                        "code": "python -m evolutionary_extensions.experiments.detector_evaluation.runner analyze --config <frozen-config>",
                    }
                ],
            }
            for k, v in query_rows.items()
        },
    }
    atomic_json(target / "reviewed-data.json", app_data)
    timings = [
        {"stage": path.stem, **json.loads(path.read_text())}
        for path in sorted((root / "timing").glob("*.json"))
    ]
    atomic_json(target / "stage_timings.json", timings)
    if timings:
        columns = sorted(set().union(*(r.keys() for r in timings)))
        write_csv(
            target / "stage_timings.csv",
            [{k: r.get(k) for k in columns} for r in timings],
        )
    lines = [
        "# Image detector evaluation",
        "",
        f"Validated primary scores: **{data['primary_score_count']}**.",
        "",
        "SDXL-Turbo uses one step, guidance zero and independently reset noise seed 0. Random embedding outputs were retained without selection. Results are conditional on this generator/noise configuration.",
        "",
        "Primary decisions use strict score > PNG calibration threshold at empirical FPR ≤5%. JPEG thresholds are also provided as secondary results. Calibration uncertainty is conditional; test rate intervals are Wilson 95%, ranking intervals use paired seeded bootstrap resampling.",
        "",
        "| Detector | Codec | Generated group | AUROC (95% CI) | AP | Detection | Real FPR | Balanced accuracy |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in primary:
        lines.append(
            f"| {m['detector']} | {m['codec']} | {m['group']} | {m['auroc']:.3f} [{m['auroc_ci95'][0]:.3f}, {m['auroc_ci95'][1]:.3f}] | {m['ap']:.3f} | {m['detection_rate']:.1%} | {m['false_positive_rate']:.1%} | {m['balanced_accuracy']:.3f} |"
        )
    lines += [
        "",
        "The matrix colors are percentiles relative to each detector's PNG real calibration scores, not probabilities. SSP raw scores are human probabilities; AEROBLADE raw scores are minimum reconstruction errors. Synthetic scores reverse those two scales.",
        "",
        "RAISE: Dang-Nguyen et al., RAISE – A Raw Images Dataset for Digital Image Forensics, ACM MMSys 2015. Sources: https://loki.disi.unitn.it/RAISE/guide.html; Synthbuster: https://zenodo.org/records/10066460.",
        "",
    ]
    (target / "findings.md").write_text("\n".join(lines))
    atomic_json(
        root / "timing/report_exports.json",
        {"exports_seconds": time.perf_counter() - started},
    )
    atomic_json(
        target / "export_checksums.json",
        {k: v for k, v in snapshot(target).items() if k != "export_checksums.json"},
    )
    return target
