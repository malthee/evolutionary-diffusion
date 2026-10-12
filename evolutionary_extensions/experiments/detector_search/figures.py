"""Scientific robustness figures from verified numeric exports, never inference."""

import argparse
import math
from pathlib import Path

import numpy as np

from evolutionary_extensions.experiments.detector_evaluation.config import DETECTORS
from evolutionary_extensions.experiments.detector_evaluation.report import COLORS
from evolutionary_extensions.persistence.packaging import atomic_json, hashes

from .protocol import search_targets
from .reporting import read_json


def export_figures(search_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(search_root).resolve() / "reports"
    manifest = read_json(root / "export_manifest.json")
    path = root / "robustness.json"
    if (
        manifest.get("status") != "numeric_exports_complete"
        or hashes(path)["sha256"] != manifest["files"][path.name]
    ):
        raise ValueError("Numeric robustness exports are not verified")
    data = read_json(path)
    directory = root / "figures"
    directory.mkdir(exist_ok=True)
    figures, sources = {}, {}
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 9, "svg.fonttype": "none"}
    )

    def save(figure, name, rows):
        figure.tight_layout()
        for suffix in ("png", "pdf", "svg"):
            destination = directory / f"{name}.{suffix}"
            figure.savefig(destination, dpi=180, bbox_inches="tight")
            figures[destination.name] = hashes(destination)["sha256"]
        plt.close(figure)
        sources[name] = rows

    targets = search_targets(data.get("evidence", {}))
    rows = data["source_target_transfer"]
    lookup = {(r["cohort"].split("/")[1], r["detector"]): r for r in rows}
    if set(lookup) != {(t, d) for t in targets for d in DETECTORS}:
        raise ValueError("Incomplete source-to-target transfer matrix")
    values = np.array(
        [[lookup[t, d]["not_detected_rate"] for d in DETECTORS] for t in targets]
    )
    figure, ax = plt.subplots(figsize=(9, 6))
    color = ax.imshow(values, cmap="Blues", vmin=0, vmax=1)
    for i, target in enumerate(targets):
        for j, detector in enumerate(DETECTORS):
            r = lookup[target, detector]
            ax.text(
                j,
                i,
                f"{r['not_detected']}/{r['n_subjects']}\n{values[i, j]:.0%}",
                ha="center",
                va="center",
                color="white" if values[i, j] > 0.6 else "black",
            )
    ax.set(
        xticks=range(5),
        xticklabels=DETECTORS,
        yticks=range(len(targets)),
        yticklabels=targets,
        xlabel="Evaluated with",
        ylabel="Optimized against",
        title="Not-detected decisions · selected images · noise 0 · PNG",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    figure.colorbar(color, ax=ax, label="Not-detected rate at frozen PNG thresholds")
    save(figure, "source-target-transfer", rows)

    cohorts = sorted({r["cohort"] for r in data["rates"]})

    def panels(title):
        fig, axes = plt.subplots(
            math.ceil(len(cohorts) / 2),
            2,
            figsize=(11, 3 * math.ceil(len(cohorts) / 2)),
            squeeze=False,
        )
        fig.suptitle(title, y=1.01)
        for ax in axes.flat[len(cohorts) :]:
            ax.set_visible(False)
        return fig, list(axes.flat)

    for codec in ("png", "jpeg95", "jpeg75"):
        rows = [r for r in data["unseen_noise_averages"] if r["codec"] == codec]
        fig, axes = panels(
            f"Unseen noise seeds 1–3 · {codec} · subject bootstrap 95% CI"
        )
        for ax, name in zip(axes, cohorts):
            selected = {r["detector"]: r for r in rows if r["cohort"] == name}
            for i, detector in enumerate(DETECTORS):
                r = selected[detector]
                value = r["mean_not_detected_rate"]
                low, high = r["subject_bootstrap_ci95"]
                ax.hlines(i, low, high, color=COLORS[i])
                ax.plot(value, i, "o", color=COLORS[i])
            ax.set(
                yticks=range(5),
                yticklabels=DETECTORS,
                xlim=(-0.03, 1.03),
                title=f"{name} · n={next(iter(selected.values()))['n_subjects']} subjects",
                xlabel="Mean not-detected rate",
            )
            ax.invert_yaxis()
        save(fig, f"unseen-noise-{codec}", rows)

    rows = [
        r
        for r in data["paired_differences"]
        if r["kind"] == "compressed minus PNG" and r["diffusion_seed"] == 0
    ]
    fig, axes = panels(
        "Compression minus PNG · noise 0 · paired subject bootstrap 95% CI"
    )
    for ax, name in zip(axes, cohorts):
        for row in (r for r in rows if r["cohort"] == name):
            i = DETECTORS.index(row["detector"])
            offset = -0.12 if row["codec"] == "jpeg95" else 0.12
            value = row["not_detected_rate_difference"]
            low, high = row["difference_ci95"]
            ax.hlines(i + offset, low, high, color=COLORS[i])
            ax.plot(value, i + offset, "o" if offset < 0 else "s", color=COLORS[i])
        ax.axvline(0, color=".5", linestyle="--", linewidth=1)
        ax.set(
            yticks=range(5),
            yticklabels=DETECTORS,
            xlim=(-1.03, 1.03),
            title=name,
            xlabel="Change in not-detected rate · ○ JPEG95 / □ JPEG75",
        )
        ax.invert_yaxis()
    save(fig, "compression-differences", rows)

    rows = data["ssp_patch_fpr_drift"]
    if len(rows) != 3:
        raise ValueError("Missing unseen-patch FPR calibration")
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, row in enumerate(rows):
        value, (low, high) = row["false_positive_rate"], row["false_positive_ci95"]
        ax.vlines(i, low, high, color=COLORS[2])
        ax.plot([i, i], [low, high], "_", color=COLORS[2])
        ax.plot(i, value, "o", color=COLORS[2])
    ax.axhline(0.05, color=".5", linestyle="--", label="Nominal calibration FPR 5%")
    ax.set(
        xticks=range(3),
        xticklabels=[str(r["patch_seed"]) for r in rows],
        ylim=(-0.03, 1.03),
        xlabel="Unseen SSP patch seed",
        ylabel="Calibration false-positive rate",
        title="Unchanged PNG threshold · 200 paired real photos · Wilson 95% CI",
    )
    ax.legend()
    save(fig, "ssp-patch-fpr-drift", rows)
    atomic_json(directory / "figure_data.json", sources)
    atomic_json(
        directory / "manifest.json",
        {
            "status": "figures_exported",
            "numeric_source_sha256": hashes(path)["sha256"],
            "figure_data_sha256": hashes(directory / "figure_data.json")["sha256"],
            "files": figures,
            "visual_audit_pending": True,
        },
    )
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    export_figures(parser.parse_args().root)


if __name__ == "__main__":
    main()
