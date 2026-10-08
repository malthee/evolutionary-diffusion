"""Notebook adaptation; execution/session ownership stays provider-independent."""

import math

import nbformat


class PromptEmbeddingExperimentAdapter:
    @staticmethod
    def parameters(values):
        return f"EXPERIMENT_PARAMETERS = {values!r}\nRUN_EXPERIMENT = True\nCONFIG_PATH = None"

    @staticmethod
    def validate_template(notebook):
        for tag in ("bootstrap", "parameters", "preflight", "optimization"):
            if sum(tag in c.metadata.get("tags", []) for c in notebook.cells) != 1:
                raise ValueError(f"Notebook requires exactly one {tag} cell")

    @staticmethod
    def warmup(template, parameters, directory):
        def cell(tag):
            return next(
                c.source for c in template.cells if tag in c.metadata.get("tags", [])
            )

        return nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_code_cell(cell("bootstrap")),
                nbformat.v4.new_code_cell(
                    "from evolutionary_extensions.experiments.prompt_embedding import ExperimentConfig, Runtime\n"
                    + f"config = ExperimentConfig(**{parameters!r})\nRUN_EXPERIMENT = True\nPREFLIGHT_DIRECTORY = {str(directory)!r}"
                ),
                nbformat.v4.new_code_cell(cell("preflight")),
            ]
        )

    @staticmethod
    def resolve_trials(config, seeds):
        """Freeze optional OSGA controls per trial without altering deployment or seeds."""
        trials = config["campaign"].get("trials")
        if trials is None:
            trials = [{} for _ in seeds]
        if not isinstance(trials, list) or len(trials) != len(seeds):
            raise ValueError("campaign.trials must match the resolved seed count")
        allowed = {"success_ratio", "comparison_factor", "max_selection_pressure"}
        resolved = []
        labels = set()
        for index, (seed, item) in enumerate(zip(seeds, trials), 1):
            if not isinstance(item, dict) or set(item) - {"label", "overrides"}:
                raise ValueError(
                    "Each campaign trial requires label/overrides fields only"
                )
            label = item.get("label", f"trial-{index}")
            if (
                not isinstance(label, str)
                or not label
                or len(label) > 64
                or label in labels
            ):
                raise ValueError(
                    "Trial labels must be distinct nonempty strings up to 64 characters"
                )
            labels.add(label)
            overrides = item.get("overrides", {})
            if not isinstance(overrides, dict) or set(overrides) - allowed:
                raise ValueError(
                    "Trial overrides support only OSGA success/threshold/pressure controls"
                )
            if overrides and config["experiment"]["algorithm"] != "osga":
                raise ValueError("OSGA trial overrides require algorithm osga")
            if any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                for v in overrides.values()
            ):
                raise ValueError("Trial controls must be finite numbers")
            settings = {**config["experiment"], **overrides, "seed": seed}
            from evolutionary.algorithms.ga import OffspringSelectionConfig

            OffspringSelectionConfig(
                settings["success_ratio"],
                settings["comparison_factor"],
                settings["max_selection_pressure"],
            )
            resolved.append(
                {
                    "label": label,
                    "seed": seed,
                    "overrides": overrides,
                    "experiment": settings,
                }
            )
        return resolved

    @staticmethod
    def estimate_run_bytes(experiment):
        embedding_bytes = 2 * (77 * 2048 + 1280)
        image_bytes = 512 * 512 * 4 + 65536
        return (
            experiment["max_evaluations"] * (embedding_bytes + image_bytes + 65536)
            + (experiment["num_generations"] + 3)
            * experiment["population_size"]
            * embedding_bytes
        )

    @staticmethod
    def validate_execution(experiment, runner_count):
        if runner_count > 1 and experiment["candidate_batch_size"] > 4:
            raise ValueError("Multiple runners require candidate_batch_size <= 4")

    @staticmethod
    def validate_parameters(parameters):
        from .config import ExperimentConfig

        ExperimentConfig(**parameters)

    @staticmethod
    def validate_artifacts(run_dir):
        from .artifacts import validate_artifacts

        return validate_artifacts(run_dir)

    report_files = (
        "result.json",
        "config.json",
        "parity.json",
        "environment.json",
        "model_manifest.json",
        "artifact_integrity.json",
        "generation_statistics.csv",
        "stage_timings.csv",
        "operator_statistics.csv",
        "embedding_statistics.csv",
    )
    report_directories = ("plots", "media")

    @staticmethod
    def campaign_report(summary):
        results, transfers = summary["results"], summary["transfers"]
        reproducibility = summary["same_seed_identical"]
        lines = [
            "# Prompt-embedding campaign",
            "",
            f"Status: {summary['status']}; same-seed identity: {reproducibility}.",
            "",
            "| Trial | Seed | Population | Generations | Evaluations | Best fitness | Core seconds | Termination |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for i, r in enumerate(results, 1):
            lines.append(
                f"| {i} | {r['seed']} | {r['population_size']} | {r['completed_generations']} | {r['evaluation_count']} | {r['best_fitness']:.6f} | {r['loop_seconds']:.2f} | {r['termination_reason']} |"
            )
        lines += [
            "",
            "",
            "Pipeline benchmarks are not full campaign completion measurements. Core seconds exclude model setup, parity, final plots and transfers.",
            "",
            "## Verified Drive archives",
            "",
        ]
        for t in transfers:
            lines.append(
                f"- [{t['file_id']}]({t.get('webViewLink') or 'https://drive.google.com/file/d/' + t['file_id'] + '/view'}): {t['bytes']} bytes; SHA-256 `{t['sha256']}`; local run copies deleted: {t['runner_copies_deleted']}."
            )
        return "\n".join(lines) + "\n"
