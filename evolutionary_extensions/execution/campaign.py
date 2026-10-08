"""Execute a finite notebook campaign through an existing managed Jupyter server.

This controller owns independent kernels and finalizes notebooks after kernel idle.
An external lifecycle owner must stop cloud compute even if this process fails.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import shutil
import time
from pathlib import Path

import aiohttp
import nbformat

from evolutionary_extensions.execution.finalization import (
    BackgroundFinalizer,
)
from evolutionary_extensions.execution.finalization import (
    finalize_run as freeze_run,
)
from evolutionary_extensions.execution.notebooks import ManagedKernelPool, execute_cells
from evolutionary_extensions.paths import checkout_root
from evolutionary_extensions.persistence.google_drive import (
    GoogleDrivePersistor,
)
from evolutionary_extensions.persistence.packaging import atomic_json, package


class SkipTrial(Exception):
    """A before_trial callback may decline a trial without failing its siblings."""

    def __init__(self, reason):
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("SkipTrial requires a nonempty reason")
        self.reason = reason
        self._from_before_trial = False
        super().__init__(reason)


def campaign_finalization_deadline(campaign, initial_deadline):
    """Finalization shares the core clock unless a later deadline is supplied."""
    deadline = campaign.get("finalization_deadline_unix", initial_deadline)
    if deadline is not None and (
        isinstance(deadline, bool)
        or not isinstance(deadline, (int, float))
        or not math.isfinite(deadline)
    ):
        raise ValueError("finalization_deadline_unix must be finite or null")
    if initial_deadline is not None and (
        deadline is None or deadline < initial_deadline
    ):
        raise ValueError("finalization_deadline_unix must be >= initial_deadline_unix")
    return deadline


def require_campaign_space(experiment, pending_bytes=0, *, estimated_bytes):
    """Reserve raw output and its future ZIP, plus all outstanding work."""
    next_run = 2 * estimated_bytes
    reserve = experiment.get("minimum_free_gib", 20) * 1024**3
    output_path = Path(experiment["output_root"]).expanduser()
    while not output_path.exists():
        output_path = output_path.parent
    for location in (experiment["output_root"], experiment["cache_dir"]):
        path = Path(location).expanduser()
        while not path.exists():
            path = path.parent
        # Bulk output/ZIP reservation belongs to its actual filesystem. The
        # model cache on a separate filesystem still retains the full reserve.
        needed = reserve + (
            next_run + pending_bytes
            if path.stat().st_dev == output_path.stat().st_dev
            else 0
        )
        if shutil.disk_usage(path).free < needed:
            raise OSError(
                "Insufficient space for the next run and finalization backlog"
            )


def default_adapter():
    from evolutionary_extensions.experiments.prompt_embedding.adapter import (
        PromptEmbeddingExperimentAdapter,
    )

    return PromptEmbeddingExperimentAdapter()


def finalize_run(run_dir, notebook, *, adapter=None):
    """Finalize using the selected recipe's artifact validation contract."""
    return freeze_run(
        run_dir,
        notebook,
        validate_artifacts=(adapter or default_adapter()).validate_artifacts,
    )


def resolve_campaign_seeds(campaign):
    """Random seeds are resolved once; explicit repeats remain valid for replay tests."""
    import secrets

    seeds = campaign.get("seeds")
    if seeds is None:
        count = campaign.get("runs", 3)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("campaign.runs must be a positive integer")
        seeds = []
        while len(seeds) < count:
            seed = secrets.randbits(32)
            if seed not in seeds:
                seeds.append(seed)
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(
            isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32
            for seed in seeds
        )
    ):
        raise ValueError(
            "campaign.seeds must be a nonempty list of integer NumPy seeds"
        )
    return seeds


def campaign_deadline(campaign):
    """None explicitly disables the campaign clock; per-request timeouts remain."""
    deadline = campaign.get("initial_deadline_unix")
    if deadline is not None:
        if (
            isinstance(deadline, bool)
            or not isinstance(deadline, (int, float))
            or not math.isfinite(deadline)
        ):
            raise ValueError("initial_deadline_unix must be finite or null")
        return deadline
    budget = campaign.get("initial_budget_seconds", 3600)
    if budget is None:
        return None
    if (
        isinstance(budget, bool)
        or not isinstance(budget, (int, float))
        or not math.isfinite(budget)
        or budget <= 0
    ):
        raise ValueError("initial_budget_seconds must be positive, finite or null")
    return time.time() + budget


def campaign_reproducibility(results, trials):
    """Only identical recipes are repetitions; paired variant seeds are not repeats."""
    groups = {}
    for result in results:
        index = int(result["run_id"].split("-")[1]) - 1
        settings = {k: v for k, v in trials[index]["experiment"].items() if k != "seed"}
        key = json.dumps(settings, sort_keys=True)
        groups.setdefault(key, {}).setdefault(result["seed"], []).append(
            result["scientific_hashes"]
        )
    repeated = [
        values
        for group in groups.values()
        for values in group.values()
        if len(values) > 1
    ]
    reproducibility = (
        all(all(value == values[0] for value in values) for values in repeated)
        if repeated
        else None
    )
    different_seeds = [group for group in groups.values() if len(group) > 1]
    distinct = (
        all(
            len({values[0]["embeddings"] for values in group.values()}) == len(group)
            for group in different_seeds
        )
        if different_seeds
        else None
    )
    return reproducibility, distinct


def prepare_campaign(config_path, output, *, adapter=None):
    """Freeze a disabled recipe and its seeds without sessions, models or inference."""
    config = json.loads(Path(config_path).read_text())
    config["campaign"]["seeds"] = resolve_campaign_seeds(config["campaign"])
    adapter = adapter or default_adapter()
    trials = adapter.resolve_trials(config, config["campaign"]["seeds"])
    campaign_finalization_deadline(
        config["campaign"], campaign_deadline(config["campaign"])
    )
    config["execution"]["enabled"] = False
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config["execution"]["campaign_root"] = str(output)
    atomic_json(output / "resolved_config.json", config)
    atomic_json(
        output / "campaign_manifest.json",
        {
            "status": "prepared",
            "execution_enabled": False,
            "seeds": config["campaign"]["seeds"],
            "experiment": config["experiment"],
            "trials": trials,
        },
    )
    return output / "resolved_config.json"


async def run_campaign(
    config_path, *, adapter=None, before_trial=None, after_trial=None
):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if not config["execution"].get("enabled", False):
        raise ValueError(
            "Campaign execution is disabled; enable explicitly after authorization"
        )
    seeds = resolve_campaign_seeds(config["campaign"])
    config["campaign"]["seeds"] = seeds
    initial_deadline = campaign_deadline(config["campaign"])
    finalization_deadline = campaign_finalization_deadline(
        config["campaign"], initial_deadline
    )
    adapter = adapter or default_adapter()
    trials = adapter.resolve_trials(config, seeds)
    repo = checkout_root()
    experiment = {**config.get("deployment", {}), **config["experiment"]}
    for name in ("output_root", "bounds_file", "cache_dir", "initial_embeddings_file"):
        if experiment.get(name) is None:
            continue
        path = Path(experiment[name]).expanduser()
        experiment[name] = str(
            (repo / path).resolve() if not path.is_absolute() else path.resolve()
        )
    run_root = Path(experiment["output_root"])
    if not config["execution"].get("campaign_root"):
        from uuid import uuid4

        run_root /= (
            time.strftime("campaign-%Y%m%dT%H%M%SZ", time.gmtime())
            + "-"
            + uuid4().hex[:8]
        )
        experiment["output_root"] = str(run_root)
    campaign_root = (
        Path(config["execution"].get("campaign_root") or run_root / "_campaign")
        .expanduser()
        .resolve()
    )
    if (campaign_root / "managed_sessions.json").exists():
        raise ValueError(
            "Choose a fresh campaign_root; recover uploads without rerunning inference"
        )
    campaign_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    reports = campaign_root / "reports"
    reports.mkdir(exist_ok=True)
    runner_count = config["execution"].get("runner_count", 1)
    if (
        isinstance(runner_count, bool)
        or not isinstance(runner_count, int)
        or runner_count < 1
    ):
        raise ValueError("runner_count must be a positive integer")
    adapter.validate_execution(experiment, runner_count)
    runner_count = min(runner_count, len(seeds))
    currents = [
        campaign_root
        / ("current.ipynb" if runner_count == 1 else f"runner-{i + 1}.ipynb")
        for i in range(runner_count)
    ]
    template_path = repo / config["execution"]["notebook"]
    notebook = nbformat.read(template_path, as_version=4)
    adapter.validate_template(notebook)
    parameters = [
        i
        for i, c in enumerate(notebook.cells)
        if "parameters" in c.metadata.get("tags", [])
    ]
    if len(parameters) != 1:
        raise ValueError("Notebook requires exactly one parameters cell")
    for current in currents:
        nbformat.write(notebook, current)
    atomic_json(
        campaign_root / "campaign_manifest.json",
        {
            "status": "resolved",
            "execution_enabled": True,
            "seeds": seeds,
            "experiment": experiment,
            "trials": trials,
            "initial_deadline_unix": initial_deadline,
            "finalization_deadline_unix": finalization_deadline,
            "runner_count": runner_count,
        },
    )
    status_file = campaign_root / "campaign_status.json"
    results, transfers, skipped_trials = [], [], []
    finalizer = BackgroundFinalizer(limit=max(2, runner_count + 1))
    admission_lock = asyncio.Lock()
    active = {}

    def status(stage, **details):
        completed = [
            j["future"].result()
            for j in finalizer.jobs
            if j["future"].done()
            and not j["future"].cancelled()
            and j["future"].exception() is None
        ]
        transfers[:] = [
            item["transfer"] for item in completed if item["transfer"] is not None
        ]
        atomic_json(
            status_file,
            {
                "stage": stage,
                "updated_unix": time.time(),
                "initial_deadline_unix": initial_deadline,
                "finalization_deadline_unix": finalization_deadline,
                "runner_count": runner_count,
                "active_runs": active,
                "results": results,
                "skipped_trials": skipped_trials,
                "transfers": transfers,
                "finalization_queue": finalizer.summary(),
                **details,
            },
        )

    status("preflight")
    drive_config = config["drive"]
    persistor = None
    if drive_config["enabled"]:
        persistor = GoogleDrivePersistor(
            drive_config["credentials"],
            drive_config["folder_id"],
            Path(
                config["execution"].get("archive_root")
                or campaign_root / "archive-queue"
            ),
            campaign_root / "receipts",
            run_root,
            deadline_unix=finalization_deadline,
        )
        persistor.check_destination()
        # Tiny Azure-originated verification, with deletion deliberately disabled.
        tiny = run_root / "transfer-check"
        tiny.mkdir(exist_ok=False)
        (tiny / "data.txt").write_text("Experiment transfer verification\n")
        atomic_json(
            tiny / "experiment_complete.json",
            {"status": "complete", "purpose": "transfer preflight"},
        )
        smoke = persistor.persist(tiny, delete_after_verification=False)
        atomic_json(campaign_root / "drive_preflight.json", smoke)
    status("connecting")
    base_url = config["execution"]["jupyter_url"].rstrip("/")
    jupyter_root = (
        Path(config["execution"].get("jupyter_root") or repo).expanduser().resolve()
    )
    managed_paths = [str(current.relative_to(jupyter_root)) for current in currents]
    async with aiohttp.ClientSession(
        cookie_jar=aiohttp.CookieJar(unsafe=True),
        timeout=aiohttp.ClientTimeout(total=180),
    ) as http:
        async with http.get(f"{base_url}/lab") as response:
            await response.read()
            cookie_url = response.url
        cookies = http.cookie_jar.filter_cookies(cookie_url)
        headers = {"X-XSRFToken": cookies["_xsrf"].value} if "_xsrf" in cookies else {}
        async with ManagedKernelPool(
            http,
            base_url,
            headers,
            managed_paths,
            config["execution"]["kernel_name"],
            campaign_root,
        ) as runners:

            async def trial(
                index, seed, overrides=None, deadline=initial_deadline, runner_index=0
            ):
                params = {
                    **experiment,
                    **trials[index - 1]["overrides"],
                    **(overrides or {}),
                    "seed": seed,
                    "run_id": f"trial-{index}-seed-{seed}",
                    "deadline_unix": deadline,
                }
                if before_trial is not None:
                    try:
                        params = await before_trial(index, params, results, status)
                    except SkipTrial as error:
                        error._from_before_trial = True
                        raise
                    adapter.validate_parameters(params)
                    trials[index - 1]["experiment"] = {
                        k: v
                        for k, v in params.items()
                        if k not in {"seed", "run_id", "deadline_unix"}
                    }
                await finalizer.wait_for_slot(finalization_deadline)
                while True:
                    async with admission_lock:
                        outstanding = sum(
                            2 * adapter.estimate_run_bytes(item)
                            for item in active.values()
                        )
                        try:
                            require_campaign_space(
                                params,
                                finalizer.reserved_bytes + outstanding,
                                estimated_bytes=adapter.estimate_run_bytes(params),
                            )
                        except OSError:
                            recoverable = bool(active) or any(
                                not job["future"].done() for job in finalizer.jobs
                            )
                            if not recoverable:
                                raise
                        else:
                            active[str(runner_index + 1)] = params
                            break
                    status("waiting_for_space", active_run=params["run_id"])
                    finalizer.check()
                    if deadline is not None and time.time() >= deadline - 120:
                        raise TimeoutError("Storage admission deadline reached")
                    await asyncio.sleep(0.1)
                status(
                    "executing",
                    active_run=params["run_id"],
                    active_seed=seed,
                    deadline_unix=deadline,
                )
                runner = runners[runner_index]
                session_id, kernel_id = runner["session_id"], runner["kernel_id"]
                current = currents[runner_index]
                copied = nbformat.read(template_path, as_version=4)
                copied.cells[parameters[0]].source = adapter.parameters(params)
                nbformat.write(copied, current)
                await execute_cells(
                    http,
                    base_url,
                    headers,
                    kernel_id,
                    session_id,
                    copied,
                    current,
                    deadline,
                )
                run_dir = run_root / params["run_id"]
                report = reports / params["run_id"]
                report.mkdir(exist_ok=False)
                # Freeze this notebook before current.ipynb is reused by the next trial.
                saved_notebook = report / "experiment.executed.ipynb"
                shutil.copyfile(current, saved_notebook)
                result = json.loads((run_dir / "result.json").read_text())
                result["trial_label"] = trials[index - 1]["label"]
                if after_trial is not None:
                    await after_trial(index, params, result, run_dir, report)
                source_bytes = sum(
                    p.stat().st_size for p in run_dir.rglob("*") if p.is_file()
                )
                source_bytes += saved_notebook.stat().st_size + 65536
                results.append(result)
                atomic_json(campaign_root / "results.json", results)

                def finish():
                    progress_path = report / "finalization.json"

                    def stage(name, **details):
                        atomic_json(
                            progress_path,
                            {
                                "stage": name,
                                "run_id": params["run_id"],
                                "updated_unix": time.time(),
                                **details,
                            },
                        )

                    try:
                        stage("checking_artifacts")
                        finalize_run(run_dir, saved_notebook, adapter=adapter)
                        for name in adapter.report_files:
                            shutil.copyfile(run_dir / name, report / name)
                        for name in adapter.report_directories:
                            shutil.copytree(run_dir / name, report / name)
                        archive_root = Path(
                            config["execution"].get("archive_root")
                            or campaign_root / "archive-queue"
                        )
                        archive_root.mkdir(parents=True, exist_ok=True)
                        if (
                            shutil.disk_usage(archive_root).free
                            < source_bytes
                            + experiment.get("minimum_free_gib", 20) * 1024**3
                        ):
                            raise OSError(
                                "Insufficient archive space; retain completed run"
                            )
                        output_bytes = sum(
                            p.stat().st_size for p in run_dir.rglob("*") if p.is_file()
                        )
                        stage("packaging_and_uploading" if persistor else "packaging")
                        if persistor:
                            persistor.client.deadline_unix = finalization_deadline
                            transfer = persistor.persist(
                                run_dir,
                                delete_after_verification=drive_config[
                                    "delete_after_verification"
                                ],
                                on_stage=stage,
                            )
                            atomic_json(report / "drive_receipt.json", transfer)
                        else:
                            manifest = package(
                                run_dir, archive_root / f"{params['run_id']}.zip"
                            )
                            atomic_json(
                                report / "local_archive.json",
                                {
                                    k: manifest[k]
                                    for k in ("archive", "bytes", "md5", "sha256")
                                },
                            )
                            transfer = None
                        stage(
                            "complete", verified=bool(transfer and transfer["verified"])
                        )
                        return {"transfer": transfer, "output_bytes": output_bytes}
                    except BaseException as error:
                        stage("failed", error_type=type(error).__name__)
                        raise

                async with admission_lock:
                    await finalizer.wait_for_slot(finalization_deadline)
                    finalizer.submit(params["run_id"], finish, source_bytes)
                    active.pop(str(runner_index + 1))
                status(
                    "trial_compute_complete",
                    active_run=params["run_id"],
                    deadline_unix=deadline,
                )

            async def drain_finalization(deadline):
                status("draining_finalization", deadline_unix=deadline)
                completed = await finalizer.drain(deadline)
                transfers[:] = [
                    item["transfer"]
                    for item in completed
                    if item["transfer"] is not None
                ]
                by_run = {
                    job["run_id"]: item for job, item in zip(finalizer.jobs, completed)
                }
                for result in results:
                    if result["run_id"] in by_run:
                        result["output_bytes"] = by_run[result["run_id"]][
                            "output_bytes"
                        ]
                atomic_json(campaign_root / "results.json", results)

            try:
                if config["execution"].get("warm_runners", False):
                    for runner_index, runner in enumerate(runners):
                        warm = adapter.warmup(
                            notebook,
                            dict(
                                experiment,
                                seed=seeds[0],
                                deadline_unix=initial_deadline,
                            ),
                            campaign_root / f"preflight-{runner_index + 1}",
                        )
                        await execute_cells(
                            http,
                            base_url,
                            headers,
                            runner["kernel_id"],
                            runner["session_id"],
                            warm,
                            campaign_root / f"warmup-{runner_index + 1}.ipynb",
                            initial_deadline,
                        )
                queue = asyncio.Queue()
                for i, seed in enumerate(config["campaign"]["seeds"], 1):
                    queue.put_nowait((i, seed))

                async def worker(runner_index):
                    while not queue.empty():
                        i, seed = queue.get_nowait()
                        try:
                            await trial(i, seed, runner_index=runner_index)
                        except SkipTrial as error:
                            if not error._from_before_trial:
                                raise
                            skipped = {
                                "index": i,
                                "label": trials[i - 1]["label"],
                                "seed": seed,
                                "reason": error.reason,
                                "skipped_unix": time.time(),
                            }
                            skipped_trials.append(skipped)
                            atomic_json(
                                campaign_root / "skipped_trials.json", skipped_trials
                            )
                            status("trial_skipped", skipped_trial=skipped)

                async with asyncio.TaskGroup() as tasks:
                    for runner_index in range(runner_count):
                        tasks.create_task(worker(runner_index))
                results.sort(key=lambda r: int(r["run_id"].split("-")[1]))
                finalizer.jobs.sort(key=lambda j: int(j["run_id"].split("-")[1]))
                await drain_finalization(finalization_deadline)
                reproducibility, distinct = campaign_reproducibility(results, trials)
                accepted = (
                    bool(results)
                    and all(r["validation_passed"] for r in results)
                    and reproducibility is not False
                    and distinct is not False
                )
                summary = {
                    "status": (
                        "passed"
                        if accepted
                        else "all_trials_skipped"
                        if len(skipped_trials) == len(trials)
                        else "validation_incomplete"
                    ),
                    "initial_trials_passed": accepted,
                    "same_seed_identical": reproducibility,
                    "different_seed_distinct": distinct,
                    "results": results,
                    "trials": trials,
                    "skipped_trials": skipped_trials,
                    "transfers": transfers,
                    "finished_unix": time.time(),
                }
                atomic_json(campaign_root / "campaign_summary.json", summary)
                (campaign_root / "REPORT.md").write_text(
                    adapter.campaign_report(summary)
                )
                status("complete", summary=summary)
            except BaseException as error:
                status("failed", error=f"{type(error).__name__}: {error}")
                raise
            finally:
                finalizer.close()


if __name__ == "__main__":
    # Configure only this dedicated CPU controller, before any inference imports.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    campaign = sub.add_parser("campaign")
    campaign.add_argument("--config", type=Path, required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--run", type=Path, required=True)
    finalize.add_argument("--notebook", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "campaign":
        asyncio.run(run_campaign(args.config))
    elif args.command == "prepare":
        print(prepare_campaign(args.config, args.output))
    else:
        print(json.dumps(finalize_run(args.run, args.notebook), indent=2))
