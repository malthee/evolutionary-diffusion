"""Sequential target notebooks on the existing Azure-managed Jupyter server."""

import argparse
import asyncio
import json
import shutil
import time
from pathlib import Path

import aiohttp
import nbformat

from evolutionary_extensions.execution.notebooks import ManagedKernelPool, execute_cells
from evolutionary_extensions.experiments.detector_evaluation.config import DETECTORS
from evolutionary_extensions.persistence.packaging import atomic_json, hashes

from .optimizer import optimizer_source_files
from .protocol import freeze_heldout, heldout_plan, search_targets
from .runner import verify_prerequisite


def write_notebook(notebook, path):
    """Keep prior executed evidence when a controller resumes a stage."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = hashes(path)["sha256"]
        archive = path.parent / "history" / f"{path.stem}-{previous}.ipynb"
        archive.parent.mkdir(exist_ok=True)
        if archive.exists() and hashes(archive)["sha256"] != previous:
            raise ValueError("Executed notebook history checksum differs")
        if not archive.exists():
            shutil.copyfile(path, archive)
    nbformat.write(notebook, path)


async def execute(config_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    _, rows, _, completion = verify_prerequisite(config)
    targets = search_targets(config)
    budget_total = 6400 * len(targets)
    settings = config["managed"]
    if not {"generate", *DETECTORS}.issubset(settings["kernels"]):
        raise ValueError(
            "Require isolated search kernels for generation and all five detectors"
        )
    root = Path(config["output"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    repo = Path(settings["repository"]).resolve()
    jupyter_root = Path(settings["jupyter_root"]).resolve()
    atomic_json(
        root / "managed_manifest.json",
        {
            "benchmark_archive_sha256": completion["archive_sha256"],
            "targets": list(targets),
            "fresh_candidates_per_target": 6400,
            "fresh_candidates_total": budget_total,
            "population": 64,
            "generation_batch": 4,
            "generator_validation_images_per_kernel": 192,
            "generator_validation_kernels": ["generate", *DETECTORS],
            "deadline": None,
            "runners": 1,
            "started_unix": time.time(),
            "source_files": {
                **optimizer_source_files(),
                **{
                    str(p.relative_to(repo)): hashes(p)["sha256"]
                    for p in (
                        repo / "evolutionary_extensions/experiments/detector_search"
                    ).glob("*.py")
                },
            },
            "lifecycle_owner": settings["lifecycle_owner"],
        },
    )
    base = settings["jupyter_url"].rstrip("/")
    async with aiohttp.ClientSession(
        cookie_jar=aiohttp.CookieJar(unsafe=True),
        timeout=aiohttp.ClientTimeout(total=180),
    ) as http:
        async with http.get(base + "/lab") as response:
            if response.status != 200:
                raise RuntimeError("Existing managed Jupyter server is unavailable")
            await response.read()
            cookies = http.cookie_jar.filter_cookies(response.url)
        headers = {"X-XSRFToken": cookies["_xsrf"].value} if "_xsrf" in cookies else {}
        # All stacks pass the same pixels before any optimizer receives feedback.
        for stage in ("generate", *DETECTORS):
            path = root / "notebooks" / f"generator-validation-{stage}.ipynb"
            notebook = nbformat.v4.new_notebook(
                cells=[
                    nbformat.v4.new_markdown_cell(
                        f"# Shared generator validation: {stage}\n192 diagnostic renders; zero optimizer candidates."
                    ),
                    nbformat.v4.new_code_cell(
                        "import os, sys\n"
                        "os.environ['NVIDIA_TF32_OVERRIDE'] = '0'\n"
                        "os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'\n"
                        "os.environ['HF_HUB_OFFLINE'] = '1'\n"
                        "os.environ['TRANSFORMERS_OFFLINE'] = '1'\n"
                        f"sys.path.insert(0, {str(repo)!r})\n"
                        "from evolutionary_extensions.experiments.detector_search.generator_validation import run_validation\n"
                        f"run_validation({str(config_path)!r}, {stage!r})"
                    ),
                ]
            )
            notebook.metadata["kernelspec"] = {
                "name": settings["kernels"][stage],
                "display_name": f"Shared generator {stage}",
                "language": "python",
            }
            write_notebook(notebook, path)
            async with ManagedKernelPool(
                http,
                base,
                headers,
                [str(path.relative_to(jupyter_root))],
                settings["kernels"][stage],
                root / "sessions" / f"generator-validation-{stage}-{time.time_ns()}",
            ) as workers:
                worker = workers[0]
                await execute_cells(
                    http,
                    base,
                    headers,
                    worker["kernel_id"],
                    worker["session_id"],
                    notebook,
                    path,
                    None,
                )
            marker = json.loads(
                (root / "generator_validation" / stage / "parity.json").read_text()
            )
            if (
                marker.get("passed") is not True
                or marker.get("stage") != stage
                or marker.get("known_physical_images") != 192
                or marker.get("candidate_budget_used") != 0
            ):
                raise ValueError("Shared generator remains unvalidated")
        for target in targets:
            path = root / "notebooks" / f"{target}.ipynb"
            path.parent.mkdir(exist_ok=True)
            notebook = nbformat.v4.new_notebook(
                cells=[
                    nbformat.v4.new_markdown_cell(
                        f"# sep-CMA-ES target: {target}\n6,400 fresh PNG fitness candidates; held-out measurements never enter tell()."
                    ),
                    nbformat.v4.new_code_cell(
                        "import os, sys\n"
                        "os.environ['NVIDIA_TF32_OVERRIDE'] = '0'\n"
                        "os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'\n"
                        "os.environ['HF_HUB_OFFLINE'] = '1'\n"
                        "os.environ['TRANSFORMERS_OFFLINE'] = '1'\n"
                        f"sys.path.insert(0, {str(repo)!r})\n"
                        "from evolutionary_extensions.experiments.detector_search.runner import run_target\n"
                        f"run_target({str(config_path)!r}, {target!r})"
                    ),
                ]
            )
            notebook.metadata["kernelspec"] = {
                "name": settings["kernels"][target],
                "display_name": f"Detector search {target}",
                "language": "python",
            }
            write_notebook(notebook, path)
            async with ManagedKernelPool(
                http,
                base,
                headers,
                [str(path.relative_to(jupyter_root))],
                settings["kernels"][target],
                root / "sessions" / f"{target}-{time.time_ns()}",
            ) as workers:
                worker = workers[0]
                await execute_cells(
                    http,
                    base,
                    headers,
                    worker["kernel_id"],
                    worker["session_id"],
                    notebook,
                    path,
                    None,
                )
            marker = json.loads((root / target / "search_complete.json").read_text())
            if marker.get("status") != "complete" or marker.get("candidates") != 6400:
                raise ValueError("Target did not finish its entire candidate budget")
    finalists = {
        target: json.loads((root / target / "finalists.json").read_text())
        for target in targets
    }
    freeze_heldout(root / "heldout_plan.json", heldout_plan(finalists, rows))
    atomic_json(
        root / "searches_complete.json",
        {
            "status": "searches_complete",
            "candidates": budget_total,
            "heldout_pending": True,
            "finished_unix": time.time(),
        },
    )

    if "aeroblade" not in targets:
        await execute_calibration(config_path, config)
    await execute_heldout(config_path, config)


async def execute_calibration(config_path, config):
    """Calibrate the transfer evaluator without creating optimizer candidates."""
    settings = config["managed"]
    root = Path(config["output"]).resolve()
    repo = Path(settings["repository"]).resolve()
    path = root / "notebooks" / "calibration-aeroblade.ipynb"
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "import os, sys\n"
                "os.environ['NVIDIA_TF32_OVERRIDE']='0'\n"
                "os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'\n"
                "os.environ['HF_HUB_OFFLINE']='1'\n"
                "os.environ['TRANSFORMERS_OFFLINE']='1'\n"
                f"sys.path.insert(0, {str(repo)!r})\n"
                "from evolutionary_extensions.experiments.detector_search.runner import run_target\n"
                f"run_target({str(config_path)!r}, 'aeroblade', calibration_only=True)"
            )
        ]
    )
    notebook.metadata["kernelspec"] = {
        "name": settings["kernels"]["aeroblade"],
        "display_name": "AEROBLADE transfer calibration",
        "language": "python",
    }
    write_notebook(notebook, path)
    base = settings["jupyter_url"].rstrip("/")
    async with aiohttp.ClientSession(
        cookie_jar=aiohttp.CookieJar(unsafe=True),
        timeout=aiohttp.ClientTimeout(total=180),
    ) as http:
        async with http.get(base + "/lab") as response:
            if response.status != 200:
                raise RuntimeError("Managed Jupyter server unavailable")
            await response.read()
            cookies = http.cookie_jar.filter_cookies(response.url)
        headers = {"X-XSRFToken": cookies["_xsrf"].value} if "_xsrf" in cookies else {}
        async with ManagedKernelPool(
            http,
            base,
            headers,
            [str(path.relative_to(Path(settings["jupyter_root"]).resolve()))],
            settings["kernels"]["aeroblade"],
            root / "sessions" / f"calibration-aeroblade-{time.time_ns()}",
        ) as workers:
            worker = workers[0]
            await execute_cells(
                http,
                base,
                headers,
                worker["kernel_id"],
                worker["session_id"],
                notebook,
                path,
                None,
            )
    receipt = json.loads((root / "aeroblade/calibration_complete.json").read_text())
    if (
        receipt.get("status") != "complete"
        or receipt.get("optimizer_candidates") != 0
        or receipt.get("calibration_score_identities") != 400
    ):
        raise ValueError("Transfer calibration incomplete")


async def execute_heldout(config_path, config):
    """Every stage owns one fresh managed kernel and closes it before the next."""
    targets = search_targets(config)
    budget_total = 6400 * len(targets)
    settings = config["managed"]
    root = Path(config["output"]).resolve()
    repo = Path(settings["repository"]).resolve()
    jupyter_root = Path(settings["jupyter_root"]).resolve()
    base = settings["jupyter_url"].rstrip("/")
    async with aiohttp.ClientSession(
        cookie_jar=aiohttp.CookieJar(unsafe=True),
        timeout=aiohttp.ClientTimeout(total=180),
    ) as http:
        async with http.get(base + "/lab") as response:
            if response.status != 200:
                raise RuntimeError(
                    "Managed Jupyter server is unavailable for held-out stages"
                )
            await response.read()
            cookies = http.cookie_jar.filter_cookies(response.url)
        headers = {"X-XSRFToken": cookies["_xsrf"].value} if "_xsrf" in cookies else {}
        for stage in ("generate", *DETECTORS):
            path = root / "notebooks" / f"heldout-{stage}.ipynb"
            call = (
                f"run_generation({str(config_path)!r})"
                if stage == "generate"
                else f"run_detector({str(config_path)!r}, {stage!r})"
            )
            notebook = nbformat.v4.new_notebook(
                cells=[
                    nbformat.v4.new_markdown_cell(
                        f"# Held-out detector robustness: {stage}\nFrozen selection; no optimizer feedback; physical queries outside the configured optimization budget."
                    ),
                    nbformat.v4.new_code_cell(
                        "import os, sys\n"
                        "os.environ['NVIDIA_TF32_OVERRIDE'] = '0'\n"
                        "os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'\n"
                        "os.environ['HF_HUB_OFFLINE'] = '1'\n"
                        "os.environ['TRANSFORMERS_OFFLINE'] = '1'\n"
                        f"sys.path.insert(0, {str(repo)!r})\n"
                        "from evolutionary_extensions.experiments.detector_search.heldout_runner import run_generation, run_detector\n"
                        + call
                    ),
                ]
            )
            notebook.metadata["kernelspec"] = {
                "name": settings["kernels"][stage],
                "display_name": f"Detector held-out {stage}",
                "language": "python",
            }
            write_notebook(notebook, path)
            async with ManagedKernelPool(
                http,
                base,
                headers,
                [str(path.relative_to(jupyter_root))],
                settings["kernels"][stage],
                root / "sessions" / f"heldout-{stage}-{time.time_ns()}",
            ) as workers:
                worker = workers[0]
                await execute_cells(
                    http,
                    base,
                    headers,
                    worker["kernel_id"],
                    worker["session_id"],
                    notebook,
                    path,
                    None,
                )
    plan = json.loads((root / "heldout_plan.json").read_text())
    summaries = {
        target: json.loads((root / "heldout" / f"{target}_complete.json").read_text())
        for target in DETECTORS
    }
    expected_rows = len(plan["variants"])
    matrix, calibration = 0, 0
    for target, marker in summaries.items():
        if marker["status"] != "complete" or marker["detector"] != target:
            raise ValueError("Held-out detector remains incomplete")
        stages = marker["summaries"]
        if (
            len(stages) != (4 if target == "ssp" else 1)
            or stages[0]["matrix_measurements"] != expected_rows
        ):
            raise ValueError("Missing held-out primary matrix scores")
        if target == "ssp" and (
            [s["patch_seed"] for s in stages] != [42, 43, 44, 45]
            or any(
                s["matrix_measurements"] != len(plan["subjects"]) * 3
                or s["calibration_measurements"] != 200
                for s in stages[1:]
            )
        ):
            raise ValueError("Incomplete SSP unseen-patch/calibration factorial")
        matrix += sum(s["matrix_measurements"] for s in stages)
        calibration += sum(s["calibration_measurements"] for s in stages)
    expected = plan["logical_primary_matrix_scores"] + plan["logical_extra_ssp_scores"]
    if matrix != expected or calibration != 600:
        raise ValueError("Held-out inference counts differ from frozen plan")
    atomic_json(
        root / "heldout_inference_complete.json",
        {
            "status": "inference_complete",
            "optimizer_candidates": budget_total,
            "heldout_matrix_scores": matrix,
            "ssp_extra_calibration_scores": calibration,
            "aeroblade_calibration_reused": 400,
            "reporting_pending": True,
            "finished_unix": time.time(),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    asyncio.run(execute(args.config))


if __name__ == "__main__":
    main()
