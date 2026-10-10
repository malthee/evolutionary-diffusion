"""Sequential stages on the existing managed Jupyter server.

An external lifecycle owner starts/stops Azure and must run this controller to
completion or preserve failure evidence. No inference deadline is imposed.
"""

import argparse
import asyncio
import json
import subprocess
import time
from pathlib import Path

import aiohttp
import nbformat

from evolutionary_extensions.execution.notebooks import ManagedKernelPool, execute_cells
from evolutionary_extensions.persistence.packaging import atomic_json, hashes

from .config import DETECTORS, load_config
from .runner import assets_for, freeze_inputs, guard_disk, status


async def execute(config_path):
    config = load_config(config_path)
    freeze_inputs(config)
    assets_for(config)
    guard_disk(config)
    settings = config["managed"]
    root = Path(config["output"])
    repo = Path(settings["repository"]).resolve()
    jupyter_root = Path(settings["jupyter_root"]).resolve()
    stage_root = root / "notebooks"
    stage_root.mkdir(exist_ok=True)
    deployment = repo / "deployment_provenance.json"
    if deployment.exists():
        provenance = json.loads(deployment.read_text())
        for filename, expected in provenance["files"].items():
            path = repo / filename
            if (
                not path.resolve().is_relative_to(repo)
                or hashes(path)["sha256"] != expected
            ):
                raise ValueError("Frozen deployed source checksum differs")
        source_commit = provenance["source_commit"]
    else:
        source_commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    manifest = {
        "source_commit": source_commit,
        "files": {
            str(p.relative_to(repo)): hashes(p)["sha256"]
            for p in (
                repo / "evolutionary_extensions/experiments/detector_evaluation"
            ).rglob("*.py")
        },
        "started_unix": time.time(),
        "stages": ["generate", *DETECTORS],
        "deadline": None,
        "runners": 1,
        "generation_batch": 4,
        "lifecycle_owner": settings["lifecycle_owner"],
    }
    atomic_json(root / "managed_manifest.json", manifest)
    base = settings["jupyter_url"].rstrip("/")
    async with aiohttp.ClientSession(
        cookie_jar=aiohttp.CookieJar(unsafe=True),
        timeout=aiohttp.ClientTimeout(total=180),
    ) as http:
        async with http.get(f"{base}/lab") as response:
            if response.status != 200:
                raise RuntimeError("Existing managed Jupyter server is unavailable")
            await response.read()
            cookies = http.cookie_jar.filter_cookies(response.url)
        headers = {"X-XSRFToken": cookies["_xsrf"].value} if "_xsrf" in cookies else {}
        for stage in ("generate", *DETECTORS):
            guard_disk(config)
            path = stage_root / f"{stage}.ipynb"
            notebook = nbformat.v4.new_notebook(
                cells=[
                    nbformat.v4.new_markdown_cell(
                        f"# Detector evaluation: {stage}\nFrozen five-detector recipe. Smoke/profile scores are excluded from primary scores."
                    ),
                    nbformat.v4.new_code_cell(
                        "import os, sys\n"
                        "os.environ['NVIDIA_TF32_OVERRIDE'] = '0'\n"
                        "os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'\n"
                        "os.environ['HF_HUB_OFFLINE'] = '1'\n"
                        "os.environ['TRANSFORMERS_OFFLINE'] = '1'\n"
                        f"sys.path.insert(0, {str(repo)!r})\n"
                        "from evolutionary_extensions.experiments.detector_evaluation.runner import main\n"
                        f"main([{stage!r}, '--config', {str(Path(config_path).resolve())!r}])"
                    ),
                ]
            )
            notebook.metadata["kernelspec"] = {
                "name": settings["kernels"][stage],
                "display_name": f"Detector evaluation {stage}",
                "language": "python",
            }
            nbformat.write(notebook, path)
            evidence = root / "sessions" / f"{stage}-{int(time.time())}"
            async with ManagedKernelPool(
                http,
                base,
                headers,
                [str(path.relative_to(jupyter_root))],
                settings["kernels"][stage],
                evidence,
            ) as runners:
                runner = runners[0]
                await execute_cells(
                    http,
                    base,
                    headers,
                    runner["kernel_id"],
                    runner["session_id"],
                    notebook,
                    path,
                    None,
                )
    from .analysis import analyze
    from .report import export_report

    analysis = analyze(config)
    export_report(config)
    atomic_json(
        root / "inference_complete.json",
        {
            "status": "complete",
            "scores": analysis["primary_score_count"],
            "finished_unix": time.time(),
        },
    )
    status(
        root,
        "inference_complete",
        primary_scores=analysis["primary_score_count"],
        interactive_report_pending=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    try:
        asyncio.run(execute(args.config))
    except BaseException as error:
        config = load_config(args.config)
        status(
            config["output"],
            "managed_failed",
            failed=True,
            error_type=type(error).__name__,
            error=str(error),
        )
        raise


if __name__ == "__main__":
    main()
