"""Bounded background jobs and notebook freezing, independent of experiment recipes."""

import asyncio
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nbformat

from evolutionary_extensions.persistence.packaging import atomic_json


class BackgroundFinalizer:
    """One CPU/I/O worker with a bounded queue of immutable run folders."""

    def __init__(self, limit=2):
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("Finalization backlog limit must be positive")
        self.limit = limit
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.jobs = []

    def check(self):
        for job in self.jobs:
            if job["future"].done():
                # Observe failures before starting another trial.
                job["future"].result()

    @property
    def reserved_bytes(self):
        return sum(j["bytes"] for j in self.jobs if not j["future"].done())

    async def wait_for_slot(self, deadline):
        self.check()
        pending = [j for j in self.jobs if not j["future"].done()]
        if len(pending) >= self.limit:
            remaining = None if deadline is None else deadline - time.time()
            if remaining is not None and remaining <= 0:
                raise TimeoutError("Finalization backlog deadline reached")
            async with asyncio.timeout(remaining):
                await asyncio.shield(pending[0]["future"])
        self.check()

    def submit(self, run_id, function, source_bytes):
        self.check()
        if sum(not j["future"].done() for j in self.jobs) >= self.limit:
            raise RuntimeError("Wait for a finalization slot before submitting")
        future = asyncio.get_running_loop().run_in_executor(self.pool, function)
        self.jobs.append({"run_id": run_id, "future": future, "bytes": source_bytes})

    def summary(self):
        return [
            {
                "run_id": j["run_id"],
                "state": (
                    "cancelled"
                    if j["future"].cancelled()
                    else "failed"
                    if j["future"].done() and j["future"].exception()
                    else "complete"
                    if j["future"].done()
                    else "pending"
                ),
            }
            for j in self.jobs
        ]

    async def drain(self, deadline):
        remaining = None if deadline is None else deadline - time.time()
        if remaining is not None and remaining <= 0:
            self.check()
            if any(not j["future"].done() for j in self.jobs):
                raise TimeoutError("Finalization deadline reached; retain outputs")
            return [j["future"].result() for j in self.jobs]
        async with asyncio.timeout(remaining):
            return [await asyncio.shield(j["future"]) for j in self.jobs]

    def close(self):
        # The supervisor owns the hard compute deadline. Do not hide a worker
        # failure or wait indefinitely during error cleanup; queued work retains files.
        self.pool.shutdown(wait=False, cancel_futures=True)


def finalize_run(run_dir, notebook, *, validate_artifacts):
    """Manual/controller finalization; supplied notebook must be saved after execution."""
    run_dir, notebook = Path(run_dir), Path(notebook)
    executed = nbformat.read(notebook, as_version=4)
    if any(
        cell.cell_type == "code"
        and any(output.output_type == "error" for output in cell.outputs)
        for cell in executed.cells
    ):
        raise ValueError("Executed notebook contains an error")
    run_cells = [
        cell
        for cell in executed.cells
        if cell.cell_type == "code" and "optimization" in cell.metadata.get("tags", [])
    ]
    if len(run_cells) != 1 or run_cells[0].execution_count is None:
        raise ValueError("Notebook lacks an executed optimization cell")
    shutil.copyfile(notebook, run_dir / "experiment.executed.ipynb")
    evidence = validate_artifacts(run_dir)
    result = json.loads((run_dir / "result.json").read_text())
    atomic_json(run_dir / "artifact_integrity.json", evidence)
    atomic_json(
        run_dir / "experiment_complete.json",
        {
            "status": "complete",
            "meaning": "artifact finalization complete",
            "optimization_status": result["status"],
            "validation_passed": result["validation_passed"],
            "checks": evidence,
            "finished_unix": time.time(),
        },
    )
    return result
