"""Execute bounds notebooks serially, enforcing a wall-clock limit per notebook."""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
NAMES = (
    "parti",
    "genai_bench",
    "t2i_compbench",
    "diffusiondb",
    "drawbench",
    "geneval",
    "comparison",
)


def run_with_timeout(command, output, seconds=3600):
    """Terminate the child and its notebook kernel together at the deadline."""
    process = subprocess.Popen(
        command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True
    )
    try:
        code = process.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise
    if code:
        raise subprocess.CalledProcessError(code, command)


def execute(name):
    path = ROOT / f"sdxl_turbo_{name}.ipynb"
    notebook = nbformat.read(path, as_version=4)
    previous_path = os.environ.get("JUPYTER_PATH")
    try:
        with tempfile.TemporaryDirectory(prefix="sdxl-bounds-kernel-") as temporary:
            spec = Path(temporary) / "kernels" / "python3"
            spec.mkdir(parents=True)
            (spec / "kernel.json").write_text(
                json.dumps(
                    {
                        "argv": [
                            sys.executable,
                            "-m",
                            "ipykernel_launcher",
                            "-f",
                            "{connection_file}",
                        ],
                        "display_name": "Project Python",
                        "language": "python",
                    }
                )
            )
            os.environ["JUPYTER_PATH"] = temporary + (
                os.pathsep + previous_path if previous_path else ""
            )
            NotebookClient(
                notebook,
                timeout=3600,
                kernel_name="python3",
                resources={"metadata": {"path": str(ROOT)}},
            ).execute()
    finally:
        if previous_path is None:
            os.environ.pop("JUPYTER_PATH", None)
        else:
            os.environ["JUPYTER_PATH"] = previous_path
        nbformat.write(notebook, path)


def main():
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--one", choices=NAMES)
    # Only the supervised child executes in-process; public modes share a deadline.
    selection.add_argument("--_execute", choices=NAMES, help=argparse.SUPPRESS)
    parser.add_argument("--start", choices=NAMES, default="parti")
    args = parser.parse_args()
    if args._execute:
        execute(args._execute)
        return
    logs = ROOT / "results" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    names = (args.one,) if args.one else NAMES[NAMES.index(args.start) :]
    for name in names:
        print(f"START {name}", flush=True)
        started = time.monotonic()
        with (logs / f"{name}.log").open("w") as output:
            # The kernel uses this interpreter through a temporary kernelspec.
            try:
                run_with_timeout([sys.executable, __file__, "--_execute", name], output)
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                print(f"FAILED {name}; see {logs / (name + '.log')}", flush=True)
                raise
        print(
            f"DONE {name} in {(time.monotonic() - started) / 60:.2f} minutes",
            flush=True,
        )


if __name__ == "__main__":
    main()
