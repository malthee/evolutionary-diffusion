"""Prepare disabled experiment settings and a kernel for an existing Azure host."""

import argparse
import json
import sys
from pathlib import Path

from evolutionary_extensions.persistence.packaging import atomic_json, hashes


def setup(
    repo,
    config_path,
    *,
    cache,
    output,
    jupyter_root,
    prepared_root,
    kernel_name="evolutionary-experiments",
    kernel_root=None,
    credentials=None,
    folder_id=None,
):
    """Write local metadata without installing, launching or deleting resources.

    An exported checkout may supply deployment_manifest.json for SHA-256 verification.
    Existing scientific settings and bounds are retained; execution stays disabled.
    """
    repo = Path(repo).expanduser().resolve(strict=True)
    prepared = Path(prepared_root).expanduser().resolve()
    kernel_base = (
        Path(kernel_root).expanduser()
        if kernel_root is not None
        else Path.home() / ".local/share/jupyter/kernels"
    )
    if (
        not kernel_name
        or Path(kernel_name).name != kernel_name
        or kernel_name in {".", ".."}
    ):
        raise ValueError("kernel_name must be a single directory name")
    if bool(credentials) != bool(folder_id):
        raise ValueError("credentials and folder_id must be supplied together")
    manifest = repo / "deployment_manifest.json"
    verified_files = 0
    if manifest.exists():
        for row in json.loads(manifest.read_text())["files"]:
            path = (repo / row["path"]).resolve(strict=True)
            if not path.is_relative_to(repo) or hashes(path)["sha256"] != row["sha256"]:
                raise ValueError("Deployment checksum mismatch")
            verified_files += 1
    config = json.loads(Path(config_path).expanduser().read_text())
    config["execution"].update(
        enabled=False,
        jupyter_root=str(Path(jupyter_root).expanduser().resolve()),
        kernel_name=kernel_name,
        campaign_root=str(prepared),
        archive_root=str(Path(output).expanduser().resolve().parent / "archives"),
    )
    config["deployment"].update(
        cache_dir=str(Path(cache).expanduser().resolve()),
        output_root=str(Path(output).expanduser().resolve()),
    )
    config["drive"].update(
        enabled=bool(credentials and folder_id),
        credentials=str(Path(credentials).expanduser().resolve())
        if credentials
        else None,
        folder_id=folder_id,
        delete_after_verification=False,
    )
    prepared.mkdir(parents=True, exist_ok=False)
    private = prepared / "private_config.json"
    atomic_json(private, config, private=True)
    kernel = kernel_base / kernel_name
    spec = {
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Evolutionary experiments (CUDA)",
        "language": "python",
        "env": {
            "NVIDIA_TF32_OVERRIDE": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "TOKENIZERS_PARALLELISM": "false",
            "EVOLUTIONARY_CHECKOUT": str(repo),
            "PYTHONPATH": str(repo),
            "MPLBACKEND": "Agg",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
        },
    }
    atomic_json(kernel / "kernel.json", spec)
    return {
        "canonical_repo": str(repo),
        "private_config": str(private),
        "kernel": str(kernel),
        "execution_enabled": False,
        "verified_files": verified_files,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "config", "cache", "output", "jupyter-root", "prepared-root"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--kernel-name", default="evolutionary-experiments")
    parser.add_argument("--kernel-root")
    parser.add_argument("--credentials")
    parser.add_argument("--folder-id")
    args = vars(parser.parse_args())
    print(json.dumps(setup(args.pop("repo"), args.pop("config"), **args), indent=2))
