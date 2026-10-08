"""Immutable ZIP64 packaging and integrity checks, independent of destinations."""

import hashlib
import json
import os
import zipfile
from pathlib import Path

CHUNK = 8 * 1024 * 1024


def atomic_json(path, data, *, private=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
    if private:
        temporary.chmod(0o600)
    temporary.replace(path)


def hashes(path):
    md5 = hashlib.md5(usedforsecurity=False)
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(CHUNK):
            md5.update(block)
            sha.update(block)
    return {
        "bytes": Path(path).stat().st_size,
        "md5": md5.hexdigest(),
        "sha256": sha.hexdigest(),
    }


def snapshot(folder):
    folder = Path(folder)
    result = {}
    for path in sorted(folder.rglob("*")):
        if path.is_symlink():
            raise ValueError("Experiment archives must not contain symlinks")
        if path.is_file():
            if path.suffix.lower() in {".pem", ".key"} or path.name in {
                "drive_oauth.json",
                "client_secret.json",
                "credentials.json",
            }:
                raise ValueError("Credential-like file found in experiment output")
            result[path.relative_to(folder).as_posix()] = hashes(path)
    return result


def package(experiment, archive):
    source = Path(experiment).resolve(strict=True)
    archive = Path(archive).resolve()
    if not source.is_dir() or archive.is_relative_to(source):
        raise ValueError("ZIP must be outside the experiment directory")
    marker = source / "experiment_complete.json"
    if not marker.exists() or json.loads(marker.read_text()).get("status") not in {
        "complete",
        "passed",
        "success",
    }:
        raise ValueError("Experiment must have a successful completion marker")
    files = snapshot(source)
    if not files:
        raise ValueError("Experiment folder is empty")
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_suffix(archive.suffix + ".partial")
    with zipfile.ZipFile(
        partial, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True
    ) as output:
        for relative in files:
            output.write(source / relative, Path(source.name) / relative)
        output.writestr(
            f"{source.name}/archive_file_manifest.json",
            json.dumps(files, sort_keys=True),
        )
    if snapshot(source) != files:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            "Experiment changed during packaging; refuse incomplete archive"
        )
    partial.replace(archive)
    manifest = {
        "experiment": str(source),
        "archive": str(archive),
        "files": files,
        **hashes(archive),
    }
    atomic_json(str(archive) + ".manifest.json", manifest)
    return manifest
