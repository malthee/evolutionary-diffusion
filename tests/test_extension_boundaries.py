"""Portable setup and lightweight controller imports protect the public boundary."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from evolutionary_extensions.azure.deployment import setup
from evolutionary_extensions.execution.campaign import prepare_campaign

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("cuda", [None, "2"])
def test_controller_import_does_not_load_inference_or_change_cuda(cuda):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(ROOT))
    if cuda is None:
        environment.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        environment["CUDA_VISIBLE_DEVICES"] = cuda
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            """
import importlib.abc
import os
import sys
before = dict(os.environ)
class RejectInference(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'diffusers', 'transformers'}:
            raise AssertionError('Controller imported inference: ' + fullname)
sys.meta_path.insert(0, RejectInference())
import evolutionary_extensions.execution.campaign
import evolutionary_extensions.execution.finalization
import evolutionary_extensions.azure.deployment
import evolutionary_extensions.experiments.prompt_embedding
assert dict(os.environ) == before
""",
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def profile(tmp_path):
    config = json.loads((ROOT / "configs/examples/osga.json").read_text())
    config["execution"]["enabled"] = True
    config["deployment"].update(bounds_source="parti", bounds_file="custom.safetensors")
    path = tmp_path / "source.json"
    path.write_text(json.dumps(config))
    return path


def prepare_host(tmp_path, config_path, **overrides):
    options = dict(
        cache=tmp_path / "cache",
        output=tmp_path / "runs",
        jupyter_root=tmp_path,
        prepared_root=tmp_path / "prepared",
        kernel_root=tmp_path / "kernels",
        kernel_name="experiment-test",
    )
    options.update(overrides)
    return setup(ROOT, config_path, **options)


def test_host_preparation_preserves_recipe_disables_execution_and_deletion(tmp_path):
    path = profile(tmp_path)
    result = prepare_host(
        tmp_path, path, credentials=tmp_path / "oauth.json", folder_id="folder"
    )
    config = json.loads(Path(result["private_config"]).read_text())
    original = json.loads(path.read_text())
    assert config["experiment"] == original["experiment"]
    assert config["deployment"]["bounds_source"] == "parti"
    assert config["deployment"]["bounds_file"] == "custom.safetensors"
    assert config["execution"]["enabled"] is False
    assert config["execution"]["jupyter_root"] == str(tmp_path)
    assert config["drive"]["enabled"] is True
    assert config["drive"]["delete_after_verification"] is False
    assert Path(result["private_config"]).stat().st_mode & 0o777 == 0o600
    kernel = json.loads((Path(result["kernel"]) / "kernel.json").read_text())
    assert kernel["argv"][0] == sys.executable
    assert kernel["env"]["EVOLUTIONARY_CHECKOUT"] == str(ROOT)
    assert kernel["env"]["NVIDIA_TF32_OVERRIDE"] == "0"
    with pytest.raises(FileExistsError):
        prepare_host(tmp_path, path)


@pytest.mark.parametrize(
    "options",
    [
        {"credentials": "oauth.json"},
        {"folder_id": "folder"},
        {"kernel_name": "../escape"},
        {"kernel_name": ".."},
    ],
)
def test_invalid_host_settings_write_nothing(tmp_path, options):
    path = profile(tmp_path)
    with pytest.raises(ValueError):
        prepare_host(tmp_path, path, **options)
    assert not (tmp_path / "prepared").exists()
    assert not (tmp_path / "kernels").exists()


def test_export_checksum_verified_before_writing(tmp_path):
    repo = tmp_path / "export"
    repo.mkdir()
    (repo / "source.py").write_text("content")
    digest = hashlib.sha256(b"content").hexdigest()
    manifest = repo / "deployment_manifest.json"
    manifest.write_text(
        json.dumps({"files": [{"path": "source.py", "sha256": digest}]})
    )
    path = profile(tmp_path)
    options = dict(
        cache=tmp_path,
        output=tmp_path / "runs",
        jupyter_root=tmp_path,
        prepared_root=tmp_path / "prepared",
        kernel_root=tmp_path / "kernels",
    )
    (repo / "source.py").write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        setup(repo, path, **options)
    assert not (tmp_path / "prepared").exists()
    (repo / "source.py").write_text("content")
    assert setup(repo, path, **options)["verified_files"] == 1


def test_preparation_accepts_an_explicit_recipe_adapter(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "experiment": {"kind": "other"},
                "campaign": {"seeds": [7]},
                "execution": {"enabled": True},
            }
        )
    )

    class Adapter:
        @staticmethod
        def resolve_trials(config, seeds):
            return [
                {"seed": seed, "experiment": config["experiment"]} for seed in seeds
            ]

    resolved = prepare_campaign(path, tmp_path / "campaign", adapter=Adapter())
    config = json.loads(resolved.read_text())
    assert config["execution"]["enabled"] is False
    assert config["experiment"] == {"kind": "other"}
    manifest = json.loads((resolved.parent / "campaign_manifest.json").read_text())
    assert manifest["trials"] == [{"seed": 7, "experiment": {"kind": "other"}}]
