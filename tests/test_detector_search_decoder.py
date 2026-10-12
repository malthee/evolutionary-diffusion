import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from evolutionary_extensions.experiments.detector_search import scoring


def test_decoder_only_factory_is_offline_and_exact(tmp_path, monkeypatch):
    calls = []

    class Auto:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls.append(("auto", args, kwargs))
            return "sd"

    class Decoder:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls.append(("decoder", args, kwargs))
            return "kandinsky"

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(
            AutoPipelineForImage2Image=Auto, KandinskyImg2ImgPipeline=Decoder
        ),
    )
    (tmp_path / "model_index.json").write_text(
        json.dumps(
            {"_class_name": "KandinskyPipeline", "movq": ["diffusers", "VQModel"]}
        )
    )
    item = {
        "name": "kandinsky-2-1",
        "repo": "kandinsky-community/kandinsky-2-1",
        "revision": "04b366659663bf60deacb96fd3e38f1cb39a3be4",
        "snapshot": str(tmp_path),
    }
    assert scoring.load_autoencoder_pipeline(item) == "kandinsky"
    assert calls[-1] == (
        "decoder",
        (str(tmp_path),),
        {
            "torch_dtype": torch.float16,
            "use_safetensors": True,
            "variant": None,
            "local_files_only": True,
        },
    )
    with pytest.raises(ValueError, match="revision"):
        scoring.load_autoencoder_pipeline({**item, "revision": "changed"})
    (tmp_path / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "KandinskyCombinedPipeline",
                "movq": ["diffusers", "VQModel"],
            }
        )
    )
    with pytest.raises(ValueError, match="definition"):
        scoring.load_autoencoder_pipeline(item)
    assert (
        scoring.load_autoencoder_pipeline({**item, "name": "stable-diffusion-v1-1"})
        == "sd"
    )
    assert calls[-1][2]["variant"] == "fp16"
    with pytest.raises(ValueError, match="autoencoder"):
        scoring.load_autoencoder_pipeline({**item, "name": "unknown"})


def test_native_lpips_uses_no_grad_and_records_shared_loader(tmp_path, monkeypatch):
    image_module, distance_module, data_module = (
        ModuleType("aeroblade.image"),
        ModuleType("aeroblade.distances"),
        ModuleType("aeroblade.data"),
    )
    image_module.AutoPipelineForImage2Image = SimpleNamespace(
        from_pretrained=lambda: None
    )
    data_module.ImageFolder = lambda paths: paths
    observed = []

    def reconstruct(*args, **kwargs):
        observed.append(("reconstruction", torch.is_grad_enabled()))
        assert (
            image_module.AutoPipelineForImage2Image.from_pretrained() == "offline-pipe"
        )
        return [tmp_path / "reconstruction.png"]

    def lpips(*args, **kwargs):
        observed.append(("lpips", torch.is_grad_enabled()))
        # Mirrors the release's conversion of a parameter-dependent result.
        parameter = torch.tensor(0.2, requires_grad=True)
        return (parameter * 1).numpy()

    image_module.compute_reconstructions = reconstruct
    distance_module._compute_lpips = SimpleNamespace(func=lpips)
    distance_module.LPIPS = lambda **kw: SimpleNamespace(
        _postprocess=lambda x: {"lpips_vgg_2": torch.tensor([-float(x)])}
    )
    parent = ModuleType("aeroblade")
    parent.image, parent.distances, parent.data = (
        image_module,
        distance_module,
        data_module,
    )
    monkeypatch.setitem(sys.modules, "aeroblade", parent)
    for mod in (image_module, distance_module, data_module):
        monkeypatch.setitem(sys.modules, mod.__name__, mod)
    record = {
        "reconstruction_errors": [0.2],
        "raw": 0.2,
        "objective": -0.2,
        "synthetic_score": -0.2,
    }
    instance = object.__new__(scoring.SearchAeroblade)
    instance.root = tmp_path
    instance.asset = {}
    instance.components = [
        {
            "asset": {"name": "kandinsky-2-1", "snapshot": "frozen"},
            "pipe": "offline-pipe",
        }
    ]
    instance.score = lambda *a, **k: [record]
    monkeypatch.setattr(
        scoring,
        "SearchAeroblade",
        lambda *a, **k: SimpleNamespace(score=lambda *a, **k: [record]),
    )
    instance.validate_native([{"path": "input.png"}], tmp_path)
    assert observed == [("reconstruction", False), ("lpips", False)]
    proof = json.loads((tmp_path / "parity.json").read_text())
    assert proof["passed"] and proof["native_reference_pipeline_reused"]
    assert proof["native_autograd_disabled"] and torch.is_grad_enabled()


def test_effective_procedure_keeps_actual_import_beside_ambiguous_metadata(
    tmp_path, monkeypatch
):
    module_file = tmp_path / "imported.py"
    module_file.write_text("frozen actual runtime")
    monkeypatch.setattr(
        scoring.importlib,
        "import_module",
        lambda name: SimpleNamespace(
            __file__=str(module_file), __version__="actually-imported"
        ),
    )
    result = scoring.scoring_procedure("aeroblade", {}, {"torch": "stale-metadata"}, 1)
    assert result["packages"]["torch"] == "stale-metadata"
    assert result["actual_runtime_imports"]["torch"]["version"] == "actually-imported"
    assert result["actual_runtime_imports"]["torch"]["path"] == str(module_file)
    assert result["native_reference"].startswith(
        "official reconstruction/LPIPS under no_grad"
    )
