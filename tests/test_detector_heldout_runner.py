import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evolutionary_extensions.experiments.detector_evaluation.config import (
    DETECTORS,
    digest,
)
from evolutionary_extensions.experiments.detector_search.heldout_runner import (
    adapted_aeroblade_thresholds,
    validate_variants,
)
from evolutionary_extensions.persistence.packaging import atomic_json, hashes


def test_variant_factorial_rejects_missing_duplicate_and_changed_selection():
    subject = {
        "subject": "a",
        "group": "optimized",
        "target": "safe",
        "noise0_sha256": "frozen",
    }
    rows = [
        {
            **subject,
            "diffusion_seed": seed,
            "codec": codec,
            "sha256": "frozen" if seed == 0 and codec == "png" else "variant",
        }
        for seed in range(4)
        for codec in ("png", "jpeg95", "jpeg75")
    ]
    plan = {"subjects": [subject], "variants": rows}
    assert validate_variants(rows, plan) == 12
    for invalid in (rows[:-1], rows + rows[:1]):
        with pytest.raises(ValueError, match="factorial"):
            validate_variants(invalid, plan)
    with pytest.raises(ValueError, match="PNG"):
        validate_variants([{**rows[0], "sha256": "regenerated"}, *rows[1:]], plan)
    with pytest.raises(ValueError, match="provenance"):
        validate_variants([{**rows[0], "target": "ssp"}, *rows[1:]], plan)


def test_overlay_package_versions_are_not_overwritten_by_readonly_base(monkeypatch):
    from evolutionary_extensions.experiments.detector_search import scoring

    distributions = [
        SimpleNamespace(metadata={"Name": "timm"}, version=v)
        for v in ("isolated", "base")
    ]
    monkeypatch.setattr(
        scoring.importlib.metadata, "distributions", lambda: distributions
    )
    monkeypatch.setattr(scoring.importlib.metadata, "version", lambda name: "isolated")
    assert scoring.package_versions() == {"timm": "isolated"}


def test_adapted_calibration_reuse_requires_matching_procedure_and_source(tmp_path):
    from evolutionary_extensions.experiments.detector_evaluation.analysis import (
        threshold_at_fpr,
    )
    from evolutionary_extensions.experiments.detector_search import heldout_runner

    folder = tmp_path / "aeroblade"
    baseline = tmp_path / "benchmark"
    baseline.mkdir()
    packages = {"torch": "pinned"}
    asset = {"checkpoint": "frozen"}
    recipe = {
        "target": "aeroblade",
        "assets": asset,
        "packages": packages,
        "source_files": {
            p.name: hashes(p)["sha256"]
            for p in Path(heldout_runner.__file__).parent.glob("*.py")
        },
    }
    identity = digest(recipe)
    atomic_json(folder / "recipe.json", {"identity": identity, "recipe": recipe})
    rows = []
    for codec in ("png", "jpeg95"):
        values = []
        for i in range(200):
            path = baseline / f"photo-{i}-{codec}"
            path.write_bytes(f"frozen pixels {i}/{codec}".encode())
            sha = hashes(path)["sha256"]
            row = {
                "image_id": f"real-{i:04d}",
                "group": "calibration",
                "codec": codec,
                "path": path.name,
                "sha256": sha,
            }
            rows.append(row)
            raw = (i + 1) / 1000
            values.append(-raw)
            atomic_json(
                folder / "calibration" / f"{row['image_id']}-{codec}.json",
                {
                    "identity": identity,
                    "image_sha256": sha,
                    "reconstruction_seed": 1,
                    "reconstruction_seed_scope": "independently reset per image",
                    "raw": raw,
                    "synthetic_score": -raw,
                    "reconstruction_errors": [raw, raw + 0.1, raw + 0.2],
                },
            )
        atomic_json(
            folder / "calibration" / f"threshold-{codec}.json",
            {
                "identity": identity,
                "threshold": threshold_at_fpr(values),
                "samples": 200,
            },
        )
    thresholds, receipts = adapted_aeroblade_thresholds(
        tmp_path, rows, {"detectors": {"aeroblade": asset}}, packages, baseline
    )
    assert len(receipts) == 400 and set(thresholds) == {
        "aeroblade/png",
        "aeroblade/jpeg95",
    }
    with pytest.raises(ValueError, match="configuration changed"):
        adapted_aeroblade_thresholds(
            tmp_path,
            rows,
            {"detectors": {"aeroblade": asset}},
            {"torch": "different"},
            baseline,
        )
    path = folder / "calibration/real-0000-png.json"
    saved = json.loads(path.read_text())
    atomic_json(path, {**saved, "reconstruction_seed_scope": "shared stream"})
    with pytest.raises(ValueError, match="receipt differs"):
        adapted_aeroblade_thresholds(
            tmp_path, rows, {"detectors": {"aeroblade": asset}}, packages, baseline
        )


@pytest.mark.parametrize("selected", [None, ("universal_fake_detect", "bfree")])
def test_managed_searches_finish_and_freeze_before_sequential_heldout_kernels(
    tmp_path, monkeypatch, selected
):
    from evolutionary_extensions.experiments.detector_search import managed

    root = tmp_path / "output"
    root.mkdir()
    rows = [
        {
            "image_id": f"{group}-{i:04d}",
            "group": group,
            "codec": "png",
            "embedding_index": i,
            "sha256": f"{group}-{i}",
        }
        for group in ("random", "prompted")
        for i in range(20)
    ]
    settings = {
        "repository": str(tmp_path),
        "jupyter_root": str(tmp_path),
        "jupyter_url": "http://127.0.0.1:8888",
        "kernels": {stage: f"test-{stage}" for stage in ("generate", *DETECTORS)},
        "lifecycle_owner": "existing-owner",
    }
    targets = selected or DETECTORS
    subject_count = len(targets) * 20 + 40
    config = {"output": str(root), "managed": settings, "targets": list(targets)}
    path = tmp_path / "config.json"
    atomic_json(path, config)
    monkeypatch.setattr(
        managed,
        "verify_prerequisite",
        lambda _: ({}, rows, [], {"archive_sha256": "a" * 64}),
    )
    order, active = [], []

    class Response:
        status, url = 200, "http://127.0.0.1:8888/lab"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def read(self):
            return b"managed"

    class Client:
        cookie_jar = SimpleNamespace(filter_cookies=lambda _: {})

        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def get(self, url):
            return Response()

    class Pool:
        def __init__(self, *args):
            pass

        async def __aenter__(self):
            assert not active
            active.append(True)
            return [{"kernel_id": "owned", "session_id": "owned"}]

        async def __aexit__(self, *args):
            active.pop()
            return False

    async def cells(*args):
        notebook = args[5]
        last = [c for c in notebook.cells if c.cell_type == "code"][
            -1
        ].source.splitlines()[-1]
        if last.startswith("run_validation"):
            stage = notebook.metadata.kernelspec.name.removeprefix("test-")
            order.append(f"generator:{stage}")
            assert not (root / "heldout_plan.json").exists()
            atomic_json(
                root / "generator_validation" / stage / "parity.json",
                {
                    "passed": True,
                    "stage": stage,
                    "known_physical_images": 192,
                    "candidate_budget_used": 0,
                },
            )
        elif "calibration_only=True" in last:
            order.append("calibration:aeroblade")
            assert (root / "heldout_plan.json").exists()
            atomic_json(
                root / "aeroblade/calibration_complete.json",
                {
                    "status": "complete",
                    "optimizer_candidates": 0,
                    "calibration_score_identities": 400,
                },
            )
        elif last.startswith("run_target"):
            assert order[:6] == [f"generator:{d}" for d in ("generate", *DETECTORS)]
            target = notebook.metadata.kernelspec.name.removeprefix("test-")
            order.append(f"search:{target}")
            assert not (root / "heldout_plan.json").exists()
            atomic_json(
                root / target / "search_complete.json",
                {"status": "complete", "candidates": 6400},
            )
            atomic_json(
                root / target / "finalists.json",
                {
                    "records": [
                        {"index": i, "image": {"sha256": f"{target}-{i}"}}
                        for i in range(20)
                    ]
                },
            )
        else:
            assert len(order) >= 5 and (root / "heldout_plan.json").exists()
            assert (
                json.loads((root / "searches_complete.json").read_text())["candidates"]
                == len(targets) * 6400
            )
            stage = notebook.metadata.kernelspec.name.removeprefix("test-")
            order.append(f"heldout:{stage}")
            if stage != "generate":
                summaries = [
                    {
                        "matrix_measurements": subject_count * 12,
                        "calibration_measurements": 0,
                        "patch_seed": 42 if stage == "ssp" else None,
                    }
                ]
                if stage == "ssp":
                    summaries += [
                        {
                            "matrix_measurements": subject_count * 3,
                            "calibration_measurements": 200,
                            "patch_seed": seed,
                        }
                        for seed in (43, 44, 45)
                    ]
                atomic_json(
                    root / "heldout" / f"{stage}_complete.json",
                    {"status": "complete", "detector": stage, "summaries": summaries},
                )

    monkeypatch.setattr(managed.aiohttp, "ClientSession", Client)
    monkeypatch.setattr(managed, "ManagedKernelPool", Pool)
    monkeypatch.setattr(managed, "execute_cells", cells)
    asyncio.run(managed.execute(path))
    assert order == [f"generator:{d}" for d in ("generate", *DETECTORS)] + [
        f"search:{d}" for d in targets
    ] + ([] if "aeroblade" in targets else ["calibration:aeroblade"]) + [
        f"heldout:{d}" for d in ("generate", *DETECTORS)
    ]
    assert not active
    receipt = json.loads((root / "heldout_inference_complete.json").read_text())
    assert receipt["heldout_matrix_scores"] == subject_count * 69
    assert receipt["ssp_extra_calibration_scores"] == 600
    assert receipt["reporting_pending"] is True


def test_resumed_notebook_keeps_prior_executed_outputs_by_checksum(tmp_path):
    import nbformat

    from evolutionary_extensions.experiments.detector_search.managed import (
        write_notebook,
    )

    path = tmp_path / "notebooks/ssp.ipynb"
    executed = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "print('completed stage')",
                execution_count=1,
                outputs=[
                    nbformat.v4.new_output(
                        "stream", name="stdout", text="completed stage\n"
                    )
                ],
            )
        ]
    )
    write_notebook(executed, path)
    previous = hashes(path)["sha256"]
    replacement = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell("resume unfinished scores")]
    )
    write_notebook(replacement, path)
    history = list((path.parent / "history").glob("*.ipynb"))
    assert len(history) == 1 and hashes(history[0])["sha256"] == previous
    assert (
        nbformat.read(history[0], as_version=4).cells[0].outputs[0].text
        == "completed stage\n"
    )


def test_completed_heldout_stage_recovers_without_model_loading_or_scoring(
    tmp_path, monkeypatch
):
    torch = pytest.importorskip("torch")
    from evolutionary_extensions.experiments.detector_search import (
        heldout_runner as runner,
    )

    root = tmp_path / "heldout"
    root.mkdir()
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    rows = []
    for seed in range(4):
        for codec in ("png", "jpeg95", "jpeg75"):
            path = root / f"a-noise{seed}-{codec}"
            path.write_bytes(f"frozen pixels {seed}/{codec}".encode())
            rows.append(
                {
                    "subject": "a",
                    "image_id": f"a-noise{seed}",
                    "group": "optimized",
                    "target": "safe",
                    "diffusion_seed": seed,
                    "codec": codec,
                    "path": path.name,
                    "sha256": hashes(path)["sha256"],
                }
            )
    plan = {
        "subjects": [
            {
                "subject": "a",
                "group": "optimized",
                "target": "safe",
                "noise0_sha256": rows[0]["sha256"],
            }
        ],
        "variants": rows,
    }
    atomic_json(tmp_path / "heldout_plan.json", plan)
    atomic_json(root / "images.json", rows)
    atomic_json(
        root / "generation_complete.json",
        {"status": "complete", "variants": 12, "plan_sha256": digest(plan)},
    )
    atomic_json(benchmark / "thresholds.json", {"safe/png": 0.7, "safe/jpeg95": 0.8})
    config_path = tmp_path / "config.json"
    monkeypatch.setattr(
        runner,
        "load_phase",
        lambda _: ({"output": str(tmp_path)}, {"output": str(benchmark)}, rows, [], {}),
    )
    monkeypatch.setattr(
        runner,
        "assets_for",
        lambda *_: {"detectors": {"safe": {"checkpoint": "frozen"}}},
    )
    monkeypatch.setattr(runner, "package_versions", lambda: {"torch": "frozen"})
    monkeypatch.setattr(
        runner,
        "scoring_procedure",
        lambda detector, asset, packages, batch: {
            "detector": detector,
            "asset": asset,
            "packages": packages,
            "batch": batch,
        },
    )
    monkeypatch.setattr(runner, "guard_disk", lambda _: None)
    monkeypatch.setattr(runner, "cuda_preflight", lambda _: None)
    monkeypatch.setattr(runner, "smoke_pool", lambda _: rows[:4])
    for name, function in (
        ("synchronize", lambda: None),
        ("empty_cache", lambda: None),
        ("max_memory_allocated", lambda: 1),
        ("max_memory_reserved", lambda: 2),
    ):
        monkeypatch.setattr(torch.cuda, name, function)
    calls = []

    class Scorer:
        def __init__(self, *args):
            pass

        def validate_native(self, *args):
            calls.append("native gate")

        def score(self, pool, *, image_root, batch):
            assert batch == 1
            calls.extend(r["image_id"] + r["codec"] for r in pool)
            return [{"raw": 0.2, "synthetic_score": 0.2} for _ in pool]

    monkeypatch.setattr(runner, "HeldoutClassifier", Scorer)
    runner.run_detector(config_path, "safe")
    assert calls[0] == "native gate" and len(calls) == 13
    before = json.loads((root / "safe_complete.json").read_text())
    assert before["summaries"][0]["queries"]["attempted_queries"] == 12
    monkeypatch.setattr(
        runner,
        "HeldoutClassifier",
        lambda *_: pytest.fail("completed stage reloaded the model"),
    )
    runner.run_detector(config_path, "safe")
    after = json.loads((root / "safe_complete.json").read_text())
    assert after["summaries"] == before["summaries"] and len(calls) == 13
