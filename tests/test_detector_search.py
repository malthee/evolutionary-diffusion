import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cma")

from evolutionary_extensions.experiments.detector_search.optimizer import (
    EmbeddingCoordinates,
    SearchOptimizer,
)


def test_private_rng_and_pending_checkpoint_preserve_future_proposals():
    np.random.seed(91)
    before = np.random.get_state()
    optimizer = SearchOptimizer(np.full(8, 0.5), budget=24, population=8)
    first = optimizer.ask()
    assert np.array_equal(first, optimizer.ask())
    after = np.random.get_state()
    assert (
        before[0] == after[0]
        and np.array_equal(before[1], after[1])
        and before[2:] == after[2:]
    )
    content = optimizer.state()
    restored = SearchOptimizer.restore_owned_state(
        content, hashlib.sha256(content).hexdigest()
    )
    assert np.array_equal(restored.ask(), first)
    scores = (first**2).sum(axis=1)
    for state in (optimizer, restored):
        state.tell(scores)
    assert optimizer.completed == restored.completed == 8
    assert np.array_equal(optimizer.ask(), restored.ask())
    with pytest.raises(ValueError, match="checksum"):
        SearchOptimizer.restore_owned_state(
            content + b"changed", hashlib.sha256(content).hexdigest()
        )


def test_budget_is_exact_and_nonfinite_scores_do_not_advance_it():
    optimizer = SearchOptimizer(np.full(4, 0.5), budget=16, population=8)
    first = optimizer.ask()
    with pytest.raises(ValueError):
        optimizer.tell(np.full(8, np.nan))
    assert optimizer.completed == 0 and np.array_equal(optimizer.ask(), first)
    optimizer.tell((first**2).sum(axis=1))
    second = optimizer.ask()
    optimizer.tell((second**2).sum(axis=1))
    assert optimizer.completed == 16
    with pytest.raises(StopIteration):
        optimizer.ask()
    with pytest.raises(ValueError):
        SearchOptimizer(np.full(4, 0.5), budget=17, population=8)


def test_coordinate_geometry_removes_constants_and_repairs_inward_fp16():
    bounds = {
        "token_min": torch.tensor([[[1.0, 0.1234]]], dtype=torch.float32),
        "token_max": torch.tensor([[[1.0, 0.4]]], dtype=torch.float32),
        "pooled_min": torch.tensor([[2.0, -1.2]], dtype=torch.float32),
        "pooled_max": torch.tensor([[2.0, 1.3]], dtype=torch.float32),
    }
    geometry = EmbeddingCoordinates(bounds)
    assert int(geometry.active.sum()) == 2
    for point in (np.zeros(2), np.ones(2)):
        argument = geometry.decode(point)
        assert argument.prompt_embeds.dtype == torch.float16
        assert argument.pooled_prompt_embeds.dtype == torch.float16
        assert (
            argument.prompt_embeds[0, 0, 0] == 1
            and argument.pooled_prompt_embeds[0, 0] == 2
        )
        for key, tensor in (
            ("token", argument.prompt_embeds),
            ("pooled", argument.pooled_prompt_embeds),
        ):
            assert (tensor.float() >= bounds[f"{key}_min"]).all() and (
                tensor.float() <= bounds[f"{key}_max"]
            ).all()
        normalized = geometry.encode(argument)
        assert ((normalized >= 0) & (normalized <= 1)).all()
    with pytest.raises(ValueError):
        geometry.decode(np.array([-0.01, 0.5]))


def test_score_recovery_does_not_query_completed_candidates(tmp_path):
    from evolutionary_extensions.experiments.detector_search.store import SearchStore

    pixel = tmp_path / "candidate.png"
    pixel.write_bytes(b"frozen pixels")
    image = {
        "path": pixel.name,
        "sha256": hashlib.sha256(pixel.read_bytes()).hexdigest(),
    }
    store = SearchStore(tmp_path, {"budget": 16, "target": "safe"})
    optimizer = SearchOptimizer(np.full(4, 0.5), budget=16, population=8)
    pending = optimizer.ask()
    store.checkpoint(optimizer)
    intent = store.begin_queries([0], [image])
    score = {"objective": -100.0, "raw": 0.0, "synthetic_score": 0.0}
    store.finish_queries(intent, [score])
    restored = SearchStore(tmp_path, {"budget": 16, "target": "safe"})
    assert np.array_equal(restored.restore().ask(), pending)
    assert restored.score(0, image)["objective"] == -100
    with pytest.raises(ValueError, match="must not be queried"):
        restored.begin_queries([0], [image])
    assert restored.query_counts()["fitness"] == {
        "attempted_queries": 1,
        "component_calls": 1,
        "unfinished_queries": 0,
    }
    with pytest.raises(ValueError, match="recipe changed"):
        SearchStore(tmp_path, {"budget": 16, "target": "ssp"})
    pixel.write_bytes(b"changed")
    with pytest.raises(ValueError, match="pixels changed"):
        restored.score(0, image)


def test_failed_batch_is_counted_separately_and_publishes_no_scores(tmp_path):
    from evolutionary_extensions.experiments.detector_search.store import SearchStore

    image_path = tmp_path / "candidate.png"
    image_path.write_bytes(b"pixels")
    image = {"path": image_path.name, "sha256": hashlib.sha256(b"pixels").hexdigest()}
    store = SearchStore(tmp_path, {"budget": 8})
    intent = store.begin_queries([0, 1], [image, image], detector_calls=3)
    with pytest.raises(ValueError, match="Non-finite"):
        store.finish_queries(
            intent,
            [
                {"objective": 0.0, "raw": 0.0, "synthetic_score": 0.0},
                {"objective": np.nan, "raw": 0.0, "synthetic_score": 0.0},
            ],
        )
    assert store.score(0, image) is None
    assert store.query_counts()["fitness"] == {
        "attempted_queries": 2,
        "component_calls": 6,
        "unfinished_queries": 2,
    }


def test_interrupted_search_reuses_committed_scores_and_matches_trajectory(tmp_path):
    from evolutionary_extensions.experiments.detector_search.loop import (
        freeze_finalists,
        search,
    )
    from evolutionary_extensions.experiments.detector_search.store import SearchStore

    def run(folder, interrupt=False):
        store = SearchStore(folder, {"budget": 16, "target": "safe"})
        optimizer = store.restore() or SearchOptimizer(
            np.full(4, 0.5), budget=16, population=8
        )

        def render(indices, proposals):
            rows = []
            for index, proposal in zip(indices, proposals):
                p = folder / f"{index}.png"
                content = proposal.tobytes()
                if p.exists():
                    assert p.read_bytes() == content
                else:
                    p.write_bytes(content)
                rows.append(
                    {"path": p.name, "sha256": hashlib.sha256(content).hexdigest()}
                )
            return rows

        def score(images):
            if interrupt and store.query_counts()["fitness"]["attempted_queries"] > 4:
                raise RuntimeError("interruption after one committed scoring batch")
            return [
                {
                    "objective": float(
                        np.frombuffer((folder / image["path"]).read_bytes()).sum()
                    ),
                    "raw": 0.5,
                    "synthetic_score": 0.5,
                }
                for image in images
            ]

        search(optimizer, store, render, score)
        return store, optimizer

    with pytest.raises(RuntimeError, match="interruption"):
        run(tmp_path / "resumed", interrupt=True)
    resumed, optimizer = run(tmp_path / "resumed")
    clean, reference = run(tmp_path / "clean")
    assert optimizer.completed == reference.completed == 16
    assert np.array_equal(optimizer.strategy.mean, reference.strategy.mean)
    assert resumed.query_counts()["fitness"]["attempted_queries"] == 20
    assert clean.query_counts()["fitness"]["attempted_queries"] == 16
    selected = freeze_finalists(resumed, 3)
    assert selected["selected"] == 3
    assert selected == freeze_finalists(resumed, 3)
    with pytest.raises(ValueError, match="Refuse to change"):
        freeze_finalists(resumed, 4)


def test_heldout_factorial_preserves_unselected_controls_and_budget_separation(
    tmp_path,
):
    from evolutionary_extensions.experiments.detector_evaluation.config import DETECTORS
    from evolutionary_extensions.experiments.detector_search.protocol import (
        freeze_heldout,
        heldout_plan,
    )

    finalists = {
        target: {
            "records": [
                {"index": i, "image": {"sha256": f"{target}-{i}"}} for i in range(20)
            ]
        }
        for target in DETECTORS
    }
    images = [
        {
            "group": group,
            "image_id": f"{group}-{i:04d}",
            "codec": "png",
            "embedding_index": i,
            "sha256": f"{group}-{i}",
        }
        for group in ("random", "prompted")
        for i in reversed(range(100))
    ]
    plan = heldout_plan(finalists, images)
    assert len(plan["subjects"]) == 140
    assert len(plan["variants"]) == 1680
    assert plan["logical_primary_matrix_scores"] == 8400
    assert plan["logical_extra_ssp_scores"] == 1260
    assert plan["new_generated_images"] == 420
    assert not plan["search_feedback"]
    assert [s["subject"] for s in plan["subjects"] if s["group"] == "random"] == [
        f"random-{i:04d}" for i in range(20)
    ]
    freeze_heldout(tmp_path / "heldout.json", plan)
    with pytest.raises(ValueError, match="already frozen"):
        freeze_heldout(tmp_path / "heldout.json", {**plan, "search_feedback": True})
    finalists["safe"]["records"] = finalists["safe"]["records"][:1]
    reduced = heldout_plan(finalists, images)
    assert reduced["reduced_unique_finalist_sets"] == {"safe": 1}
    assert len(reduced["subjects"]) == 121


def test_continuous_objectives_retain_ranking_after_probability_saturation():
    from evolutionary_extensions.experiments.detector_search.scoring import (
        classifier_scores,
    )

    safe = classifier_scores("safe", torch.tensor([[0.0, -100.0], [0.0, -200.0]]))
    assert safe[1]["objective"] < safe[0]["objective"]
    assert safe[1]["synthetic_score"] <= safe[0]["synthetic_score"]
    ssp = classifier_scores("ssp", torch.tensor([[20.0], [40.0]]))
    assert ssp[0]["raw"] == ssp[1]["raw"] == 1
    assert ssp[1]["objective"] < ssp[0]["objective"]
    assert ssp[1]["synthetic_score"] == 0
    uni = classifier_scores("universal_fake_detect", torch.tensor([[-100.0], [-200.0]]))
    assert uni[1]["objective"] < uni[0]["objective"]
    bfree = classifier_scores("bfree", torch.tensor([[2.0, 1.0], [2.0, 0.0]]))
    assert bfree[1]["objective"] == bfree[1]["synthetic_score"] == -2
    with pytest.raises(ValueError, match="finite"):
        classifier_scores("safe", torch.tensor([[0.0, float("nan")]]))


def test_search_cannot_start_before_verified_archive_or_with_wrong_budget(tmp_path):
    import json

    from evolutionary_extensions.experiments.detector_search.runner import (
        verify_prerequisite,
    )

    receipt = tmp_path / "completion.json"
    receipt.write_text(
        json.dumps(
            {"status": "complete", "primary_scores": 5000, "compute_state": "Stopped"}
        )
    )
    config = {
        "benchmark_completion": str(receipt),
        "budget": 6400,
        "population": 64,
        "generation_batch": 4,
        "algorithm": "sep-cma-es",
        "seed": 42,
        "sigma": 0.15,
    }
    with pytest.raises(ValueError, match="verified-archive"):
        verify_prerequisite(config)
    receipt.write_text(
        json.dumps(
            {
                "status": "complete",
                "primary_scores": 5000,
                "compute_state": "Stopped",
                "archive_files_verified": 1,
                "archive_sha256": "a" * 64,
            }
        )
    )
    with pytest.raises(ValueError, match="authorized recipe"):
        verify_prerequisite({**config, "budget": 32000})


def test_last_batch_resource_failure_preserves_scores_and_recovers_without_queries(
    tmp_path,
):
    from evolutionary_extensions.experiments.detector_search.loop import search
    from evolutionary_extensions.experiments.detector_search.store import SearchStore

    store = SearchStore(tmp_path, {"budget": 8, "target": "safe"})
    optimizer = SearchOptimizer(np.full(4, 0.5), budget=8, population=8)

    def render(indices, proposals):
        rows = []
        for index, proposal in zip(indices, proposals):
            p = tmp_path / f"{index}.png"
            if p.exists():
                assert p.read_bytes() == proposal.tobytes()
            else:
                p.write_bytes(proposal.tobytes())
            rows.append(
                {"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            )
        return rows

    def score(images):
        return [{"objective": 0.5, "raw": 0.5, "synthetic_score": 0.5} for _ in images]

    checks = 0

    def guard():
        nonlocal checks
        checks += 1
        if checks == 4:
            raise RuntimeError("memory limit after final persisted batch")

    with pytest.raises(RuntimeError, match="final persisted"):
        search(optimizer, store, render, score, guard=guard)
    assert (
        optimizer.completed == 0
        and len(list((tmp_path / "scores").glob("*.json"))) == 8
    )
    assert store.query_counts()["fitness"]["attempted_queries"] == 8

    def no_queries(images):
        raise AssertionError("completed queries must be reused")

    restored = store.restore()
    search(restored, store, render, no_queries)
    assert (
        restored.completed == 8
        and store.query_counts()["fitness"]["attempted_queries"] == 8
    )


def test_resource_guard_preserves_prior_peaks_and_enforces_reserved_memory(
    tmp_path, monkeypatch
):
    import json
    from types import SimpleNamespace

    from evolutionary_extensions.experiments.detector_search import runner

    monkeypatch.setattr(runner, "guard_disk", lambda config: None)
    cuda = SimpleNamespace(
        max_memory_allocated=lambda: 100, max_memory_reserved=lambda: 200
    )
    runner.guard_resources({}, tmp_path, cuda)
    cuda.max_memory_allocated = lambda: 50
    runner.guard_resources({}, tmp_path, cuda)
    saved = json.loads((tmp_path / "memory.json").read_text())
    assert saved["peak_allocated_gpu_bytes"] == 100 and saved["checks"] == 2
    cuda.max_memory_reserved = lambda: 64 * 1024**3
    with pytest.raises(RuntimeError, match="reserved"):
        runner.guard_resources({}, tmp_path, cuda)
    assert json.loads((tmp_path / "memory.json").read_text())["passed"] is False


def test_ssp_heldout_patch_coordinates_survive_json_and_reject_actual_drift(tmp_path):
    import json

    from PIL import Image

    from evolutionary_extensions.experiments.detector_evaluation.data import (
        patch_coordinates,
    )
    from evolutionary_extensions.experiments.detector_search.scoring import (
        SearchClassifier,
    )
    from evolutionary_extensions.persistence.packaging import hashes

    image = tmp_path / "image.png"
    Image.new("RGB", (512, 512), (30, 90, 120)).save(image)
    expected = json.loads(json.dumps(patch_coordinates("heldout-subject", seed=42)))
    row = {
        "image_id": "heldout-subject",
        "subject": "heldout-subject",
        "path": image.name,
        "sha256": hashes(image)["sha256"],
        "codec": "png",
        "scorer_context": {"patch_coordinates": expected},
    }
    scorer = object.__new__(SearchClassifier)
    scorer.detector, scorer.root, scorer.config = "ssp", tmp_path, {"patch_seed": 42}
    scorer.patch_identity = None
    scorer.transform = lambda patch: torch.zeros((3, 32, 32))
    tensor, details = scorer.prepare([row])
    assert tensor.shape == (1, 3, 32, 32)
    assert details[0]["coordinates"] == expected
    assert (
        json.loads((tmp_path / "patches/heldout-subject-png-patch42.json").read_text())[
            "coordinates"
        ]
        == expected
    )
    expected[0][0] = (expected[0][0] + 1) % 481
    with pytest.raises(ValueError, match="coordinates differ"):
        scorer.prepare([row])
