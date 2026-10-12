import hashlib
import json

import pytest

from evolutionary_extensions.experiments.detector_search.heldout import (
    HeldoutJournal,
    score_matrix,
    validate_value,
)


def image_rows(root):
    path = root / "image.png"
    path.write_bytes(b"immutable pixels")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return [
        {"subject": subject, "codec": codec, "path": path.name, "sha256": sha}
        for subject, codec in (("a", "png"), ("b", "png"), ("a", "jpeg95"))
    ]


def test_pixel_alias_reuses_score_but_codec_and_configuration_remain_distinct(tmp_path):
    rows = image_rows(tmp_path)
    calls = []

    def score(pool):
        calls.extend((r["subject"], r["codec"]) for r in pool)
        return [{"raw": 0.2, "synthetic_score": 0.2} for _ in pool]

    journal = HeldoutJournal(tmp_path, "safe", {"checkpoint": "frozen"}, rows)
    records = score_matrix(journal, score, batch=4)
    assert calls == [("a", "png"), ("a", "jpeg95")]
    assert len(records) == 3 and records[1]["reused_from"] is not None
    assert journal.counts()["attempted_queries"] == 2
    recovered = HeldoutJournal(tmp_path, "safe", {"checkpoint": "frozen"}, rows)
    assert (
        score_matrix(recovered, lambda _: pytest.fail("completed inference repeated"))
        == records
    )
    different = HeldoutJournal(tmp_path, "safe", {"checkpoint": "different"}, rows)
    assert all(different.existing(row) is None for row in rows)
    with pytest.raises(ValueError, match="recipe changed"):
        HeldoutJournal(tmp_path, "safe", {"checkpoint": "frozen"}, rows[:1])


def test_interrupted_batch_counts_uncertain_attempt_and_resumes_only_missing_rows(
    tmp_path,
):
    rows = image_rows(tmp_path)
    rows = [rows[0], rows[2]]
    journal = HeldoutJournal(tmp_path, "bfree", {}, rows)
    intent = journal.begin(rows)
    journal.write(rows[0], {"raw": -2.0, "synthetic_score": -2.0})
    recovered = HeldoutJournal(tmp_path, "bfree", {}, rows)
    called = []

    def score(pool):
        called.extend(pool)
        return [{"raw": 4.0, "synthetic_score": 4.0} for _ in pool]

    records = score_matrix(recovered, score)
    assert called == [rows[1]] and len(records) == 2
    assert json.loads(intent.read_text())["state"] == "started"
    assert recovered.counts()["attempted_queries"] == 3
    assert recovered.counts()["unfinished_queries"] == 2
    assert recovered.counts()["component_calls"] == 3
    with pytest.raises(ValueError, match="must not be queried"):
        recovered.begin([rows[0]])


def test_nonfinite_batch_publishes_nothing_and_counts_all_three_autoencoders(tmp_path):
    rows = image_rows(tmp_path)[::2]
    journal = HeldoutJournal(tmp_path, "aeroblade", {}, rows)
    intent = journal.begin(rows)
    valid = {
        "raw": 0.1,
        "synthetic_score": -0.1,
        "reconstruction_errors": [0.1, 0.2, 0.3],
    }
    with pytest.raises(ValueError, match="Non-finite"):
        journal.finish(intent, [valid, {**valid, "raw": float("nan")}])
    assert all(journal.existing(row) is None for row in rows)
    assert journal.counts()["component_calls"] == 6
    with pytest.raises(ValueError, match="errors differ"):
        validate_value("aeroblade", {**valid, "reconstruction_errors": [0.2, 0.3, 0.4]})
    with pytest.raises(ValueError, match="polarity"):
        validate_value("ssp", {"raw": 0.1, "synthetic_score": 0.1})


def test_ssp_actual_coordinates_control_reuse_and_changed_pixels_fail(tmp_path):
    rows = image_rows(tmp_path)[:2]
    for i, row in enumerate(rows):
        row["scorer_context"] = {"patch_coordinates": [[i, 0]] * 64}
    journal = HeldoutJournal(tmp_path, "ssp", {"patch_seed": 42}, rows)
    assert journal.score_key(rows[0]) != journal.score_key(rows[1])
    with pytest.raises(ValueError, match="all 64"):
        journal.score_key(
            {**rows[0], "scorer_context": {"patch_coordinates": [[0, 0]]}}
        )
    with pytest.raises(ValueError, match="patch provenance"):
        journal.write(
            rows[0],
            {
                "raw": 0.1,
                "synthetic_score": 0.9,
                "coordinates": [[1, 0]] * 64,
                "selected_index": 0,
            },
        )
    (tmp_path / rows[0]["path"]).write_bytes(b"changed pixels")
    with pytest.raises(ValueError, match="image changed"):
        journal.existing(rows[0])


def render_fixture(root):
    from PIL import Image

    subjects = []
    for i in range(5):
        path = root / f"source-{i}.png"
        Image.new("RGB", (512, 512), (i, 2, 3)).save(path)
        subjects.append(
            {
                "subject": f"subject-{i}",
                "group": "optimized",
                "target": "safe",
                "embedding_slot": i,
                "noise0_path": path.name,
                "noise0_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    embedding = root / "embeddings.safetensors"
    embedding.write_bytes(b"frozen embedding fixture; no GPU model needed")
    return {
        "subjects": subjects,
        "embedding_sha256": hashlib.sha256(embedding.read_bytes()).hexdigest(),
    }


def test_heldout_rendering_reuses_noise_zero_and_counts_physical_padding(tmp_path):
    from PIL import Image

    from evolutionary_extensions.experiments.detector_search.heldout_rendering import (
        render_variants,
        write_variants,
    )

    recipe = render_fixture(tmp_path)
    calls = []

    def render(subjects, seed):
        calls.append((seed, [s["embedding_slot"] for s in subjects]))
        return [
            Image.new("RGB", (512, 512), (s["embedding_slot"], seed, 5))
            for s in subjects
        ]

    rows = render_variants(tmp_path, recipe, render)
    assert len(rows) == 60
    assert calls == [
        (seed, slots) for seed in (1, 2, 3) for slots in ([0, 1, 2, 3], [4, 4, 4, 4])
    ]
    for row in rows:
        if row["diffusion_seed"] == 0 and row["codec"] == "png":
            assert (
                row["sha256"]
                == recipe["subjects"][row["embedding_slot"]]["noise0_sha256"]
            )
    ledger = [
        json.loads(p.read_text())
        for p in (tmp_path / "generation_attempts").glob("*.json")
    ]
    assert sum(r["physical_images"] for r in ledger) == 24
    assert sum(r["distinct_images"] for r in ledger) == 15
    assert sum(r["padding_images"] for r in ledger) == 9
    assert (
        render_variants(
            tmp_path, recipe, lambda *_: pytest.fail("completed rendering repeated")
        )
        == rows
    )
    with pytest.raises(ValueError, match="Unsafe"):
        write_variants(
            Image.new("RGB", (512, 512)), tmp_path, {"subject": "../escape"}, 0
        )
    with pytest.raises(ValueError, match="JPEG source pixels"):
        write_variants(
            Image.new("RGB", (512, 512)),
            tmp_path,
            recipe["subjects"][0],
            0,
            png_source=tmp_path / recipe["subjects"][0]["noise0_path"],
        )


def test_heldout_rendering_recovery_preserves_completed_batches(tmp_path):
    from PIL import Image

    from evolutionary_extensions.experiments.detector_search.heldout_rendering import (
        render_variants,
    )

    recipe = render_fixture(tmp_path)
    count = 0

    def interrupted(subjects, seed):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("interrupted generation")
        return [
            Image.new("RGB", (512, 512), (s["embedding_slot"], seed, 5))
            for s in subjects
        ]

    with pytest.raises(RuntimeError, match="interrupted"):
        render_variants(tmp_path, recipe, interrupted)
    resumed = []

    def render(subjects, seed):
        resumed.append((seed, [s["embedding_slot"] for s in subjects]))
        return [
            Image.new("RGB", (512, 512), (s["embedding_slot"], seed, 5))
            for s in subjects
        ]

    rows = render_variants(tmp_path, recipe, render)
    assert len(rows) == 60 and len(resumed) == 5
    assert resumed[0] == (1, [4, 4, 4, 4])
    assert (
        json.loads((tmp_path / "generation_attempts/000001.json").read_text())["state"]
        == "started"
    )
    (tmp_path / rows[0]["path"]).write_bytes(b"changed completed pixels")
    with pytest.raises(ValueError, match="checksum differs"):
        render_variants(tmp_path, recipe, render)


def test_ssp_heldout_pools_are_subject_paired_and_patch_artifacts_do_not_collide(
    tmp_path,
):
    torch = pytest.importorskip("torch")
    from PIL import Image

    from evolutionary_extensions.experiments.detector_evaluation.data import (
        patch_coordinates,
    )
    from evolutionary_extensions.experiments.detector_search.scoring import (
        HeldoutClassifier,
    )

    source = tmp_path / "image.png"
    Image.new("RGB", (512, 512), (1, 2, 3)).save(source)
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    # Exercise native CPU patch extraction independently of CUDA model loading.
    scorer = object.__new__(HeldoutClassifier)
    scorer.root, scorer.detector, scorer.config = tmp_path, "ssp", {"patch_seed": 42}
    scorer.transform = lambda image: torch.tensor(image.getpixel((0, 0)))
    rows = [
        {
            "image_id": f"a-noise{seed}",
            "subject": "a",
            "codec": codec,
            "path": source.name,
            "sha256": sha,
            "scorer_context": {"patch_coordinates": patch_coordinates("a", seed=42)},
        }
        for seed, codec in ((0, "png"), (0, "jpeg95"), (1, "png"))
    ]
    _, details = scorer.prepare(rows)
    assert all(d["coordinates"] == details[0]["coordinates"] for d in details)
    assert len({d["patch_path"] for d in details}) == 3
    for detail in details:
        assert (tmp_path / detail["patch_path"]).exists()
    scorer.config = {"patch_seed": 43}
    with pytest.raises(ValueError, match="coordinates differ"):
        scorer.prepare(rows[:1])
    changed = {
        **rows[0],
        "scorer_context": {"patch_coordinates": patch_coordinates("a", seed=43)},
    }
    _, extra = scorer.prepare([changed])
    assert extra[0]["coordinates"] != details[0]["coordinates"]
    assert extra[0]["patch_path"] not in {d["patch_path"] for d in details}


def test_heldout_inputs_freeze_actual_selected_and_control_embedding_slots(tmp_path):
    torch = pytest.importorskip("torch")
    from PIL import Image
    from safetensors.torch import load_file, save_file

    from evolutionary_extensions.experiments.detector_evaluation.config import DETECTORS
    from evolutionary_extensions.experiments.detector_search.heldout_rendering import (
        prepare_inputs,
    )
    from evolutionary_extensions.experiments.detector_search.protocol import (
        freeze_heldout,
        heldout_plan,
    )
    from evolutionary_extensions.persistence.packaging import atomic_json, hashes

    search, benchmark = tmp_path / "search", tmp_path / "benchmark"
    search.mkdir()
    benchmark.mkdir()
    baseline = {
        "prompt_embeds": torch.arange(40, dtype=torch.float16)
        .reshape(40, 1, 1)
        .expand(40, 77, 2048)
        .contiguous(),
        "pooled_prompt_embeds": torch.arange(40, dtype=torch.float16)
        .reshape(40, 1)
        .expand(40, 1280)
        .contiguous(),
    }
    save_file(baseline, str(benchmark / "embeddings.safetensors"))
    atomic_json(
        benchmark / "generation.json",
        {"embedding_sha256": hashes(benchmark / "embeddings.safetensors")["sha256"]},
    )
    rows = []
    for i in range(40):
        path = benchmark / f"control-{i}.png"
        Image.new("RGB", (512, 512), (i, 0, 0)).save(path)
        rows.append(
            {
                "image_id": f"{'random' if i < 20 else 'prompted'}-{i:04d}",
                "group": "random" if i < 20 else "prompted",
                "codec": "png",
                "embedding_index": i,
                "path": path.name,
                "sha256": hashes(path)["sha256"],
            }
        )
    finalists = {}
    for i, target in enumerate(DETECTORS):
        folder = search / target
        folder.mkdir()
        path = folder / "candidate.png"
        Image.new("RGB", (512, 512), (100 + i, 0, 0)).save(path)
        image = {
            "image_id": f"{target}-00000",
            "candidate_index": 0,
            "codec": "png",
            "path": path.name,
            "sha256": hashes(path)["sha256"],
            "embedding_path": "embeddings.safetensors",
            "embedding_slot": 0,
        }
        tensors = {k: torch.full_like(v[:4], 100 + i) for k, v in baseline.items()}
        save_file(tensors, str(folder / "embeddings.safetensors"))
        atomic_json(
            folder / "generation/batch-00000.json",
            {
                "embedding_path": "embeddings.safetensors",
                "embedding_sha256": hashes(folder / "embeddings.safetensors")["sha256"],
                "images": [image],
            },
        )
        finalists[target] = {"records": [{"index": 0, "image": image}]}
        atomic_json(folder / "finalists.json", finalists[target])
        atomic_json(
            folder / "search_complete.json", {"status": "complete", "candidates": 6400}
        )
    freeze_heldout(search / "heldout_plan.json", heldout_plan(finalists, rows))
    root, recipe = prepare_inputs(search, benchmark, rows)
    frozen = load_file(str(root / "embeddings.safetensors"))
    assert frozen["prompt_embeds"].shape == (45, 77, 2048)
    assert torch.equal(
        frozen["prompt_embeds"][:5, 0, 0], torch.arange(100, 105, dtype=torch.float16)
    )
    assert torch.equal(frozen["prompt_embeds"][5:], baseline["prompt_embeds"])
    assert prepare_inputs(search, benchmark, rows) == (root, recipe)
    (search / "safe/candidate.png").write_bytes(b"changed finalist pixels")
    with pytest.raises(ValueError, match="checksum differs"):
        prepare_inputs(search, benchmark, rows)


def test_selected_targets_keep_full_transfer_matrix_and_control_cohorts():
    from evolutionary_extensions.experiments.detector_evaluation.config import DETECTORS
    from evolutionary_extensions.experiments.detector_search.protocol import (
        heldout_plan,
        search_targets,
    )

    controls = [
        {
            "image_id": f"{g}-{i:03d}",
            "group": g,
            "codec": "png",
            "embedding_index": i,
            "sha256": f"{g}-{i}",
        }
        for g in ("random", "prompted")
        for i in range(20)
    ]
    finalists = {
        t: {"records": [{"index": 0, "image": {"sha256": t}}]}
        for t in ("universal_fake_detect", "bfree")
    }
    plan = heldout_plan(finalists, controls)
    assert plan["targets"] == ["universal_fake_detect", "bfree"]
    assert plan["detectors"] == list(DETECTORS)
    assert len(plan["subjects"]) == 42 and len(plan["variants"]) == 504
    assert plan["logical_primary_matrix_scores"] == 2520
    assert {s["target"] for s in plan["subjects"] if s["group"] == "optimized"} == set(
        finalists
    )
    for value in ([], ["safe", "safe"], ["dire"], "safe"):
        with pytest.raises(ValueError, match="targets"):
            search_targets({"targets": value})


def test_excluded_target_is_rejected_before_native_or_fitness_queries(
    tmp_path, monkeypatch
):
    from evolutionary_extensions.experiments.detector_search import runner

    config = tmp_path / "config.json"
    config.write_text(json.dumps({"targets": ["universal_fake_detect", "bfree"]}))
    monkeypatch.setattr(
        runner,
        "verify_prerequisite",
        lambda _: pytest.fail("Excluded target admitted before prerequisite gate"),
    )
    with pytest.raises(ValueError, match="excluded"):
        runner.run_target(config, "safe")
