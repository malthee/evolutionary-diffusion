import json

import pytest
from PIL import Image

from evolutionary_extensions.experiments.detector_evaluation.config import digest
from evolutionary_extensions.experiments.detector_search.generator_validation import (
    PASSES,
    render_pass,
    verify_receipt,
)
from evolutionary_extensions.persistence.packaging import atomic_json, hashes


@pytest.fixture
def validated(tmp_path):
    recipe = {
        "fixture": "ordered FP16 embeddings with four independently reset noise seeds"
    }
    runtime = {"fixture": "CPU pixels, no CUDA claim"}

    def render(indices, seed):
        return [Image.new("RGB", (512, 512), (i, seed, 17)) for i in indices]

    reference = None
    for stage in ("generate", "aeroblade"):
        folder = tmp_path / stage
        calls = []
        passes = {
            label: render_pass(folder, label, render, after_batch=calls.append)
            for label in PASSES
        }
        assert calls == [4] * 48
        receipt = {
            "passed": True,
            "stage": stage,
            "recipe": recipe,
            "runtime": runtime,
            "reference_sha256": reference,
            "passes": passes,
            "known_physical_images": sum(calls),
            "candidate_budget_used": 0,
            "peak_allocated_bytes": 100,
            "peak_reserved_bytes": 200,
        }
        atomic_json(folder / "parity.json", receipt)
        sha = hashes(folder / "parity.json")["sha256"]
        atomic_json(
            folder / "attempts/0000.json",
            {
                "state": "complete",
                "stage": stage,
                "receipt_sha256": sha,
                "recipe_sha256": digest(recipe),
                "runtime": runtime,
                "known_physical_images": 192,
                "maximum_physical_images": 192,
                "candidate_budget_used": 0,
            },
        )
        if stage == "generate":
            reference = sha
    return tmp_path, recipe, runtime


def rewrite(folder, update):
    path = folder / "parity.json"
    value = json.loads(path.read_text())
    update(value)
    atomic_json(path, value)
    intent_path = folder / "attempts/0000.json"
    intent = json.loads(intent_path.read_text())
    intent["receipt_sha256"] = hashes(path)["sha256"]
    atomic_json(intent_path, intent)


def test_exact_pixels_and_physical_count_cover_all_noise_seeds_and_reloads(validated):
    root, recipe, runtime = validated
    proof = verify_receipt(root, "aeroblade", recipe, runtime)
    assert proof["known_physical_images"] == 192 and proof["candidate_budget_used"] == 0
    assert len(proof["passes"]["initial"]) == 64
    # A different software stack is admissible only when its decoded pixels match.
    other_runtime = {"fixture": "different stack, same CPU pixels"}
    rewrite(root / "aeroblade", lambda r: r.update(runtime=other_runtime))
    p = root / "aeroblade/attempts/0000.json"
    intent = json.loads(p.read_text())
    atomic_json(p, {**intent, "runtime": other_runtime})
    verify_receipt(root, "aeroblade", recipe, other_runtime)
    with pytest.raises(ValueError, match="runtime changed"):
        verify_receipt(root, "aeroblade", recipe, runtime)


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda r: r["passes"].pop("offline_reload"), "offline"),
        (lambda r: r["passes"]["initial"].reverse(), "coverage/order"),
        (lambda r: r.update(known_physical_images=191), "accounting"),
        (lambda r: r.update(candidate_budget_used=1), "accounting"),
        (lambda r: r.update(peak_reserved_bytes=64 * 1024**3), "memory"),
        (lambda r: r.update(reference_sha256="0" * 64), "reference receipt"),
    ],
)
def test_incomplete_or_changed_gate_never_authorizes_search(validated, mutation, error):
    root, recipe, runtime = validated
    rewrite(root / "aeroblade", mutation)
    with pytest.raises(ValueError, match=error):
        verify_receipt(root, "aeroblade", recipe, runtime)


def test_hash_bound_pixels_and_completed_ledger_are_required(validated):
    root, recipe, runtime = validated
    path = root / "aeroblade/parity.json"
    row = json.loads(path.read_text())["passes"]["repeat"][0]
    Image.new("RGB", (512, 512), "red").save(root / "aeroblade" / row["path"])
    with pytest.raises(ValueError, match="image bytes"):
        verify_receipt(root, "aeroblade", recipe, runtime)
    path = root / "aeroblade/attempts/0000.json"
    intent = json.loads(path.read_text())
    atomic_json(path, {**intent, "state": "started"})
    with pytest.raises(ValueError, match="ledger"):
        verify_receipt(root, "aeroblade", recipe, runtime)


def test_resaving_changed_pixels_cannot_hide_cross_stack_difference(validated):
    root, recipe, runtime = validated
    folder = root / "aeroblade"

    def mutate(r):
        row = r["passes"]["initial"][0]
        image = Image.new("RGB", (512, 512), "red")
        image.save(folder / row["path"])
        from evolutionary_extensions.experiments.detector_search.generator_validation import (
            pixel_hash,
        )

        row.update(
            sha256=hashes(folder / row["path"])["sha256"], rgb_sha256=pixel_hash(image)
        )

    rewrite(folder, mutate)
    with pytest.raises(ValueError, match="decoded pixels differ"):
        verify_receipt(root, "aeroblade", recipe, runtime)


def test_diagnostic_batch_refuses_wrong_size_mode_and_count(tmp_path):
    with pytest.raises(ValueError, match="order/count"):
        render_pass(tmp_path, "count", lambda indices, seed: [])
    for mode, size in [("L", (512, 512)), ("RGB", (256, 256))]:
        with pytest.raises(ValueError, match="RGB"):
            render_pass(
                tmp_path,
                mode,
                lambda indices, seed, mode=mode, size=size: [Image.new(mode, size)] * 4,
            )


def test_generator_query_audit_preserves_failed_intents_outside_candidate_budget(
    validated,
):
    import shutil

    from evolutionary_extensions.experiments.detector_evaluation.config import DETECTORS
    from evolutionary_extensions.experiments.detector_search.generator_validation import (
        audit_generator_validation,
    )

    root, _recipe, _runtime = validated
    for stage in DETECTORS:
        if stage == "aeroblade":
            continue
        shutil.copytree(root / "aeroblade", root / stage)
        rewrite(root / stage, lambda r, stage=stage: r.update(stage=stage))
        path = root / stage / "attempts/0000.json"
        item = json.loads(path.read_text())
        atomic_json(path, {**item, "stage": stage})
    folder = root / "generator_validation"
    folder.mkdir()
    for stage in ("generate", *DETECTORS):
        (root / stage).rename(folder / stage)
    atomic_json(
        folder / "aeroblade/attempts/0001.json",
        {
            "stage": "aeroblade",
            "state": "failed",
            "known_physical_images": 4,
            "maximum_physical_images": 192,
            "candidate_budget_used": 0,
        },
    )
    audit = audit_generator_validation(root)
    assert audit["known_physical_images"] == 1156
    assert audit["maximum_attempted_physical_images"] == 1344
    assert audit["candidate_budget_used"] == 0
    atomic_json(
        folder / "aeroblade/attempts/0001.json",
        {
            "stage": "aeroblade",
            "state": "failed",
            "known_physical_images": 193,
            "maximum_physical_images": 192,
            "candidate_budget_used": 0,
        },
    )
    with pytest.raises(ValueError, match="intent differs"):
        audit_generator_validation(root)
