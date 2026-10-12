import json

import pytest

from evolutionary_extensions.experiments.detector_evaluation.config import digest
from evolutionary_extensions.experiments.detector_search.heldout import (
    HeldoutJournal,
    score_matrix,
)
from evolutionary_extensions.experiments.detector_search.reuse import (
    reuse_search_finalists,
)
from evolutionary_extensions.persistence.packaging import atomic_json, hashes


def frozen_search(root, detector="safe"):
    source = root / detector
    source.mkdir(parents=True)
    image = source / "candidate.png"
    image.write_bytes(b"frozen selected PNG")
    procedure = {
        "detector": detector,
        "asset": {"checkpoint": "frozen"},
        "packages": {"torch": "pinned"},
        "search_source_files": {"scoring.py": "frozen"},
        "batch": 1,
        "reconstruction_seed": 1 if detector == "aeroblade" else None,
        "reconstruction_seed_scope": "independently reset per image"
        if detector == "aeroblade"
        else None,
    }
    recipe = {
        "target": detector,
        "budget": 6400,
        "assets": procedure["asset"],
        "packages": procedure["packages"],
        "source_files": procedure["search_source_files"],
    }
    identity = digest(recipe)
    value = {
        "identity": identity,
        "index": 0,
        "image": {
            "path": "candidate.png",
            "sha256": hashes(image)["sha256"],
            "codec": "png",
        },
        "raw": 0.2,
        "synthetic_score": 0.2,
        "objective": -1.2,
    }
    if detector == "ssp":
        patch = source / "patch.png"
        patch.write_bytes(b"immutable selected patch")
        value.update(
            synthetic_score=0.8,
            coordinates=[[0, 0]] * 64,
            selected_index=0,
            patch_path="patch.png",
            patch_sha256=hashes(patch)["sha256"],
        )
    if detector == "aeroblade":
        value.update(
            synthetic_score=-0.2,
            objective=-0.2,
            reconstruction_errors=[0.2, 0.3, 0.4],
            reconstruction_seed=1,
            reconstruction_seed_scope=procedure["reconstruction_seed_scope"],
        )
    atomic_json(source / "recipe.json", {"identity": identity, "recipe": recipe})
    atomic_json(
        source / "scoring-procedure.json",
        {"search_identity": identity, "procedure": procedure},
    )
    atomic_json(
        source / "search_complete.json", {"status": "complete", "candidates": 6400}
    )
    atomic_json(source / "parity.json", {"passed": True})
    atomic_json(source / "loading.json", {"scoring_batch": 1})
    atomic_json(source / "scores/00000.json", value)
    atomic_json(source / "finalists.json", {"identity": identity, "records": [value]})
    heldout = root / "heldout"
    heldout.mkdir()
    (heldout / image.name).write_bytes(image.read_bytes())
    row = {
        "subject": f"{detector}-00000",
        "target": detector,
        "candidate_index": 0,
        "diffusion_seed": 0,
        **value["image"],
    }
    if detector == "ssp":
        row["scorer_context"] = {"patch_coordinates": value["coordinates"]}
    return heldout, row, procedure, value


def test_exact_external_match_preserves_receipts_and_spends_no_query(tmp_path):
    root, row, procedure, original = frozen_search(tmp_path)
    journal = HeldoutJournal(root, "safe", {"effective_procedure": procedure}, [row])
    assert reuse_search_finalists(journal, tmp_path, procedure) == 1
    records = score_matrix(
        journal, lambda _: pytest.fail("completed inference repeated")
    )
    assert records[0]["value"]["raw"] == original["raw"]
    proof = records[0]["reused_from"]
    assert proof["score_sha256"] == hashes(tmp_path / proof["score_path"])["sha256"]
    assert journal.counts()["attempted_queries"] == 0
    assert reuse_search_finalists(journal, tmp_path, procedure) == 0


@pytest.mark.parametrize("field", ["batch", "packages", "asset", "search_source_files"])
def test_external_configuration_difference_cannot_reuse(tmp_path, field):
    root, row, procedure, _ = frozen_search(tmp_path)
    changed = {**procedure, field: "different"}
    journal = HeldoutJournal(root, "safe", {"effective_procedure": changed}, [row])
    assert reuse_search_finalists(journal, tmp_path, changed) == 0
    assert journal.existing(row) is None


def test_external_ssp_requires_actual_coordinates_and_preserves_patch(tmp_path):
    root, row, procedure, value = frozen_search(tmp_path, "ssp")
    changed = {**row, "scorer_context": {"patch_coordinates": [[1, 0]] * 64}}
    journal = HeldoutJournal(root, "ssp", procedure, [row, changed])
    assert reuse_search_finalists(journal, tmp_path, procedure) == 1
    assert journal.existing(changed) is None
    imported = journal.existing(row)["value"]
    assert hashes(root / imported["patch_path"])["sha256"] == value["patch_sha256"]
    (tmp_path / "ssp/patch.png").write_bytes(b"changed source patch")
    another = HeldoutJournal(root, "ssp", {**procedure, "audit": 2}, [row])
    with pytest.raises(ValueError, match="patch changed"):
        reuse_search_finalists(another, tmp_path, procedure)


def test_external_changed_receipt_and_reconstruction_scope_fail(tmp_path):
    root, row, procedure, value = frozen_search(tmp_path, "aeroblade")
    journal = HeldoutJournal(root, "aeroblade", procedure, [row])
    changed = {**value, "reconstruction_seed_scope": "shared stream"}
    atomic_json(tmp_path / "aeroblade/scores/00000.json", changed)
    with pytest.raises(ValueError, match="provenance differs"):
        reuse_search_finalists(journal, tmp_path, procedure)
    atomic_json(
        tmp_path / "aeroblade/finalists.json",
        {"identity": value["identity"], "records": [changed]},
    )
    with pytest.raises(ValueError, match="seed scope differs"):
        reuse_search_finalists(journal, tmp_path, procedure)
    atomic_json(tmp_path / "aeroblade/scores/00000.json", value)
    atomic_json(
        tmp_path / "aeroblade/finalists.json",
        {"identity": value["identity"], "records": [value]},
    )
    assert reuse_search_finalists(journal, tmp_path, procedure) == 1
    frozen = json.loads((tmp_path / "aeroblade/recipe.json").read_text())
    atomic_json(tmp_path / "aeroblade/recipe.json", {**frozen, "identity": "changed"})
    with pytest.raises(ValueError, match="not verified"):
        reuse_search_finalists(journal, tmp_path, procedure)


def test_calibration_only_evaluator_is_not_an_optimizer_reuse_source(tmp_path):
    root, row, procedure, _ = frozen_search(tmp_path, "aeroblade")
    folder = tmp_path / "aeroblade"
    (folder / "search_complete.json").unlink()
    (folder / "scores/00000.json").unlink()
    (folder / "finalists.json").unlink()
    atomic_json(
        folder / "calibration_complete.json",
        {"status": "complete", "optimizer_candidates": 0},
    )
    journal = HeldoutJournal(root, "aeroblade", procedure, [row])
    assert reuse_search_finalists(journal, tmp_path, procedure) == 0
    assert journal.existing(row) is None
    atomic_json(folder / "scores/00000.json", {})
    with pytest.raises(ValueError, match="candidate evidence"):
        reuse_search_finalists(journal, tmp_path, procedure)
