"""Import completed search measurements only under identical scoring procedures."""

import json
import shutil
from pathlib import Path

from evolutionary_extensions.experiments.detector_evaluation.config import digest
from evolutionary_extensions.persistence.packaging import hashes

from .heldout import validate_value


def reuse_search_finalists(journal, search_root, procedure):
    """Preserve source receipts and spend no held-out query on an exact match.

    Legacy benchmark receipts have no matching continuous-scorer descriptor.
    A tolerance parity result alone cannot certify their identity. Different
    batches, packages, source hashes or AEROBLADE RNG scopes are not reused.
    """
    search_root = Path(search_root).resolve()
    folder = search_root / journal.detector
    calibration_only = folder / "calibration_complete.json"
    if calibration_only.exists():
        marker = json.loads(calibration_only.read_text())
        if (
            marker.get("status") != "complete"
            or marker.get("optimizer_candidates") != 0
            or (folder / "search_complete.json").exists()
            or any((folder / "scores").glob("*.json"))
        ):
            raise ValueError("Calibration-only evaluator contains candidate evidence")
        return 0
    descriptor = folder / "scoring-procedure.json"
    if not descriptor.exists():
        return 0
    saved = json.loads(descriptor.read_text())
    if saved["procedure"] != procedure:
        return 0
    frozen_path = folder / "recipe.json"
    frozen = json.loads(frozen_path.read_text())
    recipe, identity = frozen["recipe"], frozen["identity"]
    marker = json.loads((folder / "search_complete.json").read_text())
    parity = json.loads((folder / "parity.json").read_text())
    loading = json.loads((folder / "loading.json").read_text())
    if (
        identity != digest(recipe)
        or saved["search_identity"] != identity
        or recipe["target"] != journal.detector
        or recipe["assets"] != procedure["asset"]
        or recipe["packages"] != procedure["packages"]
        or recipe["source_files"] != procedure["search_source_files"]
        or marker.get("status") != "complete"
        or marker.get("candidates") != recipe["budget"]
        or recipe["budget"] != 6400
        or parity.get("passed") is not True
        or loading["scoring_batch"] != procedure["batch"]
    ):
        raise ValueError("External search score configuration is not verified")
    finalists = json.loads((folder / "finalists.json").read_text())
    if finalists["identity"] != identity:
        raise ValueError("External finalist selection identity differs")
    selected = {r["index"]: r for r in finalists["records"]}
    count = 0
    for row in journal.rows.values():
        if (
            journal.existing(row) is not None
            or row.get("target") != journal.detector
            or row.get("diffusion_seed") != 0
            or row["codec"] != "png"
        ):
            continue
        index = row["candidate_index"]
        path = folder / "scores" / f"{index:05d}.json"
        record = json.loads(path.read_text())
        image = record["image"]
        source = folder / image["path"]
        if (
            selected.get(index) != record
            or record["identity"] != identity
            or record["index"] != index
            or image["codec"] != row["codec"]
            or image["sha256"] != row["sha256"]
            or not source.resolve().is_relative_to(folder)
            or hashes(source)["sha256"] != row["sha256"]
        ):
            raise ValueError("External selected search score provenance differs")
        validate_value(journal.detector, record)
        if journal.detector == "aeroblade" and any(
            record.get(key) != procedure[key]
            for key in ("reconstruction_seed", "reconstruction_seed_scope")
        ):
            raise ValueError("External reconstruction seed scope differs")
        if (
            journal.detector == "ssp"
            and record.get("coordinates") != row["scorer_context"]["patch_coordinates"]
        ):
            continue
        value = {
            k: v for k, v in record.items() if k not in {"identity", "index", "image"}
        }
        if "patch_path" in value:
            patch = folder / value["patch_path"]
            if (
                not patch.resolve().is_relative_to(folder)
                or hashes(patch)["sha256"] != value["patch_sha256"]
            ):
                raise ValueError("External selected patch changed")
            relative = (
                Path("patches/reused")
                / journal.detector
                / (value["patch_sha256"] + ".png")
            )
            destination = journal.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if hashes(destination)["sha256"] != value["patch_sha256"]:
                    raise ValueError("Reused selected patch changed")
            else:
                temporary = destination.with_suffix(".tmp")
                shutil.copyfile(patch, temporary)
                temporary.replace(destination)
            value["patch_path"] = relative.as_posix()
        journal.write(
            row,
            value,
            reused_from={
                "phase": "search",
                "score_path": path.relative_to(search_root).as_posix(),
                "score_sha256": hashes(path)["sha256"],
                "recipe_sha256": hashes(frozen_path)["sha256"],
                "procedure_sha256": hashes(descriptor)["sha256"],
            },
        )
        count += 1
    return count
