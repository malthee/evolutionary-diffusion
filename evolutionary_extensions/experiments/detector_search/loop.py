"""Transactional ask/render/score/tell loop, independent of GPU implementations."""

import hashlib
import json
import math
from pathlib import Path

from evolutionary_extensions.persistence.packaging import atomic_json


def search(optimizer, store, render, score, *, batch=4, progress=None, guard=None):
    """Callbacks preserve candidate order; completed receipts skip paid scoring.

    ``render(indices, proposals)`` must itself persist images/embeddings before
    returning hash-bound paths. ``score(images)`` returns native outputs and a
    continuous minimization objective. Both are invoked only for pending entries.
    """
    if type(batch) is not int or batch < 1 or optimizer.population % batch:
        raise ValueError("Render batch must divide the frozen population")
    progress = progress or (lambda **details: None)
    guard = guard or (lambda: None)
    fresh_queries = 0
    while optimizer.completed < optimizer.budget:
        proposals = optimizer.ask()
        store.checkpoint(optimizer)
        pending_checkpoint = json.loads((store.root / "checkpoint.json").read_text())
        scores = []
        for offset in range(0, optimizer.population, batch):
            guard()
            indices = list(
                range(
                    optimizer.completed + offset, optimizer.completed + offset + batch
                )
            )
            images = render(indices, proposals[offset : offset + batch])
            if len(images) != len(indices):
                raise ValueError("Generation order/count differs from proposals")
            records = [store.score(i, image) for i, image in zip(indices, images)]
            missing = [j for j, record in enumerate(records) if record is None]
            if missing:
                pending_indices = [indices[j] for j in missing]
                pending_images = [images[j] for j in missing]
                intent = store.begin_queries(
                    pending_indices,
                    pending_images,
                    detector_calls=3 if store.target == "aeroblade" else 1,
                )
                output = score(pending_images)
                store.finish_queries(intent, output)
                fresh_queries += len(pending_images)
                records = [store.score(i, image) for i, image in zip(indices, images)]
            # Persist completed queries before rejecting another batch for resources.
            guard()
            scores.extend(record["objective"] for record in records)
            progress(
                completed_candidates=optimizer.completed + offset + batch,
                committed_candidates=optimizer.completed,
                query_counts=store.query_counts(),
                fresh_queries_this_process=fresh_queries,
            )
        atomic_json(
            store.root
            / "generations"
            / f"{optimizer.completed + optimizer.population:05d}.json",
            {
                "completed_candidates": optimizer.completed + optimizer.population,
                "objectives": scores,
                "best": min(scores),
                "pending_checkpoint_sha256": pending_checkpoint["sha256"],
                "proposal_sha256": hashlib.sha256(proposals.tobytes()).hexdigest(),
            },
        )
        optimizer.tell(scores)
        store.checkpoint(optimizer)
    return optimizer


def freeze_finalists(store, count=20):
    """Select distinct pixel hashes by fitness only, before held-out inference."""
    if type(count) is not int or count < 1:
        raise ValueError("Require a positive finalist count")
    records = [
        json.loads(p.read_text())
        for p in sorted((store.root / "scores").glob("*.json"))
    ]
    if len(records) != store.budget or {r["index"] for r in records} != set(
        range(store.budget)
    ):
        raise ValueError("Finalist selection requires the complete candidate budget")
    for record in records:
        if store.score(record["index"], record["image"]) != record or not math.isfinite(
            record["objective"]
        ):
            raise ValueError("Invalid finalist evidence")
    unique, seen = [], set()
    for record in sorted(records, key=lambda r: (r["objective"], r["index"])):
        sha = record["image"]["sha256"]
        if sha not in seen:
            seen.add(sha)
            unique.append(record)
        if len(unique) == count:
            break
    result = {
        "identity": store.identity,
        "selection": "fitness then candidate index; distinct PNG hashes",
        "requested": count,
        "selected": len(unique),
        "records": unique,
    }
    path = Path(store.root) / "finalists.json"
    if path.exists() and json.loads(path.read_text()) != result:
        raise ValueError("Refuse to change frozen held-out finalist selection")
    atomic_json(path, result)
    return result
