import random
import threading
from types import SimpleNamespace

import pytest
import torch

from evolutionary_extensions.experiments.prompt_embedding.artifacts import (
    BufferedAttemptWriter,
)
from evolutionary_extensions.experiments.prompt_embedding.config import ExperimentConfig
from evolutionary_prompt_embedding.archive import (
    EmbeddingArchiveReader,
    EmbeddingArchiveWriter,
)


def items(start, count):
    candidates = [
        SimpleNamespace(
            arguments=SimpleNamespace(
                prompt_embeds=torch.full((1, 2, 3), float(i), dtype=torch.float16),
                pooled_prompt_embeds=torch.full((1, 3), float(i), dtype=torch.float16),
            ),
            fitness=float(i),
        )
        for i in range(start, start + count)
    ]
    labels = [
        {
            "evaluation_id": i,
            "successful": i % 2 == 0,
            "parents": [i - 1],
            "experiment_id": "experiment",
        }
        for i in range(start, start + count)
    ]
    return candidates, [[] for _ in candidates], labels


def test_grouping_order_immutability_rng_and_reader_compatibility(tmp_path):
    writer = BufferedAttemptWriter(EmbeddingArchiveWriter(tmp_path))
    python_state = random.getstate()
    torch_state = torch.get_rng_state().clone()
    for first in range(1, 41, 4):
        candidates, images, labels = items(first, 4)
        writer.submit(0, candidates, images, labels)
        for candidate in candidates:
            candidate.arguments.prompt_embeds.zero_()
        labels[0]["parents"].append(999)
    writer.submit(1, *items(41, 3))
    writer.close()
    reader = EmbeddingArchiveReader(tmp_path)
    decoded = []
    for rows, tensors in reader.iter_batches(batch_size=7):
        for i, row in enumerate(rows):
            identity = row["metadata"]["evaluation_id"]
            assert tensors["prompt_embeds"][i].eq(identity).all()
            assert row["metadata"]["parents"] == [identity - 1]
            decoded.append(identity)
    assert decoded == list(range(1, 44))
    assert [s["expected_count"] for s in reader.manifest["snapshots"]] == [32, 8, 3]
    assert [s["generation"] for s in reader.manifest["snapshots"]] == [0, 0, 1]
    assert writer.peak_pending <= 32
    assert random.getstate() == python_state
    assert torch.equal(torch.get_rng_state(), torch_state)
    writer.close()
    with pytest.raises(RuntimeError, match="closed"):
        writer.submit(2, *items(44, 1))


def test_admission_blocks_at_limit_without_dropping_attempts(tmp_path):
    archive = EmbeddingArchiveWriter(tmp_path)
    original = archive.write_generation
    started, release, admitted = threading.Event(), threading.Event(), threading.Event()

    def delayed(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    archive.write_generation = delayed
    writer = BufferedAttemptWriter(archive, limit=4)
    writer.submit(0, *items(1, 4))
    assert started.wait(5)

    def producer():
        writer.submit(0, *items(5, 2))
        admitted.set()

    thread = threading.Thread(target=producer)
    thread.start()
    assert writer.pending_count == 4
    assert not admitted.is_set()
    release.set()
    thread.join(5)
    assert admitted.is_set()
    writer.close()
    assert writer.peak_pending == 4
    assert len(list(EmbeddingArchiveReader(tmp_path).iter_records())) == 6


def test_write_failure_propagates_and_preserves_uncommitted_cpu_payload(tmp_path):
    archive = EmbeddingArchiveWriter(tmp_path)

    def failed(*args, **kwargs):
        raise OSError("archive write failed")

    archive.write_generation = failed
    writer = BufferedAttemptWriter(archive, limit=4)
    writer.submit(7, *items(1, 4))
    with pytest.raises(OSError, match="archive write failed"):
        writer.close()
    assert list(EmbeddingArchiveReader(tmp_path).iter_records()) == []
    recovery = torch.load(tmp_path / "pending_attempts.pt", weights_only=False)
    assert [row[2]["evaluation_id"] for row in recovery["rows"]] == [1, 2, 3, 4]
    assert all(
        row[0].arguments.prompt_embeds.device.type == "cpu" for row in recovery["rows"]
    )
    writer.close()


def test_scalar_archive_and_buffered_archive_have_identical_scientific_records(
    tmp_path,
):
    synchronous = EmbeddingArchiveWriter(tmp_path / "sync")
    asynchronous = BufferedAttemptWriter(EmbeddingArchiveWriter(tmp_path / "async"))
    for generation, first, count in [(0, 1, 10), (1, 11, 23), (2, 34, 5)]:
        for offset in range(0, count, 4):
            batch = items(first + offset, min(4, count - offset))
            synchronous.write_generation(
                batch[0],
                generation,
                image_paths=batch[1],
                metadata=batch[2],
                snapshot_id=str(first + offset),
            )
            asynchronous.submit(generation, *batch)
        asynchronous.drain()
    asynchronous.close()

    def scientific(path):
        output = []
        for rows, tensors in EmbeddingArchiveReader(path).iter_batches(batch_size=1):
            row = rows[0]
            output.append(
                (
                    row["generation"],
                    row["fitness"],
                    row["metadata"],
                    {key: value.tolist() for key, value in tensors.items()},
                )
            )
        return output

    assert scientific(tmp_path / "sync") == scientific(tmp_path / "async")


@pytest.mark.parametrize("value", [1, None, "true"])
def test_buffered_configuration_requires_boolean(value):
    with pytest.raises(TypeError, match="buffered_attempt_writes"):
        ExperimentConfig(buffered_attempt_writes=value)


def test_recovery_save_failure_preserves_original_archive_error(tmp_path, monkeypatch):
    archive = EmbeddingArchiveWriter(tmp_path)

    def fail_archive(*args, **kwargs):
        raise OSError("original archive error")

    def fail_recovery(*args, **kwargs):
        raise RuntimeError("recovery storage failed")

    archive.write_generation = fail_archive
    monkeypatch.setattr(torch, "save", fail_recovery)
    writer = BufferedAttemptWriter(archive)
    writer.submit(0, *items(1, 2))
    with pytest.raises(OSError, match="original archive error"):
        writer.close()
