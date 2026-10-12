"""Fixed-noise SDXL batches with frozen actual embeddings and image receipts."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from evolutionary_extensions.persistence.packaging import atomic_json, hashes


class SearchRenderer:
    def __init__(self, root, target, geometry, assets, *, seed=0):
        from evolutionary_prompt_embedding.image_creation import (
            SDXLPromptEmbeddingImageCreator,
        )

        from .generator_validation import load_generator

        self.root, self.target, self.geometry = Path(root), target, geometry
        self.ledger = self.root / "generation_attempts"
        self.ledger.mkdir(exist_ok=True)
        attempts = sorted(self.ledger.glob("*.json"))
        if [p.name for p in attempts] != [
            f"{i:06d}.json" for i in range(len(attempts))
        ]:
            raise ValueError("Generation attempt ledger is not contiguous")
        self.next_attempt = len(attempts)
        if seed != 0:
            raise ValueError("Optimization uses fixed diffusion noise seed zero")
        pipe = load_generator(assets)
        self.creator = SDXLPromptEmbeddingImageCreator(
            1, 1, fixed_noise_seeds=[seed], pipeline=pipe
        )

    def __call__(self, indices, proposals):
        import torch
        from safetensors.torch import load_file, save_file

        from evolutionary_prompt_embedding.argument_types import PooledPromptEmbedData

        if (
            len(indices) != 4
            or indices != list(range(indices[0], indices[0] + 4))
            or indices[0] % 4
            or proposals.shape != (4, int(self.geometry.active.sum()))
        ):
            raise ValueError("Require four ordered distinct candidate proposals")
        digest = hashlib.sha256(
            np.asarray(proposals, dtype=np.float64).tobytes()
        ).hexdigest()
        receipt = self.root / "generation" / f"batch-{indices[0]:05d}.json"
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved["indices"] != indices or saved["proposal_sha256"] != digest:
                raise ValueError("Saved generation proposals differ")
            if [row["candidate_index"] for row in saved["images"]] != indices:
                raise ValueError("Saved generation image order differs")
            if (
                hashes(self.root / saved["embedding_path"])["sha256"]
                != saved["embedding_sha256"]
            ):
                raise ValueError("Saved candidate embeddings changed")
            for row in saved["images"]:
                if hashes(self.root / row["path"])["sha256"] != row["sha256"]:
                    raise ValueError("Saved candidate image changed")
            return saved["images"]
        arguments = [self.geometry.decode(proposal) for proposal in proposals]
        tensors = {
            "prompt_embeds": torch.cat([a.prompt_embeds.cpu() for a in arguments]),
            "pooled_prompt_embeds": torch.cat(
                [a.pooled_prompt_embeds.cpu() for a in arguments]
            ),
        }
        relative = Path("embeddings") / f"batch-{indices[0]:05d}.safetensors"
        frozen = self.root / relative
        frozen.parent.mkdir(exist_ok=True)
        if frozen.exists():
            previous = load_file(str(frozen))
            if set(previous) != set(tensors) or any(
                not torch.equal(previous[k], tensors[k]) for k in tensors
            ):
                raise ValueError("Pending actual FP16 embeddings differ")
        else:
            temporary = frozen.with_suffix(".tmp")
            save_file(tensors, str(temporary))
            temporary.replace(frozen)
        # The generator consumes the saved FP16 tensors, rather than an
        # unarchived in-memory approximation of the CMA proposals.
        arguments = [
            PooledPromptEmbedData(
                tensors["prompt_embeds"][i : i + 1],
                tensors["pooled_prompt_embeds"][i : i + 1],
            )
            for i in range(4)
        ]
        intent = self.ledger / f"{self.next_attempt:06d}.json"
        atomic_json(
            intent,
            {
                "indices": indices,
                "proposal_sha256": digest,
                "physical_images": 4,
                "state": "started",
                "started_unix": time.time(),
            },
        )
        self.next_attempt += 1
        started = time.perf_counter()
        outputs = self.creator.create_solutions(arguments)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        images = []
        for slot, (index, solution) in enumerate(zip(indices, outputs)):
            image_id = f"{self.target}-{index:05d}"
            path = Path("images") / f"{image_id}.png"
            output = self.root / path
            output.parent.mkdir(exist_ok=True)
            temporary = output.with_suffix(".tmp")
            solution.result.images[0].save(temporary, format="PNG")
            temporary.replace(output)
            images.append(
                {
                    "image_id": image_id,
                    "path": path.as_posix(),
                    "sha256": hashes(output)["sha256"],
                    "codec": "png",
                    "candidate_index": index,
                    "embedding_path": relative.as_posix(),
                    "embedding_slot": slot,
                }
            )
        record = {
            "indices": indices,
            "proposal_sha256": digest,
            "embedding_path": relative.as_posix(),
            "embedding_sha256": hashes(frozen)["sha256"],
            "images": images,
            "render_seconds": seconds,
        }
        atomic_json(receipt, record)
        atomic_json(
            intent,
            {
                **json.loads(intent.read_text()),
                "state": "complete",
                "receipt": receipt.name,
                "render_seconds": seconds,
                "finished_unix": time.time(),
            },
        )
        return images
