"""Projection and viewer adapters on a small, mixed-image fixture."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import torch

from evolutionary.evolution_base import SolutionCandidate
from evolutionary_prompt_embedding.analysis import (
    ProjectionConfig,
    _representation,
    available_representations,
    compute_projection,
    create_demo_archive,
    inspect_record,
    load_records,
)
from evolutionary_prompt_embedding.archive import (
    EmbeddingArchiveReader,
    EmbeddingArchiveWriter,
)
from evolutionary_prompt_embedding.argument_types import PromptEmbedData
from evolutionary_prompt_embedding.viewers import (
    LocalViewerServer,
    atlas_table,
    show_3d,
    show_atlas,
    show_gallery,
)


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = create_demo_archive()
        cls.archive = EmbeddingArchiveReader(cls.root)
        cls.records = load_records([cls.archive])
        cls.cache = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        import shutil

        shutil.rmtree(cls.root)
        cls.cache.cleanup()

    def test_representations_use_correct_axes(self):
        tensors = {
            "prompt_embeds": torch.arange(24).reshape(2, 1, 3, 4),
            "pooled_prompt_embeds": torch.arange(4).reshape(2, 1, 2),
        }
        self.assertEqual(_representation(tensors, "token").shape, (2, 12))
        self.assertEqual(_representation(tensors, "pooled").shape, (2, 2))
        self.assertEqual(_representation(tensors, "combined_append").shape, (2, 14))
        np.testing.assert_array_equal(
            _representation(tensors, "combined_avg")[:, :4],
            [[4, 5, 6, 7], [16, 17, 18, 19]],
        )

    def test_all_projections_no_dropped_records_and_neighbors(self):
        for representation in available_representations([self.archive]):
            for algorithm in ("pca", "umap", "tsne"):
                for dimensions in (2, 3):
                    config = ProjectionConfig(
                        representation=representation,
                        algorithm=algorithm,
                        dimensions=dimensions,
                        pca_components=4,
                        batch_size=5,
                    )
                    result = compute_projection(
                        [self.archive], self.records, config, self.cache.name
                    )
                    self.assertEqual(
                        result.record_id.tolist(), self.records.record_id.tolist()
                    )
                    axes = ["x", "y"] + (["z"] if dimensions == 3 else [])
                    self.assertTrue(np.isfinite(result[axes]).all().all())
                    for i, row in result.iterrows():
                        self.assertNotIn(row.record_id, row.neighbors)
                        self.assertEqual(
                            len(row.neighbors), len(row.neighbor_distances)
                        )
                        self.assertIn("PCA analysis space", row.neighbor_space)
                    cached = compute_projection(
                        [self.archive], self.records, config, self.cache.name
                    )
                    np.testing.assert_array_equal(cached[axes], result[axes])

    def test_cache_and_display_filter_stability(self):
        config = ProjectionConfig(algorithm="pca", pca_components=4)
        first = compute_projection(
            [self.archive], self.records, config, self.cache.name
        )
        filtered = first[first.generation == 1]
        np.testing.assert_array_equal(
            filtered[["x", "y"]], first.loc[filtered.index, ["x", "y"]]
        )
        changed = compute_projection(
            [self.archive], self.records, replace(config, seed=43), self.cache.name
        )
        self.assertNotEqual(
            first.projection_key.iloc[0], changed.projection_key.iloc[0]
        )
        subset = compute_projection(
            [self.archive], self.records.iloc[:12], config, self.cache.name
        )
        self.assertNotEqual(first.projection_key.iloc[0], subset.projection_key.iloc[0])
        record, tensors = inspect_record([self.archive], first.record_id.iloc[0])
        self.assertEqual(tuple(tensors["prompt_embeds"].shape), (1, 4, 8))
        self.assertEqual(record["generation"], 0)

    def test_cached_projection_follows_moved_archive(self):
        import shutil

        with tempfile.TemporaryDirectory() as directory:
            moved = Path(directory) / "moved"
            # Copy the already-cached archive, retaining its immutable record IDs.
            config = ProjectionConfig(algorithm="pca")
            original = compute_projection([self.archive], self.records, config)
            shutil.copytree(self.root, moved)
            reader = EmbeddingArchiveReader(moved)
            records = load_records([reader])
            cached = compute_projection([reader], records, config)
            self.assertEqual(set(cached.archive_dir), {str(moved.resolve())})
            np.testing.assert_array_equal(original[["x", "y"]], cached[["x", "y"]])

    def test_optional_pooled_compatibility_and_tiny_datasets(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = EmbeddingArchiveWriter(directory, {"model": "token-only"})
            population = [
                SolutionCandidate(PromptEmbedData(torch.randn(1, 2, 4)), None)
                for _ in range(4)
            ]
            writer.write_generation(population, 0)
            archive = EmbeddingArchiveReader(directory)
            records = load_records([archive])
            self.assertEqual(available_representations([archive]), ("token",))
            with self.assertRaisesRegex(ValueError, "unavailable"):
                compute_projection([archive], records)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                load_records([self.archive, archive])
            with self.assertRaisesRegex(ValueError, "requires at least"):
                compute_projection(
                    [archive],
                    records.iloc[:2],
                    ProjectionConfig(representation="token", algorithm="pca"),
                )
            with self.assertRaisesRegex(ValueError, "at least four"):
                compute_projection(
                    [archive],
                    records.iloc[:3],
                    ProjectionConfig(representation="token", algorithm="umap"),
                )
            result = compute_projection(
                [archive],
                records.iloc[:3],
                ProjectionConfig(representation="token", algorithm="pca"),
            )
            self.assertEqual(len(result), 3)

    def test_local_images_atlas_gallery_and_3d(self):
        projection = compute_projection(
            [self.archive],
            self.records,
            ProjectionConfig(algorithm="pca", dimensions=3),
            self.cache.name,
        )
        with LocalViewerServer([self.archive]) as server:
            adapted = atlas_table(projection, server)
            self.assertIn("No images", adapted.image_status.tolist())
            self.assertIn("Some images missing", adapted.image_status.tolist())
            with urlopen(adapted.image.dropna().iloc[0]) as response:
                self.assertEqual(response.headers.get_content_type(), "image/png")
            with self.assertRaises(HTTPError):
                urlopen(server.base_url + "/../../manifest.json")
            # Images can disappear while a notebook remains open.
            transient_image = Path(self.cache.name) / "transient.png"
            transient_image.write_bytes(b"fixture")
            transient_url = server.register(transient_image, "image/png")
            transient_image.unlink()
            with self.assertRaises(HTTPError) as missing_image:
                urlopen(transient_url)
            self.assertEqual(missing_image.exception.code, 404)
            widget = show_atlas(projection, server)
            self.assertEqual(len(widget.selection()), len(projection))
            self.assertEqual(widget._props["data"]["image"], "image")
            for row in widget.selection().itertuples():
                self.assertEqual(
                    len(row.neighbors["ids"]), len(row.neighbors["distances"])
                )
            missing = adapted[
                adapted.image_status == "Some images missing"
            ].record_id.iloc[0]
            self.assertIn(
                "file missing", show_gallery(projection, server, missing).data
            )
            # Keep automated tests offline; browser smoke checks use the real pinned asset.
            asset = Path(self.cache.name) / "fake-deck.js"
            asset.write_text("/* test */")
            with patch(
                "evolutionary_prompt_embedding.viewers.prepare_deck_asset",
                return_value=asset,
            ):
                view = show_3d(projection, server, self.cache.name, thumbnail_limit=3)
                with urlopen(view.src) as response:
                    html = response.read().decode()
                self.assertIn("deck.OrbitView", html)
                self.assertIn("deck.IconLayer", html)
                payloads = [
                    json.loads(content)
                    for content, mime in server.routes.values()
                    if mime == "application/json"
                ]
                payload = payloads[-1]
                self.assertEqual(len(payload["points"]), len(projection))
                self.assertLessEqual(payload["thumbnail_count"], 3)
                self.assertTrue(any(p["image_count"] == 0 for p in payload["points"]))
                self.assertTrue(any(None in p["images"] for p in payload["points"]))


if __name__ == "__main__":
    unittest.main()
