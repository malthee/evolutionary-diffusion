# Embedding archives and local analysis

Experiments record original embeddings once. Visualization, dimensionality
reduction and viewer-specific conversion run later in
[`../embedding_analysis.ipynb`](../embedding_analysis.ipynb).

## Run an experiment

The GA, OSGA, NSGA and island GA notebooks create a unique directory under
`embedding_runs/` (inside the configured Colab Drive base path, when mounted).
`save_embeddings=True` records every evaluated population, including generation
zero and the final generation. `save_images=True` independently saves original
PNGs. Disable either or both without changing the algorithm. Re-run the setup
cell to create a new experiment directory; a snapshot cannot be overwritten.

OSGA archives contain completed survivor populations, including the last completed
generation on early termination. Rejected offspring and unfinished-generation
attempts remain in the existing evaluation/lineage JSON export; they are not
mislabelled as completed generation snapshots. NSGA-II/III/U-NSGA-III archive the
evaluated survivors after parent/offspring survival, with full objective vectors.
The archive records the existing run configuration and evolution/diffusion seeds.

Images retain the existing `images/<generation>/` filename convention so image
grids and videos continue to work. Records contain ordered image paths per
candidate, including zero or multiple images. The Parti embedding-relations
notebook produces a generation-zero dataset with prompt/category metadata and
no images. Other experiment checkpoints remain independent.

For a custom callback:

```python
from evolutionary_prompt_embedding.archive import (
    EmbeddingArchiveWriter, create_run_directory,
)
import evolutionary_imaging.processing as ip

run_dir = create_run_directory("embedding_runs")
ip.RESULTS_FOLDER = str(run_dir / "images")
writer = EmbeddingArchiveWriter(run_dir, run_metadata={
    "model": {"id": "my-model", "revision": "resolved-revision"},
    "configuration": {"population_size": 200},
    "seeds": {"torch": 42},
})

def record_generation(generation, algorithm):
    writer.check_snapshot_available(generation)
    paths = ip.save_images_from_generation_grouped(algorithm.population, generation)
    writer.write_generation(algorithm.population, generation, image_paths=paths)
    # Use image_paths=None when saving embeddings without images.
```

Attach this callback to `post_evaluation_callback`, including for NSGA. In current
NSGA variants this hook runs after survivor selection; do not attach the same
recorder to both evaluation and sorting hooks. Island callbacks supply `island_id`
and optional per-candidate metadata.

## Archive contract

```text
run-<uuid>/
  manifest.json
  embeddings/
    g<generation>-i<island>-<uuid>.safetensors
    g<generation>-i<island>-<uuid>.jsonl
  images/<generation>/*.png                 # optional
  analysis_cache/<projection-hash>.parquet  # derived, removable
  analysis_cache/<projection-hash>.json
  analysis_cache/deck.gl-9.4.0.min.js        # lazily downloaded, checksum verified
```

Schema version 1 stores `prompt_embeds` and optional `pooled_prompt_embeds` in
Safetensors. Each shard adds a candidate axis: an original `(1, 77, 2048)` tensor
becomes `(N, 1, 77, 2048)`. Reading an individual record restores its original
shape and dtype. All candidates in a run must share tensor shapes, dtypes and
pooled availability. No flattened or averaged copies are persisted by experiments.

JSONL records contain snapshot IDs, generation, candidate slot, island, original
fitness (scalar/vector/null), optional metadata, relative image paths, and tensor
file/row references. Run metadata stores model identity, configuration, seeds and
objective names where supplied. A null model revision means it could not be
resolved from the creator; it must not be treated as a pinned revision. Snapshot
IDs do not claim individual identity or lineage across generations/migration.

Tensor files are capped at 256 MiB including their header. A smaller cap can be
supplied to the writer; a single oversized candidate is rejected. CPU buffers
are allocated only for the current shard. Temporary serialization buffers can
add a constant multiple of shard size, but no tensors accumulate across generations.
Writing is synchronous and single-writer per run directory; concurrent processes
must use different run roots. Writers hold no tensors, open handles or locks and
can be pickled with callbacks.

Only manifest-referenced shards are committed. Both tensor and metadata files
have SHA-256 checksums. Readers ignore temporary/orphan files and can read
committed shards from an interrupted snapshot. Its manifest reports `writing`
and expected/committed counts; subsequent generations can be recorded, but the
interrupted snapshot cannot be overwritten. Persistence errors propagate.

SDXL token `(77, 2048)` plus pooled `(1280,)` tensors use 317,952 bytes per candidate
in FP16. A 200 × 100 run therefore uses **5.92 GiB** of raw tensors; FP32 uses
**11.85 GiB**. PNGs and metadata are additional. Do not downcast original tensors
to achieve these numbers. Safetensors is uncompressed and supports partial reads.

## Install the isolated analysis environment

Run from the repository root with Python 3.13 and `uv`:

```bash
uv venv .venv-analysis --python 3.13
uv pip sync --python .venv-analysis/bin/python notebooks/embedding_analysis/requirements.lock
uv pip install --python .venv-analysis/bin/python --no-deps -e .
.venv-analysis/bin/python -m jupyterlab notebooks/embedding_analysis.ipynb
```

Select the analysis environment's kernel. The universal lock includes platform
markers; it is separate from the inference environment and its Torch/Transformers
pins. Regenerate it after deliberately changing `requirements.in`:

```bash
uv pip compile notebooks/embedding_analysis/requirements.in --python-version 3.13 --universal --output-file notebooks/embedding_analysis/requirements.lock
```

Copy the complete experiment run directory from Colab/Drive to the local machine,
then set `ARCHIVE_PATHS` in the notebook. Empty paths run a small synthetic example
without model downloads. The example includes no-image, multiple-image and
missing-file records. Its temporary directory is printed for later cleanup.

## Projection and viewer behavior

Choose `token`, `pooled`, `combined_avg` (mean over the token axis plus pooled),
or `combined_append` (all token coordinates plus pooled). Pooled choices require
pooled tensors. Cross-run analysis requires matching tensor specifications and
model identities. Metadata filters select the population before projection;
display filters retain the fitted coordinates.
Unknown model revisions cannot establish that different runs used identical
weights; supply resolved revisions when comparing such runs.

All native tensors are read in batches and converted to FP32 only for analysis.
IncrementalPCA centres inputs without standardization or vector normalization.
The default pipeline is combined-append → up to 64 PCA components → UMAP, seed
42, Euclidean metric, 15 neighbours (reduced for small datasets), minimum distance
0.1. PCA and t-SNE are also available in 2D/3D. t-SNE can be expensive; no automatic
sampling occurs. Unsupported tiny datasets receive an explicit error.

Parquet caches contain coordinates, record IDs, metadata and up to ten nearest
neighbours, with a JSON descriptor recording source manifests, parameters and
library versions. Neighbour distances refer to PCA analysis space. Cache keys
change with the source, representation, seed or parameters. `REPROJECT=True`
recomputes; deleting caches never removes original tensors. Cache reload updates
archive locations so moved run directories continue to work.

`VIEWER="atlas"` provides the native Atlas widget with 2D charts, table, filtering,
neighbours and first-image inspection. Select a point and rerun the inspection
cell to access original tensors and the complete image gallery. `VIEWER="3d"`
uses deck.gl orbit controls and thumbnail markers. `"both"` computes separate 2D
and 3D projections from the same selected records.

Every 3D record is a selectable point; up to 2,000 reproducibly selected records
receive thumbnail markers. Thumbnail textures use bounded 2048-pixel pages.
The view shows the marker count, supports colour by fitness/generation/island,
and individual objectives, and provides a keyboard-accessible record selector. Records without images show
metadata/tensor specifications; missing files show placeholders. The notebook
inspection cell exposes the actual original tensors.

A loopback server serves only registered assets and images. Keep it running while
inspecting views; set `CLOSE_VIEWERS=True` when finished. The first 3D launch
requires network access to download the pinned deck.gl 9.4.0 bundle and verify
its SHA-256; subsequent launches use the local asset. Atlas's installed widget
bundle needs no model download. Local GPU/browser capability determines rendering
performance; 3D inspection remains accessible through the record selector if
rendering fails.

No legacy `.pt`/TensorFlow archive conversion, custom semantic axes or additional
image-derived embeddings are included.

## Validation

```bash
# In the inference environment (the grouped image adapter uses imaging dependencies):
python -m unittest discover -s tests -p test_embedding_archive.py -v
# Opt-in: writes/removes a real ~4.06 GiB archive and checks peak memory growth.
EMBEDDING_LARGE_TEST=1 python -m unittest discover -s tests -p test_embedding_archive.py -v
# In the isolated analysis environment:
python -m unittest discover -s tests -p test_embedding_analysis.py -v
```

Sources: [Safetensors](https://huggingface.co/docs/safetensors/index),
[Atlas widget](https://apple.github.io/embedding-atlas/widget.html),
[Atlas data formats](https://apple.github.io/embedding-atlas/data-formats.html),
[deck.gl IconLayer](https://deck.gl/docs/api-reference/layers/icon-layer),
[deck.gl OrbitView](https://deck.gl/docs/api-reference/core/orbit-view).
