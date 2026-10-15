# Comparing SDXL-Turbo embedding bounds across prompt datasets

These experiments extend `sdxl_turbo.ipynb` with one notebook per dataset. The
original SD/SDXL notebooks, library APIs and packaged clamp tensors are unchanged.
Only text is encoded: no images are downloaded or generated.

## Research question

How do the observed minimum and maximum at each SDXL embedding coordinate differ
between Parti P2 and alternative prompt datasets? Report overall min/max too,
separately for token and pooled embeddings. Here “tensor span” means coordinate
intervals and their widths, not a linear-algebraic span or dimensional rank.

## Selected sources

| Dataset | Source and rationale | Loading and license declaration |
|---|---|---|
| Parti P2 | [nateraw/parti-prompts](https://huggingface.co/datasets/nateraw/parti-prompts), Google’s diverse text-to-image benchmark; 1,632 prompts. | `PartiPrompts.tsv`, `Prompt` field. Apache-2.0 listed. Full fresh baseline. |
| GenAI-Bench | [BaiqiL/GenAI-Bench](https://huggingface.co/datasets/BaiqiL/GenAI-Bench), 1,600 professional-designer prompts with basic and advanced compositional skills. Useful for realistic descriptive prompts near Parti’s sample size. | English `prompt` in the author’s [genai_image.json](https://huggingface.co/datasets/BaiqiL/GenAI-Bench-1600/blob/main/genai_image.json), plus skill metadata. Main dataset card declares Apache-2.0; companion JSON repository does not repeat it. |
| T2I-CompBench | [NinaKarine/t2i-compbench](https://huggingface.co/datasets/NinaKarine/t2i-compbench), mirror of the [author dataset](https://github.com/Karine-Huang/T2I-CompBench). Tests colour, shape, texture, spatial/non-spatial relationships and complex compositions. | Six original categories, `train` and `val`, 6,000 rows before deduplication. Load `text` only; exclude later 3D/numeracy/expanded variants. MIT listed. Split labels are retained in source IDs; this study does not train a model. |
| DiffusionDB | [poloclub/diffusiondb](https://huggingface.co/datasets/poloclub/diffusiondb), real Stable Diffusion user prompts, including keyword/style-heavy language. Complements curated benchmark text. | Official 2M `metadata.parquet` (~195 MB), `prompt` and `image_name` columns only. Deduplicate before uniform sampling. CC0-1.0 listed. Avoid the obsolete dataset-loading script and multi-terabyte image archives. |
| DrawBench | [sayakpaul/drawbench](https://huggingface.co/datasets/sayakpaul/drawbench), 200 curated generation challenges. A small diagnostic dataset rather than an equally broad substitute for Parti. | CSV `Prompts` and `Category` fields. Apache-2.0 listed. |
| GenEval | [djghosh13/geneval](https://github.com/djghosh13/geneval), 553 object/count/colour/position prompts. Useful for measuring bounds from controlled short prompts. | Official `prompts/evaluation_metadata.jsonl`, `prompt` and `tag` fields. MIT repository license; no detector or image-evaluation dependencies needed. |

Pinned revisions are defined in `dataset_bounds.py` and recorded in every result.
License declarations are reported as source metadata rather than a new legal
assessment. Dataset text is treated as input data, never as executable instructions.

### Other candidates considered

- [HPS/HPDv2](https://huggingface.co/datasets/zhwang/HPDv2) offers 3,200 benchmark
  prompts in four visual styles. The [code repository](https://github.com/tgxs002/HPSv2)
  is Apache-2.0, but the Hugging Face dataset card does not explicitly declare a
  dataset license. Not selected for this first five-dataset study.
- [Gustavosta/Stable-Diffusion-Prompts](https://huggingface.co/datasets/Gustavosta/Stable-Diffusion-Prompts)
  contains 81,910 Lexica-derived prompts, but lists its license as unknown.
  DiffusionDB supplies a documented alternative for real user prompts.
- Original Pick-a-Pic v2 and its text-only source returned HTTP 401 during this
  research. Avoid replacing them with unverified image-heavy mirrors.
- DPG-Bench is relevant to long descriptions, but includes expanded Parti-derived
  prompts and substantial 77-token truncation would complicate this first comparison.

## Execution

Use the repository’s existing virtual environment, with its installed notebook,
PyTorch, Transformers, Datasets, PyArrow, pandas and Matplotlib packages:

```sh
.venv/bin/python -m unittest discover -s pre-testing/clamp_range -p 'test_*.py' -v
.venv/bin/python pre-testing/clamp_range/run_notebooks.py
```

The runner executes Parti, the five alternatives, then the comparison notebook
serially. It supplies a temporary kernelspec pointing at the invoking Python,
so execution does not accidentally use a different notebook environment.
On macOS, the process needs local GPU access for `torch.backends.mps.is_available()`
to detect MPS; a restricted sandbox may report it unavailable. The encoder selects
MPS when available and CPU otherwise. Every comparison must use the same device.

To execute an individual notebook (Parti results must exist before comparison):

```sh
.venv/bin/python pre-testing/clamp_range/run_notebooks.py --one drawbench
.venv/bin/python pre-testing/clamp_range/run_notebooks.py --one comparison
```

Both single-notebook and sequence execution enforce the same 60-minute
wall-clock limit per notebook and save runner logs under `results/logs/`.

`--start diffusiondb` executes that notebook and all subsequent notebooks. It does
not resume a partly encoded dataset. Re-running a dataset replaces its artifacts.

## Method and runtime limits

- Load only the two SDXL tokenizers and float16 text encoders from pinned
  `stabilityai/sdxl-turbo`; preserve the production creator’s penultimate hidden
  states, concatenation and second encoder’s pooled projection.
- Verify encoder parity against the production method on short, descriptive and
  truncated text. The check reuses those text components and avoids loading an
  unused UNet/VAE. All per-dataset parity results are recorded.
- Remove blank and exact duplicate text while preserving the first source ID and
  unmodified prompt. Do not rewrite, filter by language, or exclude Parti overlap.
- Measure three warm-up prompts and 32 timed prompts with MPS synchronization.
  The sample cap is the remaining 50-minute working budget divided by the measured
  95th-percentile prompt time times 1.25. Account for source/model preparation time.
- Uniformly shuffle unique prompts with seed 42, then select the budgeted prefix.
  Small datasets are fully evaluated if their projected runtime fits. Sampling is
  from the entire deduplicated source, rather than the first downloaded rows.
- Main reductions encode each selected prompt once and collect token and pooled
  min/max together. Warm-up/parity probes are separate calibration work.
- Save running bounds and progress every 100 prompts. Stop normally at 55 minutes;
  the runner enforces a 60-minute wall-clock limit per notebook. Partial results
  are explicitly labelled and do not stand in for a full Parti baseline.
- Model outputs remain float16; convert to float32 before subtraction, aggregation
  and comparison. Shapes are `(1, 77, 2048)` and `(1, 1280)`.

## Results and interpretation

`results/REPORT.md` is the numerical readout. `sdxl_turbo_comparison.ipynb` contains
summary tables and plots. `results/comparison.csv` and `comparison.json` contain
all-coordinate, Parti-variable-coordinate and encoder-block metrics, for both
full/budgeted samples and equal-count samples. `run_manifest.json` records the
source/model revisions, versions, device, timings and sample identity.

Each dataset folder contains:

- `bounds.pt`: CPU float32 lower/upper tensors for token and pooled embeddings.
- `matched_bounds.pt`: bounds from the first `min(1632, encoded_count)` prompts.
- `bounds_differences.pt`: lower/upper/span deltas at every token and pooled
  coordinate, for both full/budgeted and equal-count comparisons.
- `summary.json`: progress/status, coverage, diagnostics and bounds growth.
- `selected_prompts.jsonl`: exact source IDs/text/categories in encoding order,
  retained locally and excluded from Git to avoid large corpus copies.
- `bounds_difference.png`, `pooled_bounds_difference.png` and `bounds_growth.png`:
  per-coordinate differences versus full Parti and growth of summed intervals.

Parti’s local float16 embedding cache is excluded from Git (~0.52 GB) and permits
equal-count comparisons without repeating model inference. For smaller datasets,
use the same number of prompts from the seeded Parti order. `parti_packaged_drift.json`
compares the fresh baseline with the original packaged tensors separately.

The saved original Parti global token extremes occur at constant first-token
coordinates. Overall extremes can therefore remain identical while other bounds
change considerably. Keep the per-coordinate comparison central. A positive
span-sum change measures a wider observed axis-aligned interval, not better image
quality, semantic coverage or a justified replacement clamp. Larger samples can
increase extrema; single equal-count samples do not estimate sampling uncertainty.
The production 77-token truncation and padding are preserved and truncation rates
are reported independently for each tokenizer. No production clamp defaults are
changed automatically.
