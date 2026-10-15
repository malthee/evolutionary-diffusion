# Pre-testing and research notebooks

These notebooks investigate models, bounds and scientific recipes. Ready-to-run
algorithm examples are in [notebooks](../notebooks/README.md).

| Location | Purpose |
|---|---|
| [prompt_embedding/prompt_embedding_experiment.ipynb](prompt_embedding/prompt_embedding_experiment.ipynb) | DiffusionDB-initialized GA/OSGA campaign recipe with optional Drive export; disabled by default |
| [detector_evaluation/detector_evaluation.ipynb](detector_evaluation/detector_evaluation.ipynb) | Frozen five-detector benchmark; disabled by default |
| [detector_evaluation/aide_evaluation.ipynb](detector_evaluation/aide_evaluation.ipynb) | SDXL detector, SSP and official/experimental DIRE comparison |
| [embedding_analysis/embedding_analysis.ipynb](embedding_analysis/embedding_analysis.ipynb) | Offline projection and embedding viewers; separate analysis environment |
| [evaluators/mo_evaluation.ipynb](evaluators/mo_evaluation.ipynb) | Multi-objective image metrics, hypervolume and radar charts |
| [clamp_range/](clamp_range/README.md) | Empirical prompt-embedding bounds, pinned datasets and bounded notebook execution |
| Existing evaluator and variation notebooks | Individual model and operator probes |

Moved notebook contents and execution guards are retained. Config profiles and
documentation use the new paths. Open Jupyter from the checkout root or the
notebook's folder; configure image/run paths before inference or analysis.

See [campaign configuration](../docs/configuration.md),
[detector evaluation](../docs/detector_evaluation.md),
[detector search](../docs/detector_search.md) and
[embedding analysis](../docs/embedding_analysis.md). Keep private profiles,
models, operational scripts and result evidence in ignored local directories.
