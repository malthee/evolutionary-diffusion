# Detector-directed embedding search

The `evolutionary_extensions.experiments.detector_search` package searches SDXL-Turbo prompt embeddings with the reusable [separable CMA-ES adapter](sep_cma.md) in `evolutionary/algorithms/sep_cma.py`. Install `.[cma]` or use `uv sync --extra cma`; the older `SearchOptimizer` class path remains compatible with existing owned checkpoints. Detector weights remain frozen. Image generation, fitness measurements and subsequent robustness measurements have separate receipts and query ledgers.

## Configuration

The managed runner consumes a JSON configuration with:

- `benchmark_config` and `benchmark_completion`: the verified baseline assets, frozen image pool and completed benchmark receipt.
- `targets`: a nonempty, distinct subset of the supported detectors. Omission selects all supported targets. Selection is frozen before optimization; evaluation detectors remain independent of optimization targets.
- `algorithm`, `budget`, `population`, `seed`, `sigma` and `generation_batch`: the validated search recipe.
- `output`: a new campaign directory.
- `managed`: the frozen repository, existing Jupyter server/root, isolated kernel names and lifecycle owner.

The currently validated recipe uses `sep-cma-es`, 6,400 candidates per selected target, population 64, seed 42, sigma 0.15 and generation batches of four. The total budget is derived from the selected targets. Source, model, environment and configuration changes require a new output directory.
Search and managed manifests include the reusable optimizer and embedding-coordinate
implementation hashes, even though those modules live outside the experiment package.

## Execution and recovery

Run `python -m evolutionary_extensions.experiments.detector_search.managed --config CONFIG` in the separately supervised Azure session. This module does not own compute start or stop. Deployment, lifecycle supervision and campaign-specific authorization belong outside the public package.

Every generator environment must first pass identical-pixel, repetition, offline reload and resource checks against the frozen reference. Each selected target then passes native detector and batch checks before fitness queries. Excluded targets cannot enter the optimization runner.

The optimizer searches the full coordinate bounds, repairs proposals to FP16 and preserves its private RNG. Pending populations, generated images, scores and query intents are persisted. Resume reuses completed score receipts and evaluates only missing candidates. Unfinished query intents remain conservative upper bounds on physical work; they do not replace evidence of completed candidate coverage.

Allocated and reserved GPU peaks are checked before and after persisted batches. Resource failure preserves the evaluated batch before stopping new work. Checkpoints are owned serialized state and must only be restored after their identity and checksum have been verified.

## Held-out evaluation and reporting

Finalists are frozen before compression, new diffusion seeds, new SSP patch pools or cross-detector measurements. These measurements never feed back into the optimizer. All supported evaluation detectors score the same selected images and unselected controls, including detectors excluded from optimization.

Independent reconstruction randomness requires matching AEROBLADE calibration. If AEROBLADE is excluded from optimization, a separate calibration-only notebook produces that reference without optimizer candidates.

Reporting validates the full frozen factorial and query coverage before exporting numeric tables, figures, an image gallery and optimizer trajectories. Rates use Wilson intervals; robustness differences use subject-paired bootstrap resampling. Intervals describe the selected image sets, rather than independent repeated optimizer runs.

CPU reporting and transfer may be retried without rerunning inference. An inference marker, export marker and final archived completion are distinct states. Final archival requires actual visual/numerical inspection and verified compute shutdown through the campaign's lifecycle supervisor.

Campaign-specific detector selection, personal schedules, host paths, budgets beyond the validated recipe and operational recovery history belong in ignored local configuration and provenance.
