# Prompt-embedding campaign configuration

Start with [OSGA](../configs/examples/osga.json) or
[GA](../configs/examples/ga.json). Both examples disable execution and Drive,
use one runner and a candidate batch limit of four. These are starting settings,
not measured capacity recommendations. Copy and customize them under `.local/`
or an external directory. No notebook installs dependencies or starts compute.

Install from a checkout with Python 3.11 or newer for campaign execution:

```sh
python -m pip install -e '.[imaging,prompt_embedding,execution]'
python -m evolutionary_extensions.execution.campaign prepare \
  --config configs/examples/osga.json --output .local/prepared/campaign
```

`prepare` freezes random distinct 32-bit seeds once and writes a disabled resolved
configuration and trial manifest. Explicit `campaign.seeds` supports replay,
including repeated seeds. Preserve the resolved file; preparation refuses an
existing output directory. The core library continues to support Python 3.10;
campaign execution uses Python 3.11 asyncio APIs. Offline analysis has a
[separate locked environment](embedding_analysis.md#install-the-isolated-analysis-environment).

| Section | Contents |
|---|---|
| `experiment` | Algorithm/objective, population and evaluation budgets, seeds, pinned model, batching, selection and variation |
| `deployment` | Local cache/output locations, device, bounds override and disk reserves |
| `campaign` | Resolved seeds, optional trial controls and explicit time policy |
| `execution` | Existing Jupyter URL/root, kernel, template, runner count and disabled/enabled switch |
| `drive` | Optional external credentials/folder and explicit verified deletion switch |

Repository-relative deployment paths are resolved against the experiment checkout.
Set `EVOLUTIONARY_CHECKOUT` when using an exported or installed source tree.
The runner needs an existing managed Jupyter server, a kernel with the inference
dependencies, and a `jupyter_root` containing the campaign notebook paths.
Once configured, explicitly set `execution.enabled` in the private resolved file:

```sh
python -m evolutionary_extensions.execution.campaign campaign \
  --config .local/prepared/campaign/resolved_config.json
```

The execution layer closes its owned sessions. Compute start/stop, resource
identity, idle policies and lifecycle supervision belong to external operations.
It performs no Azure lifecycle actions. Never replace a running kernel's source
or rerun completed inference to recover a transfer.

## Recipe and optional controls

The examples use population 64, at most 100 populations including initialization,
6,400 fresh evaluations, one-step FP16 SDXL-Turbo at a pinned revision and FP32
LAION improved aesthetics V2 scoring. Diffusion noise has its own seed. Synthetic
uniform initialization uses the packaged full-corpus DiffusionDB bounds; an
external `bounds_file` or `bounds_source: "parti"` can select another input.
Initialization, rejected offspring and unfinished attempts count against the
budget; cached elites do not. Raw scores are retained; minimizing negates fitness.

`campaign.trials` must match the seed count. Entries have distinct labels and
optional overrides for `success_ratio`, `comparison_factor` and
`max_selection_pressure`. The manifest freezes each resolved recipe.
Reproducibility groups only matching recipes and seeds; paired variants are
comparisons, not identical repetitions.

`crossover_weights`/`mutation_weights` select named operator pools; omitted weights
retain equal-weight defaults. `operator_parameters` provides validated local
Gaussian/spherical settings. Frozen initial embeddings require an explicit
SHA-256; local initialization uses an anchor plus seeded bounded neighbors.

`render_batch_size` fixes physical renderer groups and records discarded padding;
`canonical_scoring` scores each image in the same FP32 shape. Exact parent reuse
requires both settings and defaults to false. `buffered_attempt_writes` is also
opt-in. These options change the numerical or persistence recipe and must be
recorded with results; see [artifact contracts](experiment_artifacts.md).

`initial_deadline_unix` takes precedence over `initial_budget_seconds`. A null
budget with no deadline disables the campaign clock; request timeouts remain.
By default finalization shares the campaign deadline. An explicit
`finalization_deadline_unix` may reserve a later finite completion window; with a
finite inference deadline it must be at least that deadline. Invalid clocks fail
before recipe resolution or session creation. An admission callback may raise
`SkipTrial(reason)` to record a declined trial without cancelling its siblings;
other callback/execution failures still fail the campaign. Skips remain in the
status and summary, and an entirely skipped campaign is not a passed campaign.
Storage admission
reserves raw output and its ZIP on the actual output filesystem, with the model
cache retaining its separate reserve. Runner count is an upper bound; multiple
prompt-embedding runners currently require a candidate batch limit of four.
Drive is disabled by default; verified deletion is a separate opt-in.
