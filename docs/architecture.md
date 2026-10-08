# Repository architecture

The algorithm libraries are independent of cloud providers. Experiments compose
those libraries; execution owns notebook sessions; persistence owns completed
archives. Azure only prepares metadata for an existing host.

| Location | Responsibility | Dependencies |
|---|---|---|
| `evolutionary/` | Algorithms, candidates, ordered scalar/batch contracts, statistics | No experiment or cloud imports |
| `evolutionary_imaging/`, `evolutionary_sound/`, `evolutionary_model_helpers/`, `evolutionary_prompt_embedding/` | Reusable creators, evaluators, variation and embedding archives | Algorithm contracts and optional domain dependencies |
| `evolutionary_extensions/experiments/prompt_embedding/` | Validated recipe, initialization, operator pool, CUDA runtime, scientific artifacts and notebook adapter | Domain libraries; persistence primitives |
| `evolutionary_extensions/execution/` | Managed notebook sessions, trial admission, bounded finalization and campaign records | Optional Jupyter dependencies; recipe adapter loaded on demand |
| `evolutionary_extensions/persistence/` | Immutable packaging and optional verified Drive transfers | Standard library; independent of Azure and inference |
| `evolutionary_extensions/azure/` | Local deployment configuration and kernel specification | Persistence primitives; no resource lifecycle ownership |
| `configs/examples/` | Portable, disabled GA/OSGA examples | No credentials, machine inventory or empirical host tuning |
| `environments/analysis/` | Locked offline-analysis environment | Separate from inference |
| `notebooks/` | Experiment and analysis entry points | Import package implementations |
| `docs/` | Public architecture, setup and behavior documentation | Linked from the root README |
| `.local/` (ignored) | Private profiles, host evidence, operational tools, handovers, generated results | Never included in release artifacts |

Within the recipe, `config.py` owns settings, `initialization.py` owns bounds and
initial populations, `operator_pool.py` owns variation, `runtime.py` owns model
loading/preflight and `experiment.py` owns optimization. `adapter.py` translates
between that recipe and notebook execution, including trial controls, output
estimates, validation and reporting. `finalization.py` runs bounded background
work without importing the model runtime. Importing execution does not change
CUDA environment variables or import Torch/Diffusers. GPU startup settings belong
to the selected kernel or a dedicated controller process.

Keep reusable examples in `configs/examples/`; keep actual campaigns under
`.local/configs/` or outside the checkout. Scientific inputs distributed with the
package must be compact, generic and have provenance/checksums. Do not add dated
run reports, subscription/resource identities, personal paths, one-off cleanup
scripts or generated notebooks to the public tree. Generated run folders also
remain ignored for older notebook workflows.

## Documentation map

- [Campaign configuration](configuration.md): portable setup and recipe controls.
- [Azure host preparation](azure.md): selected interpreter, paths and kernel.
- [Scientific artifacts](experiment_artifacts.md): precision, recording and persistence.
- [Offline analysis](embedding_analysis.md): archive format and analysis environment.
- [GA/OSGA](osga.md) and [NSGA](nsga.md): algorithm behavior and notebook conventions.
