# Repository guidance

Read the applicable instructions, relevant code, tests and documentation before
editing. Inspect `git status --short`; preserve unrelated changes and the user's
staging choices. Make the smallest coherent change and review the final diff.
Use normal descriptive branch names without a `codex/` prefix, and avoid that
word in test inputs. If a `.codegraph/` index exists, use CodeGraph before source
searches; do not create an index as part of a code change.

## Ownership and structure

- `evolutionary/algorithms/` owns reusable algorithms and third-party optimizer
  adapters. Keep experiment, cloud and domain imports out of the core. Fitness
  is maximized; adapt minimization libraries at their boundary.
- `evolutionary_imaging/`, `evolutionary_sound/`, `evolutionary_model_helpers/` and
  `evolutionary_prompt_embedding/` own domain creators, evaluators and variation.
  Arguments evolve; results are evaluated. Preserve scalar/batch ordering.
- `evolutionary_extensions/experiments/` composes those libraries into research
  recipes. Execution owns notebook sessions; persistence owns archives; Azure
  helpers prepare host configuration. Resource lifecycle remains externally owned.
- `notebooks/` contains ready-to-run algorithm examples. Put detector evaluation,
  embedding analysis and campaign notebooks in the corresponding `pre-testing/`
  subdirectory. Keep notebooks thin and retain existing outputs when moving them.
- `configs/examples/` and `docs/` contain portable configuration and public
  behavior. Private host paths, credentials, campaign controls, models and results
  belong in ignored local directories. Never modify historical evidence as cleanup.

## Implementation and verification

Prefer existing dependencies and patterns. Ask before adding a new production
dependency. `pyproject.toml` is the package metadata source; `setup.py` is only a
compatibility entry point. Preserve optional dependency boundaries and vendor
environment pins. See [setup](docs/setup.md) and [architecture](docs/architecture.md).

Preserve scientific invariants: seeds and private RNGs, fixed diffusion noise,
coordinate bounds/FP16 repair, fresh evaluation budgets, score polarity, frozen
calibration and disjoint held-out measurements. Checkpoint compatibility and
completed-score reuse must survive refactors. Do not run GPU experiments, upload
results or start/stop cloud resources merely to test a cleanup.

Add deterministic CPU tests for changed behavior or meaningful regression risks.
Run focused tests first, then the relevant suite. Use Ruff on the affected Python
files and validate notebook structure and Python cells. For packaging changes,
build wheel/sdist and inspect assets; check the uv lock without upgrading pins.
Report commands, failures and unavailable validation accurately. Comments explain
intent and invariants; avoid speculative abstractions and unrelated formatting.
