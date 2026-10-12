# Separable CMA-ES

`evolutionary.algorithms.sep_cma` owns the reusable pycma adapter. It uses the
existing frozen pycma 4.4.4 dependency, available through `.[cma]` or
`uv sync --extra cma`. It does not import image, embedding, experiment or cloud
packages. GA/OSGA and NSGA remain in `evolutionary/algorithms/` too.

## Framework interface

[`sep_cma_notebook.ipynb`](../notebooks/sep_cma_notebook.ipynb) is a complete CPU
example. `SepCMAES` accepts `num_generations`, `population_size`, a solution
creator, a scalar maximization evaluator, `initial_mean`, `seed`, `sigma` and the
usual `post_evaluation_callback(generation, algorithm)` and `ident` arguments.
It exposes `population`, `statistics`, `completed_generations`, `evaluation_count`,
`best_solution()` and `run()` like the other framework algorithms.

Each generation evaluates a fresh population. Generation zero samples around
the mean; it does not evaluate an initial argument bank. The total fresh budget
is `num_generations * population_size`, and the returned solution is the best
across all evaluated populations. Ordered `create_solutions`/`evaluate_batch`
methods are used when supplied. Repeated `run()` calls reset adaptation and
statistics with the configured seed.

The geometry uses normalized `[0, 1]` coordinates with at least two variable
dimensions, population at least four and finite positive sigma. For domain
arguments, supply `decode(vector)`. SDXL coordinates live in
`evolutionary_prompt_embedding.coordinates.EmbeddingCoordinates`, which encodes
token/pooled bounds, removes fixed coordinates and repairs decoded values to
FP16. Experiment-specific initialization stays with the recipe.

## Ask/tell and checkpoints

`SepCMAOptimizer` exposes `ask()`, `tell(objectives)`, `state()` and
`restore_owned_state(bytes, sha256)`. This low-level interface **minimizes**;
`SepCMAES` negates framework fitness before passing it to `tell`.

The strategy uses diagonal covariance, no diagonal decoding or active covariance,
native `[0, 1]` bounds, no final-mean evaluation, and a private NumPy RandomState.
Its RNG is independent of global NumPy randomness and diffusion noise. Budgets
must contain whole populations. Repeated `ask()` returns the pending population
unchanged; invalid or incomplete scores do not advance it. The configured budget
controls termination, preserving the detector study's original ask/tell behavior.

The detector-search `SearchOptimizer` class retains its old import/pickle path
as a thin subclass. Existing owned checkpoints can resume without changing
future proposals. Pickle checkpoints are trusted local state: verify transfer
integrity and source/dependency identity before restoring them. Calibration,
images, query ledgers and held-out measurements remain in the detector recipe.
