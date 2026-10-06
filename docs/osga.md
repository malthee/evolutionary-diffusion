# Offspring selection and operator pools

`GeneticAlgorithm` supports fixed random variation pools, scalar evaluation records,
and optional offspring selection. Its creator, evaluator, selector, crossover and
mutator interfaces are unchanged. `notebooks/ga_osga_notebook.ipynb` is a cleared
copy of the existing GA notebook with an initial reproducible configuration. No
optimization experiments have been performed as part of this implementation.

```python
from evolutionary.algorithms.ga import GeneticAlgorithm, OffspringSelectionConfig
from evolutionary.operators import MultiCrossover, MultiMutator

# Pass these to the existing GA constructor along with creator/evaluator/selector.
crossover = MultiCrossover({"uniform": uniform_operator, "arithmetic": arithmetic_operator})
mutator = MultiMutator({"gaussian": gaussian_operator, "spherical": spherical_operator})
selection = OffspringSelectionConfig(success_ratio=0.6, comparison_factor=1.0,
                                    max_selection_pressure=10.0)
# GeneticAlgorithm(..., crossover=crossover, mutator=mutator,
#                  offspring_selection=selection, max_evaluations=10_000)
```

## Configuration choices

| Setting | Library default | Impact / notebook choice |
| --- | --- | --- |
| `offspring_selection` (keyword only) | `None` | Ordinary GA with the same tracing and budget support. Notebook enables OSGA. |
| `success_ratio` | `1.0` | Minimum fraction of the full offspring population that must succeed, before elitism. Notebook uses `0.6`; lower values admit more unsuccessful children. |
| `comparison_factor` | `1.0` | Parent threshold: `0` worse parent, `0.5` mean, `1` better parent. Notebook uses `1.0`. |
| `max_selection_pressure` | `10.0` | Retry cap: `floor(pressure * N)` offspring evaluations per generation. Notebook uses `10.0`. |
| `max_evaluations` (keyword only) | `None` | Whole-run evaluator-call cap, including initialization and rejected children. Notebook uses `10_000`. Cached elites cost nothing. Must cover the initial population. |
| Pool `weights` | Equal | Finite, nonnegative fixed weights in mapping order, with a positive finite sum. Crossover and mutation choices are independent. |
| Image creator `fixed_noise_seeds` (keyword only) | `None` | One seed per image. Explicit seeds recreate independent diffusion generators for every candidate and retry. Notebook uses `[0]`. |

Ratios and comparison factors must be finite in `[0, 1]`; pressure must be finite
and at least one. Population size, generation count, tournament selection,
elitism, crossover event probability and mutation event probability remain the
existing controls. **Mutation event probability** chooses whether to invoke the
mutator. The mutator's coordinate/row participation controls how much of the
embedding it changes after that event is chosen. These are separate probabilities.

OSGA requests a fresh evaluation for every newly constructed child, even if a
creator supplied precomputed fitness. Population evaluation and elite carryover
reuse cached fitness; ordinary GA retains its cache behavior.

`strict_osga=True` now maps to `OffspringSelectionConfig(1.0, 1.0, 10.0)`.
It requires strict improvement over the better parent and has bounded retries.
This intentionally changes the old worse-parent `>=` behavior. Combining this
legacy flag with explicit `offspring_selection` raises `ValueError`.

There are no threshold schedules, adaptive weights, probabilistic acceptance,
equality tolerances, selection before mutation, or configurable failure/survivor
policies. OSGA and evaluation limits currently require standalone GA execution:
`IslandModel.run()` rejects them before creating candidates because its existing
migration loop has no protocol for interrupted island generations. Ordinary
unbounded GA islands remain supported.

## Selection rules and fidelity

For maximization and two parents, the threshold is

```
min(parent_fitness) + comparison_factor * (max(parent_fitness) - min(parent_fitness))
```

A child succeeds only if its evaluated fitness is **strictly greater** than the
threshold, after crossover and optional mutation. With one parent the threshold
is that parent's fitness. Equality fails.

For a population of size `N`, construct at least `N` children and obtain at least
`ceil(success_ratio * N)` successes. Keep successes and uniformly sample remaining
slots from unsuccessful children without replacement. Ratio zero still constructs
`N` children; ratio one admits only successes. Then reinsert the previous
population's elites by replacing the worst selected children. This ordering means
successful children can be displaced and final survivors need not meet the quota.
The quota describes the constructed offspring population, before elitism.

Selection pressure is evaluated offspring attempts divided by `N`. If pressure or
the whole-run evaluation cap prevents a complete offspring population meeting the
quota, the run stops without relaxing the quota or committing a partial generation.
`termination_reason` is `max_selection_pressure`, `max_evaluations`, or
`num_generations`. On interruption, the returned best candidate includes all
candidates evaluated so far, including uncommitted children. On normal completion,
the best member of the final population is returned, preserving ordinary GA behavior.

Ordinary GA retains its elites-first construction with `N - E` children per
transition. It also stops without committing a partial population when its cap is
exhausted. For 100 populations including initialization, size 100 and one elite,
its cost is `100 + 99 * 99 = 9,901` evaluations. OSGA instead constructs `N` children
before elitism and pays for rejected attempts. The common notebook cap of 10,000
is a maximum; later comparisons must report actual costs and completed populations.

The mathematical implementation was derived independently from the original
scheme by Michael Affenzeller and Stefan Wagner:
[Offspring Selection: A New Self-Adaptive Selection Scheme for Genetic Algorithms](https://www.researchgate.net/publication/226905020_Offspring_Selection_A_New_Self-Adaptive_Selection_Scheme_for_Genetic_Algorithms).
As consistency references, HeuristicLab's
[parent comparator](https://github.com/heal-research/HeuristicLab/blob/main/HeuristicLab.Optimization.Operators/3.3/WeightedParentsQualityComparator.cs)
confirms the parent threshold and strict inequality, and its
[main operator](https://github.com/heal-research/HeuristicLab/blob/main/HeuristicLab.Algorithms.OffspringSelectionGeneticAlgorithm/3.3/OffspringSelectionGeneticAlgorithmMainOperator.cs)
places elitism after offspring selection. No HeuristicLab implementation was copied.
Stopping without committing an incomplete population is this framework's explicit
bounded-run policy; it is not a claim to reproduce every reference implementation's
termination/fallback policy.

## Embedding operators

All operators implement the existing single-child interfaces. Prompt variants
operate on `PromptEmbedData`; `Pooled*` variants return `PooledPromptEmbedData`.
One pooled event selects one family for both tensors. Point crossover cuts prompt
embeddings at token-row boundaries (`-2`) and pooled embeddings at coordinate
boundaries (`-1`); insufficient internal boundaries return a clone.

| Family | Prompt class / pooled class | Initial parameters |
| --- | --- | --- |
| Coordinate uniform | `UniformCrossover` / `PooledUniformCrossover` | Swap probability `0.5` |
| Token-row uniform | `RowUniformCrossover` / `PooledRowUniformCrossover` | Row swap probability `0.5`; pooled coordinates independently use `0.5` |
| One-point | `OnePointCrossover` / `PooledOnePointCrossover` | One internal boundary |
| Two-point | `TwoPointCrossover` / `PooledTwoPointCrossover` | Two distinct internal boundaries |
| Arithmetic | `ArithmeticCrossover` / `PooledArithmeticCrossover` | Weight `0.5`, full participation |
| SLERP | `SlerpCrossover` / `PooledSlerpCrossover` | Ratio `0.5`, optional bounds |
| BLX-α | `BLXAlphaCrossover` / `PooledBLXAlphaCrossover` | α `0.5`, required bounds |
| Bounded SBX | `SBXCrossover` / `PooledSBXCrossover` | Index `15`, coordinate participation `0.5`, required bounds |
| Gaussian | `UniformGaussianMutator` / `PooledUniformGaussianMutator` | Notebook retained: coordinate fraction `0.2`, strength `2.0` prompt / `0.4` pooled |
| Spherical | `SphericalRotationMutator` / `PooledSphericalRotationMutator` | Row participation `0.1`, angle at most `0.1` radians |
| Donor replacement | `DonorReplacementMutator` / `PooledDonorReplacementMutator` | Participation `0.1`, frozen clones of initial embeddings |
| Bounded polynomial | `PolynomialMutator` / `PooledPolynomialMutator` | Index `20`, coordinate participation `0.1`, required bounds |

Uniform coordinate crossover now follows its documented swap-rate direction:
zero returns the first parent's values and one the second parent's. The former
implementation reversed these endpoints. Equal probability remains unbiased.

SBX uses the classical bounded distribution and randomly returns one complete
child; pooled tensors share the child-branch choice. Polynomial mutation uses the
classical bounded coordinate distribution. Constant bounds and equal-parent
coordinates are handled without division by zero. Bounds may be scalars or
broadcastable tensors; invalid/nonfinite bounds are rejected. The notebook uses
its existing scalar embedding limits for repair. Repair rounds interval endpoints
inward to representable values in the final output dtype, so casting to half
precision cannot move a repaired value outside the interval. An interval with no
representable output value raises `ValueError`.

Distribution references: Deb's [SBX/BLX description](https://www.egr.msu.edu/~kdeb/resources.shtml),
and the bounded equations in the pinned minimum pymoo release's
[SBX](https://github.com/anyoptimization/pymoo/blob/0.6.1.6/pymoo/operators/crossover/sbx.py)
and [polynomial mutation](https://github.com/anyoptimization/pymoo/blob/0.6.1.6/pymoo/operators/mutation/pm.py)
operators. Numerical tests use closed-form quantiles and independently calculated
reference values at indices 15 and 20; the operators do not delegate to pymoo.

Spherical mutation accepts `[..., D]`, rotates normalized directions, and restores
original row norms **before** optional bounds repair. Zero rows and one-dimensional
vectors remain unchanged. Pooled participation refers to whole pooled vectors.
SLERP interpolates directions and radii; parallel/zero rows use linear interpolation.
Opposite directions follow a deterministic orthogonal great circle, or linear
interpolation in one dimension. Bounds repair may change norms. Sensitive tensor
calculations use float32, then restore the input shape, dtype and device. Parents
are never modified. Donor mutation uses the same row positions and one selected
initial donor for both prompt and pooled tensors; the bank never tracks survivors.

## Records, lineage and reproducibility

`SolutionHistoryItem` retains existing keys/parent links and adds optional
`crossover_name`, `mutation_name`, `evaluation_id`, and `creation_kind`
(`initial`, `offspring`, `elite`). Indices are assigned after survivor selection.
Every carried elite receives a fresh history entry pointing to its previous entry
and reuses its original evaluation ID. Operator names are `None` when skipped.
Named pool choices are recorded immediately; individual operators use class names.

`statistics.evaluation_records` contains scalar `EvaluationRecord` dataclasses for
initial evaluations and every evaluated child, without images/tensors. Evaluation
IDs start at one and are unique within an algorithm run; combine them with the
run identity/algorithm `ident` across runs. Records hold target generation,
parent history keys/fitness, operators, child fitness, comparison factor, threshold,
and success classification. Classification is `None` for initialization/ordinary
GA. Fitness, thresholds and classification use native Python floats/bools, including
when the evaluator returns NumPy scalars, so JSON export and summary counts agree
with the selection decision. `survivor_key` is assigned only to final survivors of
a committed population.
Rejected, elitism-displaced, and uncommitted attempts have `None`; displacement
does not rewrite their success classification.

`statistics.generation_summaries` holds attempts, successes, unsuccessful final
survivors, quota, pressure, cumulative evaluation count, completion/termination and
creation/evaluation seconds and the evaluation IDs of children displaced by
elitism. Generation zero describes initialization, with zero
quota/pressure. Fitness series and callbacks occur once per completed population.
Existing timing lists describe completed populations; interrupted-generation costs
are retained separately in its summary and raw records.

`statistics.operator_summary()` groups offspring records by target generation,
algorithm identifier and crossover/mutation pair. `attempts` counts records,
`successes` counts `successful=True`, and `survivors` counts final survivor keys.
`success_rate` is successes / classified attempts (all OSGA attempts, `None` for
ordinary GA); `survival_rate` is survivors / attempts. Initial candidates and elite
carryovers are excluded. Marginalizing these pairs by one operator describes
outcomes of combinations, not isolated causal effects. Generation and evaluation
indices permit later phase analysis without phase configuration.

The evolutionary notebooks load models before seeding Python, NumPy and Torch,
then build initial embeddings. This avoids changing the variation stream between
cold model initialization and cached setup in the same kernel.
Explicit noise seeds recreate separate diffusion generators on every candidate
and retry; omitted seeds retain the existing advancing-stream behavior. Fixed
seeds do not guarantee identical results across different devices, library
versions or nondeterministic kernels. The copied notebook retains image saving,
visualization, optional secondary analysis and pickle workflows, and exports JSON
configuration, operator parameters/weights, seeds, versions, checkout revision/diff,
untracked implementation source, actual costs, termination, all evaluation records,
lineage and existing statistics. Both standalone GA notebooks export this metadata;
the original retains its population 200 and ordinary operators, while the OSGA
copy uses population 100 and the full pool. Operator configuration dictionaries
also construct the operators, avoiding divergence between recorded and actual
parameters. Save notebooks before running, keep the source unchanged through
export, and commit the feature before published experiments. Only explicit run
cells start optimization. Island-GA and NSGA share checkout installation,
seeding-after-model-setup and fixed noise; their algorithm settings remain intact.

## Verification

Success criteria: strict threshold endpoints/equality; rounded quota over `N`;
uniform failure sampling; elitism after selection; bounded retries/evaluation
calls; complete-only callbacks/history; traceable rejected/displaced attempts;
seeded, bounded operators without parent mutation; identical mocked diffusion
noise across candidates/retries; numerical BLX, bounded SBX and polynomial
quantiles; final-dtype bounds; identical cold/warm variation state; cleared,
syntactically valid notebooks with constructible GA configuration and JSON export.

Run the unit and mocked integration suite with `python -m pytest tests` in a test
environment containing the project's dependencies plus pytest, nbformat and IPython.
Notebook tests construct configuration with mocked models and exercise JSON export;
they never execute the optimization or analysis cells. No production dependency
was added.
