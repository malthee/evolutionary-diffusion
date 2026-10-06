# Multi-objective optimization

Use `U_NSGA_III` as the primary method. NSGA-II remains a useful baseline for two
or three objectives; neither method has demonstrated superiority on this project's
embeddings. The number of fitness objectives, rather than embedding dimensions,
determines whether a problem is many-objective.

## Reference directions

Keep **fixed Riesz energy directions**, generated once with a separate seed and
saved with the run. Energy generation permits chosen point counts. Das–Dennis
restricts `H = binomial(M + p - 1, p)` (M objectives, p divisions), and coarse
high-dimensional lattices put all points on simplex boundaries. See
[Blank et al. (2021)](https://www.egr.msu.edu/~kdeb/papers/c2020002.pdf).

- `N` controls population size and evaluation cost; `H` controls reference coverage.
  Configure them independently, with `H <= N` for default U-NSGA-III mating.
- Keep `H = 20`, `N = 20` for the existing ten-objective notebook. This is an initial
  budget-conscious setup, not evidence of adequate coverage of the full front.
- For two or three objectives, fewer directions can allow more within-niche
  competition. For example, `H = 10`, `N = 20` is a reasonable starting comparison,
  not a tuned optimum. Das–Dennis is also suitable when its lattice count fits.
- Keep directions and their seed unchanged when comparing population sizes.
  Energy directions do not adapt to the Pareto front; changing objective definitions
  requires a new direction set of matching dimension.

The library's `ref_dirs_method="energy"` convenience generates `H = N`. To choose
`H` independently, supply `ref_dirs`, as the notebook does. Pass energy seeds to
`RieszEnergyReferenceDirectionFactory(...).do(seed=...)`. Seed Python and Torch
for mating/variation separately; diffusion noise uses fresh fixed-seed generators.

## Algorithm behavior and compatibility

All variants maximize finite objective vectors of consistent length, evaluate `N`
children, and select `N` survivors from the parent/child union. Cached fitness costs
no evaluator calls. `evaluation_count` reports actual calls; uncached cost is `N * G`
because `num_generations` includes initialization.

- [NSGA-II (2002), §III](https://doi.org/10.1109/4235.996017): rank/crowding mating and
  front-wise crowding survival. Normalized crowding is now the default; `False`
  retains the scale-sensitive option. Constant objectives contribute zero. Extra
  `elitism_count` is rejected because union survival already supplies elitism.
- [NSGA-III (2014), §IV](https://www.egr.msu.edu/~kdeb/papers/k2012009.pdf): random mating
  and least-occupied reference-niche survival. Empty niches choose the closest
  candidate; occupied niches choose randomly. Existing pymoo supplies persistent,
  guarded [hyperplane normalization](https://www.egr.msu.edu/~kdeb/papers/c2018009.pdf),
  reset between runs. Fitness is negated only at pymoo's minimization boundary.
- [U-NSGA-III (2016), author report §3](https://www.egr.msu.edu/~kdeb/papers/c2014022.pdf):
  adjacent tournaments plus a shuffled pass construct the mating pool. Within a
  niche, prefer lower rank, then smaller perpendicular distance; exact ties choose
  the second contestant. Across niches, choose randomly. Default population size
  must be divisible by four. Each pair produces two sibling events through the
  existing single-child embedding operators, rather than complementary SBX children.

Custom selectors/operators are extensions. `NSGAIIIBinaryRankSelector` is a custom
rank tournament, not unified mating; equal ranks now tie randomly instead of using
raw objective sums. Constraints, adaptive directions, aspiration regions and island
migration are outside the verified scope.

`pareto_front` is the result. `best_solution()` returns an equal-weight, min/max-scaled
representative from that front; this decision rule never affects selection. Callbacks
and fitness statistics occur once per completed population, with current fronts and
survivor lineage. Parent carryover receives an `elite` entry linked to its previous key.

The [notebook](../notebooks/nsga_notebook.ipynb) retains NSGA-II/III comparisons,
JSON configuration/seeds/directions, costs, Pareto fitness, lineage and statistics,
plus pickle. Later comparisons should share objectives, initial embeddings, operators,
noise and evaluation allowance, and report repeated seeds and Pareto-set quality.
Tests use fixed vectors and mocked models; no optimization experiments were run.
