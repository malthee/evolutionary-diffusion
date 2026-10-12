# Algorithm notebooks

Ready-to-run examples using the reusable implementations in `evolutionary/algorithms/`:

| Notebook | Algorithm |
|---|---|
| [ga_notebook.ipynb](ga_notebook.ipynb) | Genetic algorithm in prompt-embedding space |
| [ga_osga_notebook.ipynb](ga_osga_notebook.ipynb) | GA/OSGA with fixed operator pools and complete attempt/lineage export |
| [island_ga_notebook.ipynb](island_ga_notebook.ipynb) | Island GA with migration |
| [nsga_notebook.ipynb](nsga_notebook.ipynb) | U-NSGA-III, NSGA-II and NSGA-III |
| [sep_cma_notebook.ipynb](sep_cma_notebook.ipynb) | Separable CMA-ES; deterministic CPU example, no model downloads |

See [setup](../docs/setup.md), [GA/OSGA](../docs/osga.md),
[NSGA](../docs/nsga.md) and [Sep-CMA](../docs/sep_cma.md) for configuration.
Image notebooks preserve fixed diffusion noise, original embeddings and existing
image/JSON/pickle exports. Configure output paths before running.

Campaign recipes, detector comparisons and offline embedding analysis are under
[pre-testing](../pre-testing/README.md). Their execution switches remain disabled
by default. Actual campaign configuration and generated results stay outside the
public examples.
