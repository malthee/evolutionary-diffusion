# Notebooks of Experiments
This folder contains templates and ready-to-use notebooks to execute a 
variety of evolutionary experiments with image-generation.

- [prompt_embedding_experiment.ipynb](prompt_embedding_experiment.ipynb): reusable
  DiffusionDB-initialized GA/OSGA with complete scientific outputs and optional Drive.
  [OSGA](../configs/examples/osga.json) and [GA](../configs/examples/ga.json) profiles use the
  one-step/population-64/6,400-evaluation recipe. Execution is disabled by default.
  See [campaign configuration](../docs/configuration.md) and
  [Azure host preparation](../docs/azure.md).

- [ga_osga_notebook.ipynb](ga_osga_notebook.ipynb): GA/OSGA with fixed operator pools,
  reproducible diffusion noise and complete evaluation/lineage JSON export.
  Configuration only; experiments have not been run.

GA, OSGA, island-GA and NSGA use the working checkout, seed after model setup and
recreate fixed diffusion noise. Both standalone GA notebooks save JSON and pickle;
analysis uses completed generations. See [configuration and reproducibility](../docs/osga.md)
for operator parameters, evaluation costs, export fields and island restrictions.

The NSGA notebook defaults to U-NSGA-III, with NSGA-II/III comparisons and JSON/pickle
export. See [configuration and algorithm behavior](../docs/nsga.md) for reference
directions and implementation limits.

## Embedding persistence and offline visualization

GA, OSGA, NSGA and island GA persist original embeddings incrementally in a
viewer-independent Safetensors archive. `save_embeddings` and `save_images` are
independent switches; each experiment has a unique run directory. Analyze it
locally using [embedding_analysis.ipynb](embedding_analysis.ipynb), which handles
projection, Embedding Atlas 2D exploration, 3D image markers and optional images.
See [the archive and analysis guide](../docs/embedding_analysis.md) for the format,
separate locked environment, storage estimates and validation commands.
