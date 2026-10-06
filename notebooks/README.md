# Notebooks of Experiments
This folder contains templates and ready-to-use notebooks to execute a 
variety of evolutionary experiments with image-generation.

- [ga_osga_notebook.ipynb](ga_osga_notebook.ipynb): GA/OSGA with fixed operator pools,
  reproducible diffusion noise and complete evaluation/lineage JSON export.
  Configuration only; experiments have not been run.

The GA, OSGA, island-GA and NSGA templates install the working checkout and seed
Python/NumPy/Torch after model initialization, before constructing populations.
They recreate fixed diffusion noise for each candidate and retry. Both standalone
GA notebooks retain pickle saving and export scalar records, configuration, actual
costs, dependency versions, Git revision/diff and untracked implementation source
as JSON. Save notebooks before running and keep source unchanged through export;
use a committed feature revision for published experiments. The original GA keeps
its 200×100 settings (19,901 evaluator calls with one elite); the OSGA copy uses
100×100 and a 10,000-call cap. Island execution requires OSGA/caps disabled.
Analysis cells use completed population counts; final saved-image indices are one
less than those counts. These templates have been validated without experiments.
