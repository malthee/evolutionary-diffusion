# Notebooks of Experiments
This folder contains templates and ready-to-use notebooks to execute a 
variety of evolutionary experiments with image-generation.

- [ga_osga_notebook.ipynb](ga_osga_notebook.ipynb): GA/OSGA with fixed operator pools,
  reproducible diffusion noise and complete evaluation/lineage JSON export.
  Configuration only; experiments have not been run.

GA, OSGA, island-GA and NSGA use the working checkout, seed after model setup and
recreate fixed diffusion noise. Both standalone GA notebooks save JSON and pickle;
analysis uses completed generations. See [configuration and reproducibility](../docs/osga.md)
for operator parameters, evaluation costs, export fields and island restrictions.
