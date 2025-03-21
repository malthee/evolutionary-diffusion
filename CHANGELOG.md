# Changelog for the evolutionary package
## 0.7.0 (16.03.2025)
* Added `TensorboardEmbedVisualizer` to allow for easy visualization of embeddings with different algorithms like PCA, UMAP, etc. TensorBoard comes with a lot of functionality for that.
* `load_torch_model` weights only is now `True`
* `plotting.py` now supports post_evaluation time tracking
* History of solutions (parents, mutations) now tracked in `statistics.py` with dataclasses in `history.py`
* Family tree visualization made possible with `graphviz`, updated image `processing.py` and added `family_tree.py`

## 0.6.0 (18.01.2024)
* Update `transformers`, `diffusers` and `datasets` to fix dependency issues with collab

### 0.4.3 (28.07.2024)
* Fix `numpy` version to `1.26.4` to avoid `2.0` incompatibility

### 0.4.2 (07.04.2024)
* added `completed_generations` to `Algorithm` to allow for better access to generations when 
run was not finished.

### 0.4.1 (01.04.2024)
* Strict OSGA can be enabled in GA
* ArithmeticCrossover now supports `proportion`, a parameter to control the amount of crossover
* Statistics time tracking is now handeled by Algorithm implementations. IslandModel uses
sum of all islands for time statistics.

## 0.4.0 (27.03.2024)
* Breaking changes to Crossover/Mutation
  * moved `tensor_variation` to model_helpers.
  * Mutation does not modify the original tensor anymore
  * changed parameter names (crossover_rate, to interpolation_weight to avoid confusion)
* GA, NSGA-II now use `crossover-` and `mutation-rate`
* Breaking changes to Fitness-Statistics (avg, best, worst), now own class in `statistics.py`
* Can now plot time spent on evaluation and creation of individuals

## 0.3.0 (25.03.2024)
* Added GoalDiminishingEvaluator, CappedEvaluator 
* Added 'ring' and 'random' topology to the IslandModel to allow control of migration
* Group-By-Ident visualization for image grid to allow visualization of best image per island
* Radar chart visualization for multi-objective optimization now supports max-value, so the charts remain the same size
* Fixed RankSelector division by zero error

### 0.2.6 (22.03.2024)
* NSGA-II can now use a binary tournament selection for the selection of parents

### 0.2.5 (20.03.2024)
* NSGA-II now returns result which has the best fitness sum with optional normalization

### 0.2.4 (08.03.2024)
* Add optional identifiers to image-saving do differentiate between islands in the island modal

### 0.2.3 (08.03.2024)
* Fixed plotting when only one variation (ex. avg_fitness) is plotted
* Added generation argument to Algorithm.perform_generation 
* NSGA now supports callback after NDS 

### 0.2.2 (06.03.2024)
* Fixed imports, structure to work with colab

## 0.2.0 (06.03.2024)
* Refactored package structure to single setup.py with extras, which allow 
a finer selection of dependencies and packages.
