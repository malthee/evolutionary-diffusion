# Changelog for the evolutionary package
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