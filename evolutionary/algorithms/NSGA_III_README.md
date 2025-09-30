# NSGA-III Algorithm

## Overview

NSGA-III (Non-dominated Sorting Genetic Algorithm III) is a many-objective optimization algorithm designed to handle problems with 4 or more objectives. It extends NSGA-II by replacing the crowding distance mechanism with a reference point-based approach.

## Key Features

- **Reference Point-Based Selection**: Uses predefined reference points on a normalized hyperplane to maintain diversity
- **Better Scalability**: Performs better than NSGA-II on problems with many objectives (4+)
- **Uniform Distribution**: Provides more uniform distribution of solutions across the Pareto front
- **Adaptive Niching**: Associates solutions with reference points to preserve diversity

## When to Use NSGA-III vs NSGA-II

### Use NSGA-III when:
- You have 4 or more objectives to optimize
- You need uniform distribution across the Pareto front
- You want explicit control over the solution distribution via reference points

### Use NSGA-II when:
- You have 2-3 objectives
- Crowding distance provides sufficient diversity
- Simpler implementation is preferred

## Usage Example

```python
from evolutionary.algorithms.nsga_iii import NSGA_III, NSGAIIITournamentSelector
from evolutionary_prompt_embedding.image_creation import SDXLPromptEmbeddingImageCreator
from evolutionary_imaging.evaluators import AestheticsImageEvaluator, CLIPScoreEvaluator, SingleCLIPIQAEvaluator
from evolutionary.evaluators import MultiObjectiveEvaluator

# Setup multi-objective evaluator
evaluator = MultiObjectiveEvaluator([
    AestheticsImageEvaluator(),                    # Objective 1
    CLIPScoreEvaluator(prompt="your prompt"),       # Objective 2
    SingleCLIPIQAEvaluator(prompts=["quality"]),    # Objective 3
    SingleCLIPIQAEvaluator(prompts=["brightness"])  # Objective 4
])

# Create NSGA-III instance
nsga3 = NSGA_III(
    num_generations=100,
    population_size=50,
    solution_creator=creator,
    selector=NSGAIIITournamentSelector(),
    mutator=mutator,
    crossover=crossover,
    evaluator=evaluator,
    initial_arguments=initial_embeddings,
    mutation_rate=0.2,
    crossover_rate=0.9,
    reference_point_divisions=6  # Controls number of reference points
)

# Run optimization
best = nsga3.run()

# Access results
print(f"Best fitness: {best.fitness}")
print(f"First front size: {len(nsga3.fronts[0])}")
print(f"Reference points: {nsga3.reference_points.shape[0]}")
```

## Parameters

### Required Parameters
- `num_generations`: Number of generations to evolve
- `population_size`: Size of the population
- `solution_creator`: Creates solutions from arguments
- `selector`: Selection operator (use NSGAIIITournamentSelector)
- `mutator`: Mutation operator
- `crossover`: Crossover operator
- `evaluator`: Multi-objective evaluator
- `initial_arguments`: Initial population arguments

### Optional Parameters
- `mutation_rate`: Probability of mutation (default: 0.1)
- `crossover_rate`: Probability of crossover (default: 0.9)
- `elitism_count`: Number of elite solutions to preserve (default: None)
- `reference_point_divisions`: Number of divisions for reference points (auto-calculated if None)
- `post_evaluation_callback`: Callback after each evaluation
- `post_non_dominated_sort_callback`: Callback after non-dominated sorting
- `ident`: Identifier for tracking (useful for island models)

## Reference Point Generation

NSGA-III uses the Das-Dennis method to generate uniformly distributed reference points. The number of reference points is determined by:

```
Number of points = C(M + p - 1, p)
```

where:
- M = number of objectives
- p = number of divisions

Example reference point counts:
- 3 objectives, 6 divisions → 28 reference points
- 4 objectives, 5 divisions → 56 reference points
- 5 objectives, 4 divisions → 70 reference points

You can control this via the `reference_point_divisions` parameter, or let it be auto-calculated based on the number of objectives.

## Algorithm Flow

1. **Initialization**: Generate initial population and reference points
2. **Evaluation**: Evaluate fitness of all solutions
3. **Non-dominated Sorting**: Create Pareto fronts
4. **Normalization**: Normalize objective values
5. **Association**: Associate solutions with reference points
6. **Niching**: Select solutions based on niche preservation
7. **Reproduction**: Create offspring via crossover and mutation
8. **Repeat** steps 2-7 for specified generations

## Implementation Details

The implementation includes:
- `NSGAIIISolutionCandidate`: Solution candidate with reference point attributes
- `NSGAIIITournamentSelector`: Tournament selection based on rank and reference point distance
- `_generate_reference_points`: Das-Dennis reference point generation
- `_normalize_objectives`: Objective normalization
- `_associate_to_reference_points`: Association mechanism
- `_niching_selection`: Niching-based selection

## Comparison with NSGA-II

| Feature | NSGA-II | NSGA-III |
|---------|---------|----------|
| Diversity Mechanism | Crowding Distance | Reference Points |
| Best for Objectives | 2-3 | 4+ |
| Distribution | May be uneven | More uniform |
| Computational Cost | Lower | Slightly higher |
| Parameter Control | Less explicit | Reference point divisions |

## See Also

- [NSGA-II Implementation](nsga_ii.py)
- [Example Script](../examples/nsga_iii_example.py)
- [Original Paper: Deb & Jain (2014)](https://doi.org/10.1109/TEVC.2013.2281535)

## Notes

- The algorithm automatically generates reference points based on the number of objectives
- Population size should ideally match or be a multiple of the number of reference points
- For image generation, use with 4+ evaluation metrics for best results
- Compatible with all existing evolutionary operators (crossover, mutation, etc.)
