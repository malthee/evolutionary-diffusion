# Evolutionary-Diffusion Developer Guidelines

## Project Overview
Evolutionary-Diffusion combines evolutionary computing with diffusion models for generating and optimizing images and audio. The framework separates solution representation into arguments (A) and results (R), where evolutionary operations (mutation, crossover) happen on arguments, while fitness evaluation happens on results.

## Project Structure

```
evolutionary-diffusion/
├── evolutionary/              # Core evolutionary framework
│   ├── algorithms/            # Evolutionary algorithm implementations (GA, Island Model, NSGA-II)
│   ├── evolution_base.py      # Core abstractions and interfaces
│   ├── evaluators.py          # Base evaluator implementations
│   └── ...
├── evolutionary_imaging/      # Image-specific extensions
│   ├── evaluators.py          # Image evaluators (Aesthetics, CLIP, etc.)
│   └── ...
├── evolutionary_model_helpers/ # Model loading and utility functions
├── evolutionary_prompt_embedding/ # Prompt embedding representations
│   ├── image_creation.py      # Image creation from embeddings
│   ├── sound_creation.py      # Sound creation from embeddings
│   └── variation.py           # Mutation and crossover for embeddings
├── evolutionary_sound/        # Sound-specific extensions
└── notebooks/                 # Example notebooks
```

## Object Hierarchy

1. **Core Components**:
   - `SolutionCandidate<A, R, Fitness>`: Holds arguments, results, and fitness
   - `SolutionCreator<A, R>`: Creates solutions from arguments
   - `Evaluator<R, Fitness>`: Evaluates results to produce fitness
   - `Mutator<A>`: Mutates arguments
   - `Crossover<A>`: Combines two arguments
   - `Selector<Fitness>`: Selects candidates based on fitness

2. **Algorithms**:
   - `GeneticAlgorithm`: Standard genetic algorithm
   - `IslandModel`: Multiple populations with migration
   - `NSGA_II`: Multi-objective optimization

3. **Domain-Specific Components**:
   - Image creators (SDXLPromptEmbeddingImageCreator, etc.)
   - Image evaluators (AestheticsImageEvaluator, CLIPScoreEvaluator, etc.)
   - Sound creators (AudioLDMSoundCreator)
   - Sound evaluators (AudioboxAestheticsEvaluator)

## Adding New Components

### Adding a New Evaluator
1. Subclass `SingleObjectiveEvaluator` or create a multi-objective evaluator
2. Implement the `evaluate` method
3. Place in the appropriate domain-specific package

```python
class MyImageEvaluator(SingleObjectiveEvaluator[Image]):
    def evaluate(self, result: Image) -> float:
        # Implement evaluation logic
        return score
```

### Adding a New Creator
1. Subclass `SolutionCreator`
2. Implement the `create_solution` method
3. Place in the appropriate domain-specific package

```python
class MyImageCreator(SolutionCreator[EmbeddingArguments, Image]):
    def create_solution(self, argument: EmbeddingArguments) -> SolutionCandidate[EmbeddingArguments, Image, Any]:
        # Implement creation logic
        return SolutionCandidate(argument, result)
```

### Adding a New Algorithm
1. Subclass `EvolutionaryAlgorithm` from algorithm_base.py
2. Implement required methods
3. Place in the algorithms directory

## Best Practices

1. **Type Annotations**: Always use proper type annotations with generics to ensure type safety.

2. **Separation of Concerns**: 
   - Keep argument representation separate from result representation
   - Keep domain-specific code in the appropriate packages

3. **Extending the Framework**:
   - Create new evaluators for different fitness criteria
   - Create new creators for different generation methods
   - Reuse existing components when possible

4. **Testing**:
   - Use notebooks for experimentation
   - Validate new components with existing algorithms
   - Compare results with baseline implementations

5. **Performance Considerations**:
   - Batch operations when possible
   - Consider evaluation time in fitness functions
   - Use appropriate hardware acceleration (CUDA, MPS)