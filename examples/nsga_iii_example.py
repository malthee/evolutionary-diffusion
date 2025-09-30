"""
Example: Using NSGA-III for Multi-Objective Optimization

This example demonstrates how to use NSGA-III with the evolutionary-diffusion library.
NSGA-III is particularly well-suited for many-objective optimization problems (3+ objectives).

Key differences from NSGA-II:
- Uses reference points instead of crowding distance for diversity preservation
- Better suited for problems with many objectives (4+)
- Provides more uniform distribution of solutions across the Pareto front

This example can be adapted to work with image generation by replacing the simple
evaluator with MultiObjectiveEvaluator combining multiple image quality metrics.
"""

# Basic imports
from typing import List
from evolutionary.evolution_base import SolutionCreator, Mutator, Crossover, SolutionCandidate, Evaluator
from evolutionary.evaluators import MultiObjectiveEvaluator
from evolutionary.algorithms.nsga_iii import NSGA_III, NSGAIIITournamentSelector


# Example with a simple test problem (replace with your actual solution types)
class ExampleSolutionCreator(SolutionCreator[List[float], List[float]]):
    """
    Simple solution creator for demonstration.
    In practice, you would use something like SDXLPromptEmbeddingImageCreator
    """
    def create_solution(self, argument: List[float]) -> SolutionCandidate[List[float], List[float], any]:
        return SolutionCandidate(argument, argument)


class ExampleMutator(Mutator[List[float]]):
    """Simple Gaussian mutation for demonstration."""
    def __init__(self, mutation_strength: float = 0.1):
        self.mutation_strength = mutation_strength
    
    def mutate(self, argument: List[float]) -> List[float]:
        import random
        return [x + random.gauss(0, self.mutation_strength) for x in argument]


class ExampleCrossover(Crossover[List[float]]):
    """Simple arithmetic crossover for demonstration."""
    def crossover(self, argument1: List[float], argument2: List[float]) -> List[float]:
        import random
        alpha = random.random()
        return [alpha * x1 + (1 - alpha) * x2 for x1, x2 in zip(argument1, argument2)]


# Individual objective evaluators
class Objective1Evaluator(Evaluator[List[float], float]):
    """Example objective 1: distance from [1, 0, 0, ...]"""
    def evaluate(self, result: List[float]) -> float:
        return -sum((result[i] - (1 if i == 0 else 0)) ** 2 for i in range(len(result)))


class Objective2Evaluator(Evaluator[List[float], float]):
    """Example objective 2: distance from [0, 1, 0, ...]"""
    def evaluate(self, result: List[float]) -> float:
        return -sum((result[i] - (1 if i == 1 else 0)) ** 2 for i in range(len(result)))


class Objective3Evaluator(Evaluator[List[float], float]):
    """Example objective 3: distance from [0, 0, 1, ...]"""
    def evaluate(self, result: List[float]) -> float:
        return -sum((result[i] - (1 if i == 2 else 0)) ** 2 for i in range(len(result)))


class Objective4Evaluator(Evaluator[List[float], float]):
    """Example objective 4: distance from [0.5, 0.5, 0.5, ...]"""
    def evaluate(self, result: List[float]) -> float:
        return -sum((result[i] - 0.5) ** 2 for i in range(len(result)))


def run_nsga_iii_example():
    """
    Example demonstrating NSGA-III usage.
    
    For actual image generation, replace the example components with:
    - creator: SDXLPromptEmbeddingImageCreator
    - evaluator: MultiObjectiveEvaluator with image quality metrics
    - mutator: PooledUniformGaussianMutator
    - crossover: PooledArithmeticCrossover
    - initial_args: PooledPromptEmbedData instances
    """
    import random
    
    # Configuration
    num_dimensions = 6
    population_size = 40  # Adjust based on number of objectives and reference points
    num_generations = 50
    num_objectives = 4  # NSGA-III excels at 4+ objectives
    
    print(f"NSGA-III Multi-Objective Optimization Example")
    print(f"  Population Size: {population_size}")
    print(f"  Generations: {num_generations}")
    print(f"  Objectives: {num_objectives}")
    print(f"  Dimensions: {num_dimensions}\n")
    
    # Create initial population
    initial_args = [[random.random() for _ in range(num_dimensions)] 
                    for _ in range(population_size)]
    
    # Create algorithm components
    creator = ExampleSolutionCreator()
    # Create multi-objective evaluator by combining individual objectives
    evaluator = MultiObjectiveEvaluator([
        Objective1Evaluator(),
        Objective2Evaluator(),
        Objective3Evaluator(),
        Objective4Evaluator()
    ])
    selector = NSGAIIITournamentSelector()
    mutator = ExampleMutator(mutation_strength=0.1)
    crossover = ExampleCrossover()
    
    # Create NSGA-III instance
    nsga3 = NSGA_III(
        num_generations=num_generations,
        population_size=population_size,
        solution_creator=creator,
        selector=selector,
        mutator=mutator,
        crossover=crossover,
        evaluator=evaluator,
        initial_arguments=initial_args,
        mutation_rate=0.2,
        crossover_rate=0.9,
        elitism_count=2,
        reference_point_divisions=5,  # Controls number of reference points
        # Callback after non-dominated sorting (optional)
        post_non_dominated_sort_callback=lambda gen, alg: 
            print(f"Generation {gen}: {len(alg.fronts[0])} solutions in first front")
            if gen % 10 == 0 else None
    )
    
    print("Running NSGA-III...")
    print(f"Reference points generated: {nsga3.reference_points.shape[0] if nsga3.reference_points is not None else 'Will be generated'}\n")
    
    # Run the algorithm
    best = nsga3.run()
    
    # Display results
    print(f"\n{'='*60}")
    print("Results:")
    print(f"{'='*60}")
    print(f"Best solution fitness: {[f'{f:.4f}' for f in best.fitness]}")
    print(f"Sum of fitness: {sum(best.fitness):.4f}")
    print(f"\nNumber of Pareto fronts: {len([f for f in nsga3.fronts if f])}")
    print(f"First front size: {len(nsga3.fronts[0])}")
    print(f"Reference points used: {nsga3.reference_points.shape[0]}")
    
    # Show diversity of solutions in first front
    print(f"\nDiversity in first Pareto front (showing first 5):")
    for i, sol in enumerate(nsga3.fronts[0][:5]):
        print(f"  Solution {i}: {[f'{f:.4f}' for f in sol.fitness]}")
    
    # Statistics
    print(f"\nStatistics:")
    print(f"  Total generations completed: {nsga3.completed_generations}")
    
    return nsga3


def usage_with_imaging():
    """
    Example of how to adapt this for actual image generation.
    
    NOTE: This is pseudo-code to show the structure - requires actual setup
    """
    print("\nFor image generation with NSGA-III, use:")
    print("""
    from evolutionary.algorithms.nsga_iii import NSGA_III, NSGAIIITournamentSelector
    from evolutionary_prompt_embedding.image_creation import SDXLPromptEmbeddingImageCreator
    from evolutionary_prompt_embedding.variation import PooledArithmeticCrossover, PooledUniformGaussianMutator
    from evolutionary_imaging.evaluators import AestheticsImageEvaluator, CLIPScoreEvaluator
    from evolutionary.evaluators import MultiObjectiveEvaluator
    
    # Setup evaluator with multiple objectives
    evaluator = MultiObjectiveEvaluator([
        AestheticsImageEvaluator(),                    # Objective 1: Aesthetics
        CLIPScoreEvaluator(prompt="your prompt"),       # Objective 2: Prompt matching
        SingleCLIPIQAEvaluator(prompts=["quality"]),    # Objective 3: Quality
        SingleCLIPIQAEvaluator(prompts=["brightness"])  # Objective 4: Brightness
    ])
    
    # Create NSGA-III with imaging components
    nsga3 = NSGA_III(
        num_generations=100,
        population_size=50,
        solution_creator=SDXLPromptEmbeddingImageCreator(...),
        selector=NSGAIIITournamentSelector(),
        mutator=PooledUniformGaussianMutator(...),
        crossover=PooledArithmeticCrossover(...),
        evaluator=evaluator,
        initial_arguments=initial_embeddings,
        reference_point_divisions=6  # Adjust based on objectives
    )
    
    best = nsga3.run()
    """)


if __name__ == "__main__":
    # Run the example
    result = run_nsga_iii_example()
    
    # Show how to adapt for imaging
    usage_with_imaging()
    
    print("\n" + "="*60)
    print("Example completed successfully!")
    print("="*60)
