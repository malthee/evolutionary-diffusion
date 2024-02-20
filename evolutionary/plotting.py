from typing import List, Optional, Any
import matplotlib.pyplot as plt

from evolutionary.evolution_base import Algorithm, SingleObjectiveFitness


def plot_fitness_statistics(algo: Algorithm[Any, Any, SingleObjectiveFitness],
                            include: List[str] = None,
                            custom_generations: Optional[int] = None):
    """
    Plots the fitness statistics of an evolutionary algorithm over generations.
    Limits the number of generations to custom_generations if specified.
    By default, includes best, worst and average fitness.
    Only supports single-objective optimization.
    """
    if include is None:
        include = ['best', 'worst', 'avg']
    generations = list(range(0, custom_generations)) if custom_generations else list(range(0, algo.num_generations))

    if 'best' in include:
        plt.plot(generations, algo.best_fitness, label='Best Fitness')
    if 'worst' in include:
        plt.plot(generations, algo.worst_fitness, label='Worst Fitness')
    if 'avg' in include:
        plt.plot(generations, algo.avg_fitness, label='Average Fitness')

    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Statistics over Generations')
    plt.legend()
    plt.show()
