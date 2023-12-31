from typing import List, Optional
import matplotlib.pyplot as plt

from evolutionary.evolution_base import Algorithm


def plot_fitness_statistics(algo: Algorithm, include: List[str] = None, custom_generations: Optional[int] = None):
    if include is None:
        include = ['best', 'worst', 'avg']
    generations = list(range(0, custom_generations)) if custom_generations else list(range(0, algo.num_generations + 1))
    # +1 as we account for the last generation

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
