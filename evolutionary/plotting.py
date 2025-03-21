from typing import List, Optional
import matplotlib.pyplot as plt

from evolutionary.evolution_base import Fitness


def plot_fitness_statistics(num_generations: int,
                            best_fitness: Optional[List[Fitness]] = None,
                            worst_fitness: Optional[List[Fitness]] = None,
                            avg_fitness: Optional[List[Fitness]] = None,
                            title: str = 'Fitness Statistics over Generations',
                            # Labels used when plotting multi-objective fitness
                            labels: Optional[List[str]] = None,
                            # Only plot the fitness of the index objective.
                            multi_objective_plot_index: Optional[int] = None
                            ):
    """
    Plots the fitness statistics over generations.
    Plots single of multi objective fitness with optional custom labels as descriptors for each fitness.
    When multi_objective_plot_index is specified, only plots the fitness of the index objective.
    """

    def fitness_from_index(fitness: List[Fitness], index: int) -> List[Fitness]:
        return [f[index] for f in fitness]

    generations = list(range(0, num_generations))
    use_custom_index = multi_objective_plot_index is not None

    if best_fitness is not None:
        plt.plot(generations,
                 fitness_from_index(best_fitness, multi_objective_plot_index) if use_custom_index else best_fitness,
                 label=[label + ' Best Fitness' for label in labels] if labels else 'Best Fitness')
    if worst_fitness is not None:
        plt.plot(generations,
                 fitness_from_index(worst_fitness, multi_objective_plot_index) if use_custom_index else worst_fitness,
                 label=[label + ' Worst Fitness' for label in labels] if labels else 'Worst Fitness')
    if avg_fitness is not None:
        plt.plot(generations,
                 fitness_from_index(avg_fitness, multi_objective_plot_index) if use_custom_index else avg_fitness,
                 label=[label + ' Average Fitness' for label in labels] if labels else 'Average Fitness')

    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title(title)
    plt.legend()
    plt.show()


def plot_time_statistics(evaluation_time: List[float],
                         creation_time: List[float],
                         post_evaluation_time: Optional[List[float]] = None,
                         title: str = 'Time Statistics over Generations in Seconds'):
    """
    Plots the summarized time statistics in a Pie chart.
    """
    labels = ['Evaluation', 'Creation'] + (['Post Evaluation'] if post_evaluation_time else [])
    sizes = [sum(evaluation_time), sum(creation_time)] + ([sum(post_evaluation_time)] if post_evaluation_time else [])

    def autopct_format(values):
        def autopct_with_values(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return f'{pct:.1f}%\n({val:d})'
        return autopct_with_values

    plt.pie(sizes, labels=labels, autopct=autopct_format(sizes), startangle=90)
    plt.axis('equal')
    plt.title(title)
    plt.show()
