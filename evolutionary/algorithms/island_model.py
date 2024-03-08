import random
from typing import List
from tqdm import tqdm
from evolutionary.evolution_base import Algorithm, A, R, Fitness, SolutionCandidate


class IslandModel:
    def __init__(self, algorithms: List[Algorithm[A, R, Fitness]], migration_interval: int, migration_size: int):
        self._islands = algorithms
        self._migration_interval = migration_interval
        self._migration_size = migration_size
        assert migration_interval > 0, "Migration interval must be greater than 0"
        assert migration_size > 0, "Migration size must be greater than 0"
        assert len(self._islands) > 1, "Island model requires at least 2 islands"
        assert all(algorithms[0].num_generations == algorithm.num_generations for algorithm in algorithms), \
            "All islands must have the same number of generations"
        """
        Parameters
        ----------
        algorithms : List[Algorithm[A, R, Fitness]]
            The algorithms to run on each island. Typification must be the same.
        migration_interval : int
            How many generations to wait before migrating individuals between islands.
        migration_size : int
            How many individuals to migrate in total in each migration event.
        """

    def _migrate(self):
        """
        Perform random migration of individuals between islands.
        After this some islands may have fewer individuals than others.
        This could be enhanced with replacement strategies, like taking the best ones from the source island etc.
        """
        for _ in range(self._migration_size):
            source_island, destination_island = random.sample(self._islands, 2)

            migrant = random.choice(source_island.population)
            source_island.population.remove(migrant)  # Remove migrant from source island

            # If the destination island has reached population_size, replace a random individual
            if len(destination_island.population) >= destination_island.population_size:
                replace_index = random.randrange(destination_island.population_size)
                destination_island.population[replace_index] = migrant
            else:
                # Otherwise, simply add the migrant to the destination island
                destination_island.population.append(migrant)

    def run(self) -> List[SolutionCandidate[A, R, Fitness]]:
        """
        Executes the generations of the islands and migration between them.
        Returns the best solutions from each island.
        """
        for island in self._islands:
            island.create_initial_population()

        generations = self._islands[0].num_generations
        for generation in tqdm(range(generations), unit='generation'):
            for island in self._islands:
                island.evaluate_population(generation)

            # If this is the last generation, finish here
            if generation == generations - 1:
                continue

            for island in self._islands:
                island.perform_generation(generation)

            if generation % self._migration_interval == 0:
                self._migrate()

        return [island.best_solution() for island in self._islands]

    def _island_fitness_aggregated(self, island_fitness_list: List[List[Fitness]]):
        num_generations = len(island_fitness_list[0])

        if isinstance(island_fitness_list[0][0], list):  # Is multi-objective when first islands fitness values are
            num_objectives = len(island_fitness_list[0][0])
            avg_over_time = []

            for gen in range(num_generations):
                avg_per_objective = [0] * num_objectives
                for o in range(num_objectives):
                    avg_per_objective[o] = sum(island[gen][o] for island in island_fitness_list) / len(self._islands)
                avg_over_time.append(avg_per_objective)
            return avg_over_time

        else:  # Single-objective
            return [sum(island[gen] for island in island_fitness_list) / len(self._islands) for gen in
                    range(num_generations)]

    @property
    def avg_fitness(self):
        """Average fitness over all islands"""
        avg_fitness = [island.avg_fitness for island in self._islands]
        return self._island_fitness_aggregated(avg_fitness)

    @property
    def worst_fitness(self):
        """Average worst fitness over all islands"""
        worst_fitness = [island.worst_fitness for island in self._islands]
        return self._island_fitness_aggregated(worst_fitness)

    @property
    def best_fitness(self):
        """Average best fitness over all islands"""
        best_fitness = [island.best_fitness for island in self._islands]
        return self._island_fitness_aggregated(best_fitness)
