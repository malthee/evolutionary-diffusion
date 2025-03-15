import random
from itertools import chain
from typing import List, Literal, Generic, Dict
from tqdm import tqdm
from dataclasses import replace
from evolutionary.evolution_base import A, R, Fitness, SolutionCandidate
from evolutionary.algorithms.algorithm_base import Algorithm
from evolutionary.history import SolutionSourceMeta, SOLUTION_SOURCE_META_KEY
from evolutionary.statistics import StatisticsTracker, TimeList, SolutionHistoryKey, SolutionHistoryItem


class IslandStatisticsTracker(StatisticsTracker):
    """
    Custom StatisticsTracker that overwrites the time properties to return the sum of all islands.
    """

    def __init__(self, islands: List[Algorithm[A, R, Fitness]]):
        super().__init__()
        self._islands = islands
        self._solution_history_cache = {} # Avoid recalculating history of all islands every time its accessed

    @property
    def evaluation_time(self) -> TimeList:
        return [sum(island.statistics.evaluation_time) for island in self._islands]

    @property
    def creation_time(self) -> TimeList:
        return [sum(island.statistics.creation_time) for island in self._islands]

    @property
    def post_evaluation_time(self) -> TimeList:
        return [sum(island.statistics.post_evaluation_time) for island in self._islands]

    @property
    def solution_history(self) -> Dict[SolutionHistoryKey, SolutionHistoryItem]:
        """Combines the solution history of all islands. Cached for performance."""
        if self._solution_history_cache:
            return self._solution_history_cache

        combined_history = {}
        for island in self._islands:
            combined_history |= island.statistics.solution_history
        return combined_history

    def reset_solution_history_cache(self):
        self._solution_history_cache = {}


class IslandModel(Generic[A, R, Fitness]):
    def __init__(self, algorithms: List[Algorithm[A, R, Fitness]], migration_interval: int, migration_size: int,
                 topology: Literal['ring', 'random'] = 'ring'):
        self._islands = algorithms
        self._migration_interval = migration_interval
        self._migration_size = migration_size
        self._topology = topology
        self._statistics = IslandStatisticsTracker(self._islands)
        self._completed_generations = 0
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
        topology : Literal['ring', 'random']
            The migration topology of the islands. 'ring' means that each island is connected to its neighbors in a ring.
            'random' means that each island can migrate to any other island.
        """

    def _migrate(self):
        """
        Perform migration of individuals between islands.
        After this some islands may have fewer individuals than others.
        This could be enhanced with replacement strategies, like taking the best ones from the source island etc.
        """
        for _ in range(self._migration_size):
            if self._topology == 'ring':
                source_index = random.randrange(len(self._islands))
                source_island = self._islands[source_index]
                destination_index = (source_index + 1) % len(self._islands)
                destination_island = self._islands[destination_index]
            elif self._topology == 'random':
                source_island, destination_island = random.sample(self._islands, 2)
            else:
                raise ValueError(f"Invalid topology: {self._topology}")

            migrant = random.choice(source_island.population)
            migrant_history_key = SolutionHistoryKey(source_island.population.index(migrant), source_island.completed_generations, source_island.ident)
            migrant_history = source_island.statistics.solution_history.get(migrant_history_key)

            source_island.population.remove(migrant) # Remove migrant from source island
            if migrant_history is not None:
                migrant.meta[SOLUTION_SOURCE_META_KEY] = SolutionSourceMeta(index=migrant_history.index,
                                                                            ident=migrant_history.ident)

            # If the destination island has reached population_size, replace a random individual
            if len(destination_island.population) >= destination_island.population_size:
                replace_index = random.randrange(destination_island.population_size)
                destination_island.population[replace_index] = migrant
                migrant_index = replace_index
            else:
                # Otherwise, simply add the migrant to the destination island
                destination_island.population.append(migrant)
                migrant_index = len(destination_island.population) - 1

            # Update history of the destination island with the new migrant
            if migrant_history is not None:
                source_island.statistics.shift_history_after_removal(migrant_history_key)
                destination_island.statistics.add_history_item(replace(migrant_history, key=replace(migrant_history_key,
                                                                                                    index=migrant_index,
                                                                                                    generation=destination_island.completed_generations)))

    def run(self) -> List[SolutionCandidate[A, R, Fitness]]:
        """
        Executes the generations of the islands and migration between them.
        Returns the best solutions from each island.
        """
        self._completed_generations = 0

        for island in self._islands:
            island.create_initial_population()

        generations = self._islands[0].num_generations
        for generation in tqdm(range(generations), unit='generation'):
            for island in self._islands:
                island.evaluate_population(generation)
                island._completed_generations = generation + 1

            # Update statistics with whole population across islands
            self._statistics.update_fitness(chain.from_iterable(island.population for island in self._islands))
            self._completed_generations = generation + 1

            # If this is the last generation, finish here
            if generation == generations - 1:
                continue

            for island in self._islands:
                island.perform_generation(generation)

            if generation % self._migration_interval == 0:
                self._migrate()

        return [island.best_solution() for island in self._islands]

    @property
    def statistics(self):
        return self._statistics

    @property
    def completed_generations(self):
        return self._completed_generations
