from typing import List
from evolutionary.evolution_base import Algorithm, A, R, Fitness


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

    def _migrate(self):
        # Perform migration between islands' populations
        for i in range(len(self._islands)):
            source_island = self._islands[i]
            destination_island = self._islands[(i + 1) % len(self._islands)]
            # Example migration logic; adjust as needed
            migrants = source_island.population[:self._migration_size]
            destination_island.population.extend(migrants)
            # Ensure populations remain within size constraints

    def run(self):
        for generation in range(self._islands[0].num_generations):  # Assuming all islands have the same num_generations
            for island in self._islands:
                pass
                #  TODO island._perform_generation()
            if generation % self._migration_interval == 0:
                self._migrate()
