import random
from typing import List
from evolutionary.evolution_base import SolutionCandidate, Selector, SingleObjectiveFitness


class TournamentSelector(Selector[SingleObjectiveFitness]):
    def __init__(self, tournament_size: int):
        """
        Tournament selection: Selects the best individual from a random subset of the population.
        Args:
            tournament_size (int): Number of individuals in each tournament.
        """
        assert tournament_size > 1, "Tournament size must be greater than 1."
        self.tournament_size = tournament_size

    def select(self, candidates: List[SolutionCandidate]) -> SolutionCandidate:
        selected = random.sample(candidates, self.tournament_size)
        winner = max(selected, key=lambda candidate: candidate.fitness)
        return winner


class RankSelector(Selector[SingleObjectiveFitness]):
    def __init__(self, selection_pressure: float = 1.5):
        """
        Rank selection: Selects individuals based on ranking with specified selection pressure.
        Args:
            selection_pressure (float): Determines bias towards higher-ranked individuals.
        """
        assert selection_pressure > 1.0, "Selection pressure must be greater than 1."
        self.selection_pressure = selection_pressure

    def select(self, candidates: List[SolutionCandidate]) -> SolutionCandidate:
        candidates.sort(key=lambda candidate: candidate.fitness, reverse=True)
        total_candidates = len(candidates)
        if total_candidates == 1:
            return candidates[0]

        rank_weights = [(self.selection_pressure - 2.0 * (self.selection_pressure - 1.0) * (i / (total_candidates - 1)))
                        for i in range(total_candidates)]
        chosen_index = random.choices(range(total_candidates), weights=rank_weights, k=1)[0]
        return candidates[chosen_index]


class RouletteWheelSelector(Selector[SingleObjectiveFitness]):
    """
    Roulette Wheel selection: Selects individuals based on fitness proportionate probabilities.
    """

    def select(self, candidates: List[SolutionCandidate]) -> SolutionCandidate:
        total_fitness = sum(candidate.fitness for candidate in candidates if candidate.fitness is not None)
        pick = random.uniform(0, total_fitness)
        current = 0
        for candidate in candidates:
            if candidate.fitness is not None:
                current += candidate.fitness
                if current >= pick:
                    return candidate
        return random.choice(candidates)
