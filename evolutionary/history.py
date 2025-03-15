# Key for a SolutionCandidate metadata field for identifying solutions across Algorithms
from dataclasses import dataclass
from typing import Optional

SOLUTION_SOURCE_META_KEY = 'source_meta'
"""Key used in the SolutionCandidate metadata for identifying solutions across Algorithms."""

@dataclass(frozen=True)
class SolutionHistoryKey:
    """
    Key for accessing history items based on (index, generation, ident).
    """
    index: int
    generation: int
    ident: Optional[int] = None

    def __str__(self) -> str:
        return f"HistoryKey(index={self.index}, generation={self.generation}, ident={self.ident})"

    def short_str(self) -> str:
        return f"I={self.index},G={self.generation},ID={self.ident}"

@dataclass(frozen=True)
class SolutionHistoryItem:
    """
    Read-only data class for tracking a solution's history.
    The unique identity of the history item is encapsulated in its key,
    which is composed of (index, generation and optional ident).
    """
    key: SolutionHistoryKey
    mutated: bool
    parent_1: Optional[SolutionHistoryKey] = None  # First parent's key; None only for initial generation
    parent_2: Optional[SolutionHistoryKey] = None  # Optional second parent's key

    @property
    def index(self) -> int:
        return self.key.index

    @property
    def generation(self) -> int:
        return self.key.generation

    @property
    def ident(self) -> Optional[int]:
        return self.key.ident

    def __eq__(self, other):
        if not isinstance(other, SolutionHistoryItem):
            return NotImplemented
        return self.key == other.key

    def __hash__(self):
        return hash(self.key)

    def __str__(self):
        parent_info = f"P1: {self.parent_1}" if self.parent_1 is not None else "P1: None"
        if self.parent_2 is not None:
            parent_info += f", P2: {self.parent_2}"
        mutation_info = "Mutated" if self.mutated else "No Mutation"
        return f"SolutionHistoryItem({self.key}) - {mutation_info}, {parent_info}"

    def short_str(self) -> str:
        return f"{self.key.short_str()},{'M' if self.mutated else 'NM'}"

@dataclass(frozen=True)
class SolutionSourceMeta:
    """
    Data class storing (index, ident) of a solution's source.
    """
    index: int
    ident: int

    def __str__(self) -> str:
        return f"SourceMeta(index={self.index}, ident={self.ident})"