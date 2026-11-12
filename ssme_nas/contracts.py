"""Interfaces for native search spaces, staged evaluators, and memory."""
from typing import Any, Mapping, Protocol, Sequence

Genotype = tuple[str, ...]
Record = dict[str, Any]


class SearchSpace(Protocol):
    def canonical(self, genotype: Genotype) -> str:
        """Return the complete canonical architecture identity."""
        ...

    def validate(self, genotype: Genotype) -> None:
        """Reject invalid native genotypes without reading performance."""
        ...

    def validate_edit(self, parent: Genotype, child: Genotype) -> None:
        """Require a legal native edit with the permitted edit order."""
        ...


class ValidationEvaluator(Protocol):
    # Deliberately no Test access in the search-loop interface.
    def validation(self, genotype: Genotype, intent: Mapping[str, Any]) -> Record:
        """Return one seeded validation repeat; intent includes the run search_seed."""
        ...


class CompiledMemory(Protocol):
    data: Mapping[str, Any]


class AuditSink(Protocol):
    def append(self, kind: str, record: Mapping[str, Any]) -> None:
        """Persist intent/evaluation/decision before the next dependent action."""
        ...


class MemoryCompiler(Protocol):
    def compile(self, records: Sequence[Record], concepts: Sequence[Mapping]) -> CompiledMemory:
        """Compile current-run observed evidence with frozen structural concepts."""
        ...


PrivateCompiler = MemoryCompiler  # Compatibility with the original interface name.
