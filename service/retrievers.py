"""Retriever implementations used by the JTBD DSPy agent."""

from __future__ import annotations

from typing import Iterable, List

from modaic import PrecompiledConfig, Retriever


class NullRetrieverConfig(PrecompiledConfig):
    """Configuration placeholder for the null retriever."""


class NotesRetrieverConfig(PrecompiledConfig):
    """Serializable configuration for the in-memory notes retriever."""

    notes: List[str] = []
    top_k: int = 3


class NullRetriever(Retriever):
    """No-op retriever for environments without contextual data."""

    config: NullRetrieverConfig

    def __init__(self, config: NullRetrieverConfig | None = None, **kwargs):
        super().__init__(config or NullRetrieverConfig(), **kwargs)

    def retrieve(self, query: str) -> str:  # type: ignore[override]
        return ""


class NotesRetriever(Retriever):
    """Very small keyword-based retriever backed by an in-memory list of notes."""

    config: NotesRetrieverConfig

    def __init__(
        self,
        notes: Iterable[str] | None = None,
        top_k: int | None = None,
        config: NotesRetrieverConfig | None = None,
        **kwargs,
    ):
        if config is None:
            cfg = NotesRetrieverConfig()
            cfg.notes = list(notes or [])
            if top_k is not None:
                cfg.top_k = int(top_k)
        else:
            cfg = config
            if notes is not None:
                cfg.notes = list(notes)
            if top_k is not None:
                cfg.top_k = int(top_k)

        super().__init__(cfg, **kwargs)

    def retrieve(self, query: str) -> str:  # type: ignore[override]
        terms = {token for token in query.lower().split() if token}
        if not terms:
            return ""

        scored: List[tuple[int, str]] = []
        for note in self.config.notes:
            tokens = {token for token in note.lower().split() if token}
            score = len(terms & tokens)
            if score > 0:
                scored.append((score, note))

        scored.sort(key=lambda item: item[0], reverse=True)
        top_matches = [note for _, note in scored[: self.config.top_k]]
        return "\n".join(top_matches)
