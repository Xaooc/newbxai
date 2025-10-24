"""Простая реализация графа состояний для тестов."""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Dict, List, Tuple


NodeCallable = Callable[[dict], Awaitable[dict] | dict]
RouterCallable = Callable[[dict], str]

END = "__end__"


class StateGraph:
    """Минимальный граф состояний."""

    def __init__(self, state_type: Any) -> None:  # noqa: D401, ANN401
        self._state_type = state_type
        self._nodes: Dict[str, NodeCallable] = {}
        self._edges: Dict[str, List[str]] = {}
        self._conditional_edges: Dict[str, Tuple[RouterCallable, Dict[str, str]]] = {}
        self._entry: str | None = None

    def add_node(self, name: str, func: NodeCallable) -> None:
        self._nodes[name] = func

    def set_entry_point(self, name: str) -> None:
        self._entry = name

    def add_edge(self, src: str, dst: str) -> None:
        self._edges.setdefault(src, []).append(dst)

    def add_conditional_edges(
        self,
        src: str,
        router: RouterCallable,
        mapping: Dict[str, str],
    ) -> None:
        self._conditional_edges[src] = (router, mapping)

    def compile(self) -> "CompiledGraph":
        if self._entry is None:
            raise ValueError("Не задана точка входа")
        return CompiledGraph(
            nodes=self._nodes,
            edges=self._edges,
            conditional=self._conditional_edges,
            entry=self._entry,
        )


class CompiledGraph:
    """Исполняемый граф."""

    def __init__(
        self,
        *,
        nodes: Dict[str, NodeCallable],
        edges: Dict[str, List[str]],
        conditional: Dict[str, Tuple[RouterCallable, Dict[str, str]]],
        entry: str,
    ) -> None:
        self._nodes = nodes
        self._edges = edges
        self._conditional = conditional
        self._entry = entry

    async def ainvoke(self, state: dict) -> dict:
        """Асинхронное выполнение графа."""

        current = self._entry
        current_state = dict(state)
        while True:
            node = self._nodes[current]
            result = node(current_state)
            if asyncio.iscoroutine(result):
                result = await result
            if result is not None:
                current_state = result
            if current in self._conditional:
                router, mapping = self._conditional[current]
                branch = router(current_state)
                current = mapping[branch]
                if current == END:
                    return current_state
                continue
            next_nodes = self._edges.get(current, [])
            if not next_nodes:
                return current_state
            next_node = next_nodes[0]
            if next_node == END:
                return current_state
            current = next_node
