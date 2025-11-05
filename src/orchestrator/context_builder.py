"""Построение текстового резюме состояния для передачи модели."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

DEFAULT_SUMMARY_LIMIT = 1200


def build_state_summary(state_snapshot: Dict[str, Any], limit: int = DEFAULT_SUMMARY_LIMIT) -> str:
    """Формирует краткую сводку по состоянию агента для системного промпта.

    Параметры
    ---------
    state_snapshot:
        Словарь с текущим состоянием (`agent_state.to_dict()`).
    limit:
        Максимальная длина результирующей строки. Значение ``0`` или ``None``
        отключает обрезку.
    """

    lines: List[str] = []

    goals = _as_list(state_snapshot.get("goals"))
    if goals:
        displayed = "; ".join(goals[:3])
        extra = " (ещё есть цели)" if len(goals) > 3 else ""
        lines.append(f"Активные цели: {displayed}{extra}.")
    else:
        lines.append("Активных целей нет — ожидаю новую задачу.")

    last_plan = state_snapshot.get("last_plan") or {}
    plan_summary = _safe_text(last_plan.get("summary")) if last_plan.get("summary") else ""
    if plan_summary:
        lines.append(f"Последний план: {plan_summary}.")

    in_progress = _as_list(state_snapshot.get("in_progress"))
    if in_progress:
        items = "; ".join(
            _safe_text(item.get("description") or "ожидается уточнение") for item in in_progress[:3]
        )
        tail = " (есть и другие)" if len(in_progress) > 3 else ""
        lines.append(f"Открытые шаги: {items}{tail}.")

    done = _as_list(state_snapshot.get("done"))
    if done:
        recent = "; ".join(
            _safe_text(entry.get("description", "действие")) for entry in done[-3:]
        )
        lines.append(f"Недавние действия: {recent}.")

    objects = state_snapshot.get("objects") or {}
    known_ids = []
    if objects.get("current_deal_id"):
        known_ids.append(f"текущая сделка {objects['current_deal_id']}")
    if objects.get("current_contact_id"):
        known_ids.append(f"контакт {objects['current_contact_id']}")
    if objects.get("current_company_id"):
        known_ids.append(f"компания {objects['current_company_id']}")
    if objects.get("current_task_id"):
        known_ids.append(f"задача {objects['current_task_id']}")
    if known_ids:
        lines.append("Известные объекты: " + ", ".join(known_ids) + ".")

    summary = "\n".join(lines).strip()
    if not summary:
        summary = "Состояние пустое — никаких действий ранее не выполнялось."

    return _trim_summary(summary, limit)


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return list(value if isinstance(value, Iterable) else [value])


def _safe_text(value: Any) -> str:
    if not value:
        return "без описания"
    text = str(value).strip()
    return text or "без описания"


def _trim_summary(summary: str, limit: int | None) -> str:
    if not limit or len(summary) <= limit:
        return summary
    cutoff = summary.rfind("\n", 0, limit)
    if cutoff == -1:
        cutoff = summary.rfind(" ", 0, limit)
    if cutoff == -1:
        cutoff = limit
    return summary[:cutoff].rstrip() + "\n…"

