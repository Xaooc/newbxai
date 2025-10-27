from __future__ import annotations

import json
import logging
import re
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


PLAN_SYSTEM_PROMPT = textwrap.dedent(
    """
    You are the planning assistant for Bitrix24. Your job is to understand the user's goal,
    figure out which intermediate steps are required, and map them to the Bitrix24 tools
    that will achieve the result without extra back-and-forth.

    Available tools:
    {capabilities}

    Planning rules:
    - Keep the plan within {max_steps} steps.
    - Always decide whether the goal needs several actions in sequence. If the user did not list them
      explicitly but they are required, spell them out in the right order.
    - For every step choose the Bitrix24 tool that best fits and write it into the "tool" field. When
      several tools are needed, break the work into multiple steps. If the action is manual, set "tool"
      to null and explain why.
    - Use "details" to describe what will happen, the data you will use, and the expected outcome.
    - Capture risks, prerequisites, or reminders for the user in "notes".
    - Do not duplicate steps or add unnecessary work.
    - The plan must be self-contained so that finishing all steps delivers the user's goal.

    Return JSON shaped like:
    {
      "plan_summary": "short sentence about the goal",
      "steps": [
        {
          "id": 1,
          "title": "step name",
          "tool": "tasks.task.add",
          "details": "what happens and which data is required"
        }
      ],
      "notes": [
        "important reminders or assumptions"
      ]
    }
    """
).strip()




class PlanGenerationError(RuntimeError):
    """Raised when the language model fails to produce a valid plan."""


@dataclass
class PlanStep:
    number: int
    title: str
    details: str
    tool: Optional[str] = None


@dataclass
class ActionPlan:
    request: str
    steps: List[PlanStep]
    summary: str
    raw_source: str
    notes: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def format_for_user(self) -> str:
        lines: List[str] = []
        lines.append("Here is what I suggest doing next.")
        if self.summary:
            lines.append(f"Summary: {self.summary}")
        lines.append("")
        lines.append("Step-by-step plan:")
        for step in self.steps:
            lines.append(f"{step.number}. {step.title}")
            if step.details:
                lines.append(f"   What happens: {step.details}")
            if step.tool:
                lines.append(f"   Tool: {step.tool}")
        if self.notes:
            lines.append("")
            lines.append("Keep in mind:")
            for note in self.notes:
                lines.append(f"- {note}")
        lines.append("")
        lines.append("If everything looks good, reply with yes/ok/go. Tell me what to change if it needs edits.")
        return "\n".join(lines)



def _extract_text(payload: object) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    content = getattr(payload, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: List[str] = []
        for item in content:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or item.get("value")
                if isinstance(text, str):
                    pieces.append(text)
        return "\n".join(pieces)
    return str(payload)


def _strip_code_fences(text: str) -> str:
    fenced = text.strip()
    if fenced.startswith("```"):
        fenced = re.sub(r"^```[a-zA-Z0-9]*", "", fenced)
        if fenced.endswith("```"):
            fenced = fenced[: -3]
    return fenced.strip()


class PlanGenerator:
    def __init__(
        self,
        llm: BaseChatModel,
        *,
        capabilities: str,
        max_steps: int = 5,
    ) -> None:
        self._llm = llm
        self._capabilities = capabilities.strip()
        self._max_steps = max(1, max_steps)

    def generate(self, request: str, *, history: Sequence[str] | None = None) -> ActionPlan:
        if not request or not request.strip():
            raise ValueError("request must be a non-empty string")
        history_text = self._format_history(history)
        system_prompt = PLAN_SYSTEM_PROMPT.format(
            capabilities=self._capabilities or "—",
            max_steps=self._max_steps,
        )
        user_prompt = self._build_user_prompt(request.strip(), history_text)
        logger.debug("Отправляю запрос на генерацию плана: %s", user_prompt)
        response = self._llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        raw_text = _extract_text(response)
        logger.debug("Сырой ответ планировщика: %s", raw_text)
        plan_payload = self._parse_plan(raw_text)
        steps = self._convert_steps(plan_payload.get("steps", []))
        if not steps:
            raise PlanGenerationError("Модель вернула пустой список шагов.")
        summary = str(plan_payload.get("plan_summary") or "").strip()
        notes_raw = plan_payload.get("notes") or []
        notes = [str(item).strip() for item in notes_raw if str(item).strip()]
        return ActionPlan(
            request=request.strip(),
            steps=steps,
            summary=summary,
            raw_source=raw_text,
            notes=notes,
        )

    def _format_history(self, history: Sequence[str] | None) -> str:
        if not history:
            return ""
        trimmed = [chunk.strip() for chunk in history if chunk and chunk.strip()]
        return "\n".join(trimmed[-4:])

    def _build_user_prompt(self, request: str, history_text: str) -> str:
        history_section = history_text or "No recent conversation"
        return textwrap.dedent(
            f"""
            User request: {request}

            Recent context:
            {history_section}

            Build a clear plan for the assistant. Decide how many steps are required and put them
            in order so the goal is reached without extra questions. For every step mention the data
            it needs and which Bitrix24 tool should be called. The plan is limited to {self._max_steps} steps.
            """
        ).strip()




    def _parse_plan(self, raw_text: str) -> dict:
        cleaned = _strip_code_fences(raw_text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error("Не удалось разобрать план как JSON: %s", exc)
            raise PlanGenerationError("Модель вернула невалидный JSON.") from exc

    def _convert_steps(self, steps_payload: Sequence[object]) -> List[PlanStep]:
        steps: List[PlanStep] = []
        for idx, entry in enumerate(steps_payload, start=1):
            if not isinstance(entry, dict):
                logger.debug("Пропускаю некорректный шаг плана: %r", entry)
                continue
            number = self._safe_int(entry.get("id"), idx)
            title = str(entry.get("title") or entry.get("action") or "").strip()
            details = str(entry.get("details") or entry.get("explanation") or "").strip()
            tool_raw = entry.get("tool") or entry.get("tool_name")
            tool = str(tool_raw).strip() if tool_raw else None
            if not title:
                logger.debug("Пропускаю шаг без названия: %r", entry)
                continue
            steps.append(
                PlanStep(
                    number=number,
                    title=title,
                    details=details,
                    tool=tool or None,
                )
            )
        ordered = sorted(steps, key=lambda step: step.number)
        trimmed = ordered[: self._max_steps]
        normalized: List[PlanStep] = []
        for idx, step in enumerate(trimmed, start=1):
            normalized.append(
                PlanStep(
                    number=idx,
                    title=step.title,
                    details=step.details,
                    tool=step.tool,
                )
            )
        return normalized


    @staticmethod
    def _safe_int(value: object, fallback: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback


__all__ = ["ActionPlan", "PlanGenerator", "PlanGenerationError", "PlanStep"]



