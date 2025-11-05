"""Главный класс автономного агента Bitrix24."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping

from .client import BitrixWebhookClient
from .config import BitrixConfig
from .io import ConsoleIO, IOHandler
from .memory import AgentMemory
from .normalizer import parse_amount, parse_deadline
from .planner import PlanStep, SimplePlanner
from .tools import BitrixError, bitrix_batch, bitrix_call


logger = logging.getLogger(__name__)


Handler = Callable[[PlanStep], Mapping[str, Any] | None]


@dataclass
class ExecutionContext:
    """Контекст выполнения плана."""

    memory: AgentMemory
    config: BitrixConfig
    io: IOHandler
    raw_goal: str
    steps: List[PlanStep]
    outputs: List[Mapping[str, Any]] = field(default_factory=list)
    report: str | None = None


class BitrixAutonomousAgent:
    """Автономный агент, реализующий цикл Plan → Act → Observe → Adjust → Report."""

    def __init__(self, *, config: BitrixConfig | None = None, io: IOHandler | None = None) -> None:
        self.config = config or BitrixConfig.from_env()
        self.io = io or ConsoleIO()
        self.memory = AgentMemory()
        self.planner = SimplePlanner(self.memory)
        self.client = BitrixWebhookClient(config=self.config)
        self._handlers: Dict[str, Handler] = {
            "collect_context": self._handle_collect_context,
            "generic_action": self._handle_generic_action,
            "collect_deal_context": self._handle_collect_deal_context,
            "collect_deal_stages": self._handle_collect_deal_stages,
            "crm.deal.list": self._handle_deal_list,
            "crm.deal.add": self._handle_deal_add,
            "crm.deal.update": self._handle_deal_update,
            "tasks.task.list": self._handle_task_list,
            "prepare_task_fields": self._handle_prepare_task_fields,
            "tasks.task.add": self._handle_task_add,
            "tasks.task.update": self._handle_task_update,
            "summarize": self._handle_summarize,
        }

    def run(self, prompt: str) -> Dict[str, Any]:
        """Запустить агента на одном запросе."""

        self.memory.goal = prompt
        steps = self.planner.build_plan(prompt)
        context = ExecutionContext(
            memory=self.memory,
            config=self.config,
            io=self.io,
            raw_goal=prompt,
            steps=steps,
        )
        if steps:
            plan_lines = "\n".join(
                f"{index}. {step.description}" for index, step in enumerate(steps, start=1)
            )
            self.io.notify(f"Построен план работы:\n{plan_lines}")
        else:
            self.io.notify("План пустой, агент выполнит цель напрямую.")

        total_steps = len(steps)
        for index, step in enumerate(steps, start=1):
            self.memory.next = step.description
            self.io.notify(f"Шаг {index}/{total_steps}: {step.description}")
            handler = self._handlers.get(step.action, self._handle_generic_action)
            try:
                result = handler(step)
            except BitrixError as exc:
                logger.error("Ошибка шага %s: %s", step.description, exc)
                self.memory.add_risk(str(exc))
                if not self.memory.unknowns:
                    answer = self.io.ask(
                        f"Возникла ошибка: {exc}. Укажите дополнительные сведения или напишите 'пропустить'"
                    )
                    if answer.strip().lower() == "пропустить":
                        self.memory.mark_step_done(step.description)
                        continue
                    self.memory.knowns.setdefault("notes", []).append(answer)
                else:
                    for field in list(self.memory.unknowns):
                        answer = self.io.ask(f"Укажите {field}")
                        self.memory.knowns[field] = answer
                        self.memory.unknowns.remove(field)
                result = handler(step)
            if result:
                context.outputs.append(result)
            self.memory.mark_step_done(step.description)
            self.io.notify(f"Шаг завершён: {step.description}")
        context.report = self._build_report()
        if context.report:
            self.io.notify(f"Итоговый отчёт:\n{context.report}")
        else:
            self.io.notify("Агент завершил работу, отчёт не сформирован.")
        return {
            "memory": self.memory.to_json(),
            "results": context.outputs,
            "report": context.report,
        }

    # ---- обработчики шагов ----
    def _handle_collect_context(self, step: PlanStep) -> Mapping[str, Any]:
        """Запросить дополнительную информацию."""

        missing = self.memory.unknowns
        if not missing:
            return {"step": step.description, "status": "нет неизвестных"}
        answers: Dict[str, str] = {}
        for item in list(missing):
            answer = self.io.ask(f"Уточните {item}")
            answers[item] = answer
            self.memory.knowns[item] = answer
            self.memory.unknowns.remove(item)
        return {"step": step.description, "answers": answers}

    def _handle_generic_action(self, step: PlanStep) -> Mapping[str, Any]:
        return {"step": step.description, "status": "обработано по умолчанию"}

    def _handle_collect_deal_context(self, step: PlanStep) -> Mapping[str, Any]:
        """Попробовать извлечь параметры сделки из цели."""

        goal = self.memory.goal.lower()
        if "на" in goal:
            try:
                amount, currency = parse_amount(goal.split("на", 1)[1])
                self.memory.knowns.setdefault("deal_fields", {})["OPPORTUNITY"] = float(amount)
                self.memory.knowns["deal_fields"]["CURRENCY_ID"] = currency
            except ValueError:
                if "сумма сделки" not in self.memory.unknowns:
                    self.memory.unknowns.append("сумма сделки")
        if (
            "назнач" in goal
            and "ID ответственного" not in self.memory.unknowns
            and "ID ответственного" not in self.memory.knowns
        ):
            self.memory.unknowns.append("ID ответственного")
        if "сделк" in goal and "созд" in goal:
            title = self.memory.goal.strip()
            self.memory.knowns.setdefault("deal_fields", {})["TITLE"] = title
        self._extract_responsible_from_goal(self.memory.goal)
        self._extract_stage_from_goal(self.memory.goal)
        return {"step": step.description, "knowns": self.memory.knowns.get("deal_fields", {})}

    def _handle_deal_list(self, step: PlanStep) -> Mapping[str, Any]:
        filters: MutableMapping[str, Any] = {}
        deal_fields = self.memory.knowns.get("deal_fields", {})
        if "TITLE" in deal_fields:
            filters["TITLE"] = deal_fields["TITLE"]
        response = self.client.crm_deal_list(
            filter=dict(filters),
            select=["ID", "TITLE", "STAGE_ID", "CATEGORY_ID"],
        )
        result = response.get("result", []) if isinstance(response, Mapping) else []
        if result:
            self.memory.knowns.setdefault("existing_deals", result)
        return {"step": step.description, "count": len(result)}

    def _handle_collect_deal_stages(self, step: PlanStep) -> Mapping[str, Any]:
        """Собрать список стадий сделки и привести поля к ID."""

        deal_fields = self.memory.knowns.setdefault("deal_fields", {})
        category_id = deal_fields.get("CATEGORY_ID")
        existing = self.memory.knowns.get("existing_deals")
        if not category_id and existing:
            category_id = existing[0].get("CATEGORY_ID")
            if category_id:
                deal_fields["CATEGORY_ID"] = category_id

        stages: List[Mapping[str, Any]] = []
        entity_id = None
        if category_id is not None:
            entity_id = f"DEAL_STAGE_{category_id}"
            stages = list(self.client.crm_status_list(filter={"ENTITY_ID": entity_id}))
        if not stages:
            entity_id = "DEAL_STAGE"
            stages = list(self.client.crm_status_list(filter={"ENTITY_ID": entity_id}))

        stage_map: Dict[str, str] = {}
        stage_ids: List[str] = []
        for stage in stages:
            status_id = str(stage.get("STATUS_ID", "")).upper()
            name = str(stage.get("NAME", "")).strip().lower()
            if status_id:
                stage_ids.append(status_id)
            if name:
                stage_map[name] = status_id

        self.memory.knowns["deal_stage_source"] = entity_id
        self.memory.knowns["deal_stage_map"] = stage_map
        self.memory.knowns["deal_stage_ids"] = sorted(set(stage_ids))

        stage_name = deal_fields.get("STAGE_NAME")
        if stage_name and "STAGE_ID" not in deal_fields:
            mapped = stage_map.get(stage_name.strip().lower())
            if mapped:
                deal_fields["STAGE_ID"] = mapped
            else:
                if "название стадии" not in self.memory.unknowns:
                    self.memory.unknowns.append("название стадии")
        if "STAGE_ID" in deal_fields and stage_ids and str(deal_fields["STAGE_ID"]).upper() not in stage_ids:
            raise BitrixError("Указанный STAGE_ID отсутствует в справочнике")

        return {
            "step": step.description,
            "stage_count": len(stages),
            "source": entity_id,
        }

    def _handle_deal_add(self, step: PlanStep) -> Mapping[str, Any]:
        deal_fields = dict(self.memory.knowns.get("deal_fields", {}))
        if "TITLE" not in deal_fields:
            raise BitrixError("Не указан TITLE для сделки")
        if "ASSIGNED_BY_ID" not in deal_fields:
            resolved = self._resolve_responsible_id()
            if resolved is not None:
                deal_fields["ASSIGNED_BY_ID"] = resolved
            elif "ID ответственного" in self.memory.knowns:
                deal_fields["ASSIGNED_BY_ID"] = int(self.memory.knowns.get("ID ответственного"))
            else:
                if "ID ответственного" not in self.memory.unknowns:
                    self.memory.unknowns.append("ID ответственного")
                raise BitrixError("Нужен ID ответственного")
        if "STAGE_ID" in deal_fields:
            stage_ids = {str(item).upper() for item in self.memory.knowns.get("deal_stage_ids", [])}
            if stage_ids and str(deal_fields["STAGE_ID"]).upper() not in stage_ids:
                raise BitrixError("STAGE_ID не найден среди загруженных стадий")
        self.memory.knowns["deal_fields"] = deal_fields
        deal_id = self.client.crm_deal_add(deal_fields)
        if deal_id:
            link = f"https://portal.magnitmedia.ru/crm/deal/details/{deal_id}/"
            self.memory.progress.append(f"Создана сделка #{deal_id}")
            self.memory.knowns["last_deal_id"] = deal_id
            self.memory.knowns.setdefault("links", []).append(link)
            return {"step": step.description, "deal_id": deal_id, "link": link}
        return {"step": step.description, "status": "нет id"}

    def _handle_deal_update(self, step: PlanStep) -> Mapping[str, Any]:
        deal_id = self.memory.knowns.get("last_deal_id")
        if not deal_id:
            raise BitrixError("Нет ID сделки для обновления")
        deal_fields = dict(self.memory.knowns.get("deal_fields", {}))
        result = self.client.crm_deal_update(deal_id, deal_fields)
        return {"step": step.description, "result": result}

    def _handle_task_list(self, step: PlanStep) -> Mapping[str, Any]:
        response = self.client.tasks_task_list(select=["ID", "TITLE", "STATUS"])
        tasks = response.get("result", {}).get("tasks", []) if isinstance(response, Mapping) else []
        self.memory.knowns["tasks"] = tasks
        return {"step": step.description, "count": len(tasks)}

    def _handle_prepare_task_fields(self, step: PlanStep) -> Mapping[str, Any]:
        goal = self.memory.goal
        fields: Dict[str, Any] = {"TITLE": goal[:250]}
        if "завтра" in goal.lower():
            deadline = parse_deadline("завтра 18:00").dt.isoformat()
            fields["DEADLINE"] = deadline
        self.memory.knowns["task_fields"] = fields
        return {"step": step.description, "fields": fields}

    def _handle_task_add(self, step: PlanStep) -> Mapping[str, Any]:
        fields = dict(self.memory.knowns.get("task_fields", {}))
        if "TITLE" not in fields:
            raise BitrixError("Не указан TITLE задачи")
        task = self.client.tasks_task_add(fields)
        task_id = task.get("id") if isinstance(task, Mapping) else None
        if task_id:
            self.memory.knowns["task_id"] = task_id
            responsible_id = task.get("responsibleId") if isinstance(task, Mapping) else None
            if responsible_id:
                self.memory.knowns["task_responsible"] = responsible_id
            link = None
            if responsible_id:
                link = (
                    "https://portal.magnitmedia.ru/company/personal/user/"
                    f"{responsible_id}/tasks/task/view/{task_id}/"
                )
                self.memory.knowns.setdefault("links", []).append(link)
            self.memory.progress.append(f"Создана задача #{task_id}")
        return {"step": step.description, "task_id": task_id}

    def _handle_task_update(self, step: PlanStep) -> Mapping[str, Any]:
        task_id = self.memory.knowns.get("task_id")
        if not task_id:
            raise BitrixError("Нет ID задачи для обновления")
        fields = dict(self.memory.knowns.get("task_fields", {}))
        task = self.client.tasks_task_update(task_id, fields)
        if task:
            self.memory.progress.append(f"Обновлена задача #{task_id}")
        return {"step": step.description, "result": task}

    def _handle_summarize(self, step: PlanStep) -> Mapping[str, Any]:
        summary = self._build_report()
        return {
            "step": step.description,
            "memory": self.memory.to_json(),
            "report": summary,
        }

    # ---- утилиты ----
    def enrich_knowns(self, data: Mapping[str, Any]) -> None:
        """Добавить известные данные в память."""

        for key, value in data.items():
            self.memory.knowns[key] = value

    def batch(self, commands: Mapping[str, str]) -> Mapping[str, Any]:
        """Выполнить batch-запрос."""

        return self.client.batch(commands)

    def call(self, method: str, params: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        """Выполнить одиночный вызов Bitrix24."""

        return bitrix_call(method, dict(params or {}), config=self.config)

    def run_batch_sequence(self, batched: Iterable[Mapping[str, str]]) -> List[Mapping[str, Any]]:
        """Выполнить несколько батчей подряд."""

        responses = []
        for commands in batched:
            responses.append(bitrix_batch(commands, config=self.config))
        return responses

    def _build_report(self) -> str:
        """Сформировать итоговый отчёт для пользователя."""

        lines: List[str] = []

        outcome_parts: List[str] = []
        if self.memory.progress:
            outcome_parts.append(
                ", ".join(self.memory.progress[-3:])
            )
        if not outcome_parts:
            outcome_parts.append("План выполнен без изменений")
        lines.append(f"Итог: {'; '.join(outcome_parts)}.")

        links = self.memory.knowns.get("links", [])
        key_bits: List[str] = []
        if last_deal := self.memory.knowns.get("last_deal_id"):
            key_bits.append(f"Сделка #{last_deal}")
        if task_id := self.memory.knowns.get("task_id"):
            key_bits.append(f"Задача #{task_id}")
        if links:
            key_bits.extend(links)
        if key_bits:
            lines.append("Ключи: " + "; ".join(str(item) for item in key_bits) + ".")
        else:
            lines.append("Ключи: нет новых идентификаторов.")

        if self.memory.unknowns:
            lines.append("Дальше: требуется уточнить " + ", ".join(self.memory.unknowns) + ".")
        else:
            lines.append("Дальше: готов к новым задачам.")

        return "\n".join(lines)

    # ---- служебные методы ----
    def _extract_responsible_from_goal(self, goal: str) -> None:
        """Определить ФИО ответственного из текста цели."""

        pattern = re.compile(
            r"ответственн\w*\s+(?:на|за)?\s*([A-ZА-ЯЁ][a-zа-яё]+)\s+([A-ZА-ЯЁ][a-zа-яё]+)(?:\s+([A-ZА-ЯЁ][a-zа-яё]+))?",
            re.IGNORECASE,
        )
        match = pattern.search(goal)
        if not match:
            return
        name = match.group(1)
        last_name = match.group(2)
        middle = match.group(3)
        self.memory.knowns["responsible_name"] = {
            "NAME": name.capitalize(),
            "LAST_NAME": last_name.capitalize(),
        }
        if middle:
            self.memory.knowns["responsible_name"]["SECOND_NAME"] = middle.capitalize()

    def _extract_stage_from_goal(self, goal: str) -> None:
        """Выделить название стадии из свободного текста."""

        pattern = re.compile(r"стад(?:и[ея]|ию)\s+[«\"']?([A-ZА-ЯЁa-zа-яё0-9\s]+)[»\"']?", re.IGNORECASE)
        match = pattern.search(goal)
        if not match:
            return
        stage_name = match.group(1).strip()
        if not stage_name:
            return
        self.memory.knowns.setdefault("deal_fields", {})["STAGE_NAME"] = stage_name

    def _resolve_responsible_id(self) -> int | None:
        """Попробовать определить ID ответственного через user.get."""

        responsible = self.memory.knowns.get("responsible_name")
        if not responsible:
            return None
        filter_data = {
            "NAME": responsible.get("NAME"),
            "LAST_NAME": responsible.get("LAST_NAME"),
        }
        if not all(filter_data.values()):
            return None
        users = list(self.client.user_get(filter=filter_data, select=["ID", "NAME", "LAST_NAME"]))
        for user in users:
            if str(user.get("NAME", "")).lower() == filter_data["NAME"].lower() and \
               str(user.get("LAST_NAME", "")).lower() == filter_data["LAST_NAME"].lower():
                self.memory.knowns["ID ответственного"] = int(user["ID"])
                if "ID ответственного" in self.memory.unknowns:
                    self.memory.unknowns.remove("ID ответственного")
                return int(user["ID"])
        return None
