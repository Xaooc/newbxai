from pathlib import Path
import re

path = Path("bitrix_assistant.py")
text = path.read_text(encoding="utf-8")
pattern = r"def _build_calendar_events_tool\(client: Bitrix24Client\) -> StructuredTool:\n(?:    .+\n)+?    return StructuredTool.from_function\(\n        func=_run,\n        name=\"calendar_events_for_users\",\n        description=\(\n            \".*?\"\n            \".*?\"\n        \),\n        args_schema=CalendarEventsForUsersArgs,\n        infer_schema=False,\n    \)"
match = re.search(pattern, text, re.S)
if not match:
    raise SystemExit("function block not found")
new_block = '''def _build_calendar_events_tool(client: Bitrix24Client) -> StructuredTool:
    def _run(**kwargs: Any) -> str:
        try:
            args = CalendarEventsForUsersArgs(**kwargs)
        except ValidationError as exc:
            payload = {
                "status": "error",
                "method": "calendar_events_for_users",
                "message": "�������� ������ ������ ��� ��ࠬ���� ����������.",
                "details": exc.errors(),
            }
            return json.dumps(payload, ensure_ascii=False)

        payload = _calendar_events_for_users(client, args)
        return json.dumps(payload, ensure_ascii=False, default=str)

    return StructuredTool.from_function(
        func=_run,
        name="calendar_events_for_users",
        description=(
            "Получает календарные события за указанный день для выбранных сотрудников или всех активных, если `user_ids` не заданы."
            " По умолчанию дата берётся как «сегодня», а часовой пояс — из параметра `timezone` или локальной настройки портала."
        ),
        args_schema=CalendarEventsForUsersArgs,
        infer_schema=False,
    )'''
text = text[:match.start()] + new_block + text[match.end():]
path.write_text(text, encoding="utf-8")
