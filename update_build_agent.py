from pathlib import Path

path = Path("bitrix_assistant.py")
text = path.read_text(encoding="utf-8")
unix = text.replace("\r\n", "\n")
old_start = "def build_agent(\n\n    config: BitrixAgentConfig,\n\n    *,\n\n    agent_kwargs: Mapping[str, Any] | None = None,\n\n):\n\n    toolset = BitrixToolset(config.client)\n\n    base_tools = toolset.build_tools()\n\n    extra_tools = build_additional_tools(config.client)\n\n    all_tools = base_tools + extra_tools\n\n    if not all_tools:\n\n        raise RuntimeError(\"В Bitrix24Client не найдено доступных методов для инструментов.\")\n\n\n\n    capabilities = _summarize_tool_capabilities(all_tools)\n\n\n\n    memory = config.memory or ConversationBufferMemory(memory_key=\"chat_history\", return_messages=True)\n\n\n\n    system_prompt = "
new_start = "def build_agent(\n\n    config: BitrixAgentConfig,\n\n    *,\n\n    agent_kwargs: Mapping[str, Any] | None = None,\n\n):\n\n    toolset = BitrixToolset(config.client)\n\n    base_tools = toolset.build_tools()\n\n    extra_tools = build_additional_tools(config.client)\n\n    all_tools = base_tools + extra_tools\n\n    logger.debug(\"Сформирован набор инструментов: base=%s extra=%s total=%s\", len(base_tools), len(extra_tools), len(all_tools))\n\n    if not all_tools:\n\n        raise RuntimeError(\"В Bitrix24Client не найдено доступных методов для инструментов.\")\n\n\n\n    capabilities = _summarize_tool_capabilities(all_tools)\n\n    logger.debug(\"Краткое описание возможностей:\n%s\", capabilities)\n\n    memory = config.memory or ConversationBufferMemory(memory_key=\"chat_history\", return_messages=True)\n\n    system_prompt = "
if old_start not in unix:
    raise SystemExit("build_agent header not found")
unix = unix.replace(old_start, new_start, 1)
updated = unix.replace("\n", "\r\n")
path.write_text(updated, encoding="utf-8")
