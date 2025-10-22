import re
from pathlib import Path

text = Path("bitrix_assistant.py").read_text(encoding="utf-8")
pattern = r"def _build_tool\(self, spec: BitrixMethodSpec\) -> StructuredTool \| None:\n(?P<body>(?:    .+\n)+?)\n\s+def _compose_description"
match = re.search(pattern, text)
if not match:
    raise SystemExit("pattern not found")
print(match.group("body"))
