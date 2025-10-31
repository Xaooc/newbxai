.PHONY: test lint typecheck check-warnings check-all

PYTEST ?= pytest
RUFF ?= ruff
MYPY ?= mypy

test:
$(PYTEST)

check-warnings:
$(PYTEST) -W error::DeprecationWarning

lint:
$(RUFF) check src tests

typecheck:
$(MYPY) --explicit-package-bases --follow-imports=skip src/app/telegram_runner.py src/adapters/telegram_bot.py

check-all: lint typecheck check-warnings
