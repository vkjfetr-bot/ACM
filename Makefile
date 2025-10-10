VENV?=.venv

.PHONY: setup test lint mypy
setup:
	python -m venv $(VENV)
	$(VENV)/Scripts/pip install -U pip
	$(VENV)/Scripts/pip install -e .
	$(VENV)/Scripts/pip install pytest mypy

test:
	$(VENV)/Scripts/pytest -q

mypy:
	$(VENV)/Scripts/mypy acmnxt --strict

