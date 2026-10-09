.PHONY: test audit format lint check reproduce clean

VENV = .venv
PYTHON = $(VENV)/bin/python
PYTEST = $(VENV)/bin/pytest
RUFF = $(VENV)/bin/ruff

test:
	$(PYTEST) tests/ -v

audit:
	$(PYTHON) -m src.lean.audit.runner --milestone $(or $(M),M0)

format:
	$(RUFF) format src/ tests/ vault/

lint:
	$(RUFF) check src/ tests/ vault/

check: lint test audit

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
