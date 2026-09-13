.PHONY: setup test leakage-test backtest predict report clean

VENV := .venv/bin

setup:
	python3 -m venv .venv
	$(VENV)/pip install --upgrade pip
	$(VENV)/pip install -e ".[dev]"

test:
	$(VENV)/pytest tests/ -v

leakage-test:
	$(VENV)/pytest tests/leakage -v

# Build the as-of-date feature dataset for one season, e.g.: make features SEASON=2024
features:
	$(VENV)/python scripts/build_features.py $(SEASON)

# Walk-forward backtest one season against a prior-season training pool, e.g.:
#   make backtest SEASON=2024 PRIOR=2023
backtest:
	$(VENV)/python scripts/run_backtest.py $(SEASON) --prior $(PRIOR)

report:
	$(VENV)/python scripts/evaluate_backtest.py $(SEASON) --prior $(PRIOR)

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -name "__pycache__" -type d -exec rm -rf {} +
