# PROGRESS.md – Proof Lab Milestone Tracker

**This is the single source of truth for what the AI agent is allowed to work on right now.**

Before starting any work, the agent must have read `AGENTS.md`.

All permanent rules (branching strategy, commit format, how to mark tasks, what the agent may and may not do) live in `AGENTS.md`.  
This file only holds the milestones themselves.

## M00 – Foundation

**Branch:** `feat/m00-foundation`

### Context for the Agent

You are building **Proof Lab**.

Proof Lab is a quantitative research and algorithmic trading platform. It does not attempt to predict the exact future price. Instead it classifies whether a predefined trading setup (target, stop-loss, and time horizon) is likely to succeed. Every strategy must survive strict chronological validation, realistic cost modeling, robustness testing, and paper trading before any live capital is risked.

The central trust mechanism is the Proof Engine. The governing principle is:

> A strategy is not valuable because it produces an impressive backtest. A strategy is valuable only to the extent that its statistical behavior survives rigorous validation, realistic execution assumptions, robustness testing and controlled live observation.

Implementation language is Python 3.12+. The quantitative engine must remain independent of any UI. Research integrity takes priority over interface polish. The recommended core libraries include numpy, pandas, scipy, scikit-learn, xgboost, PyTorch, Optuna, pydantic, PyYAML, SQLAlchemy, Alembic, pytest, joblib, pyarrow and duckdb. FastAPI is the intended API layer. DuckDB is preferred for analytical workloads; PostgreSQL may be used for application state. Persistence must be isolated behind repository interfaces.

This milestone creates only the technical foundation: repository layout, configuration, logging, packaging and CI. No trading or research logic is written yet.

### Required Repository Structure

```
prooflab/
├── pyproject.toml
├── README.md
├── LICENSE
├── .env.example
├── config/
│   ├── default.yaml
│   ├── development.yaml
│   ├── testing.yaml
│   └── production.yaml
├── src/prooflab/
│   ├── __init__.py
│   ├── api/
│   ├── data/
│   ├── labels/
│   ├── features/
│   ├── models/
│   ├── validation/
│   ├── backtest/
│   ├── risk/
│   ├── regime/
│   ├── news/
│   ├── live/
│   ├── strategies/
│   ├── experiments/
│   ├── explainability/
│   ├── monitoring/
│   └── reporting/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── regression/
│   └── property/
├── notebooks/
├── scripts/
├── data/
│   ├── raw/
│   ├── processed/
│   └── cache/
└── artifacts/
    ├── models/
    ├── experiments/
    └── reports/
```

### Tasks

- [ ] Create the full repository structure shown above
- [ ] Write `pyproject.toml` with all core dependencies and tool configuration (ruff, mypy, pytest, coverage)
- [ ] Implement the configuration system (YAML files + pydantic settings + environment variable overrides). All future strategy parameters must be configuration-driven and versioned with experiments
- [ ] Implement structured logging (human-readable in development, machine-readable JSON in production) with levels DEBUG, INFO, WARNING, ERROR, CRITICAL
- [ ] Create a minimal deterministic CLI entry point (`prooflab --help`, `prooflab --version`)
- [ ] Add a basic CI workflow that runs lint, type-check and unit tests
- [ ] Add `README.md`, `LICENSE` and `.env.example` (secrets must never be stored in Git)

### Human Review Checklist

- [ ] `pip install -e .` succeeds
- [ ] Configuration loads from YAML and environment variables
- [ ] Logging produces the expected structured output
- [ ] CLI responds correctly
- [ ] Layout matches the required structure
- [ ] No trading, labeling, feature or model logic has been introduced

_End of current work._
