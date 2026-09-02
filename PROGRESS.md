# PROGRESS.md – Proof Lab Milestone Tracker

**This is the single source of truth for what the AI agent is allowed to work on right now.**

Before starting any work, the agent must have read `AGENTS.md`.

All permanent rules (branching strategy, commit format, how to mark tasks, what the agent may and may not do) live in `AGENTS.md`.  
This file only holds the milestones themselves.

## M00 – Foundation 🔄 IN PROGRESS

**Branch:** `feat/m00-foundation`  
**Status:** Active – all tasks implemented, awaiting human review

#### Tasks

- [x] Create the full repository structure shown above
- [x] Write `pyproject.toml` with all core dependencies and tool configuration (ruff, mypy, pytest, coverage)
- [x] Implement the configuration system (YAML files + pydantic settings + environment variable overrides). All future strategy parameters must be configuration-driven and versioned with experiments
- [x] Implement structured logging (human-readable in development, machine-readable JSON in production) with levels DEBUG, INFO, WARNING, ERROR, CRITICAL
- [x] Create a minimal deterministic CLI entry point (`prooflab --help`, `prooflab --version`)
- [x] Add a basic CI workflow that runs lint, type-check and unit tests
- [x] Add `README.md`, `LICENSE` and `.env.example` (secrets must never be stored in Git)

#### Human Review Checklist

- [ ] `pip install -e .` succeeds
- [ ] Configuration loads from YAML and environment variables
- [ ] Logging produces the expected structured output
- [ ] CLI responds correctly
- [ ] Layout matches the required structure
- [ ] No trading, labeling, feature or model logic has been introduced

_End of current work._
