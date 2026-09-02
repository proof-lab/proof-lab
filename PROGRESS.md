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

- [x] `pip install -e .` succeeds
- [x] Configuration loads from YAML and environment variables
- [x] Logging produces the expected structured output
- [x] CLI responds correctly
- [x] Layout matches the required structure
- [x] No trading, labeling, feature or model logic has been introduced

---

## M01 – Data Engine 🔄 IN PROGRESS

**Branch:** `feat/m01-data-engine`  
**Status:** Active – all tasks implemented, awaiting human review

### Context for the Agent

Proof Lab begins with data. Before any labels, features or models can exist, the system must ingest historical market data, prove it is clean, and store it in an immutable versioned form.

Supported data includes OHLC, OHLCV, bid/ask, spread, tick data and volume. The baseline is historical MT5 data. Every bar should contain: timestamp, symbol, timeframe, open, high, low, close, volume, tick_volume, spread, source. Optional tick schema: timestamp, symbol, bid, ask, last, volume.

All timestamps must be timezone-aware, stored internally in UTC, and must never be silently converted without recording the transformation. The data validator must detect duplicate timestamps, missing timestamps, impossible OHLC relationships, negative prices or volume, invalid spreads, timestamp disorder, extreme unexplained gaps and corrupted rows. Every dataset must produce a health report containing row counts, missing/duplicate/invalid rows, time span, symbols, timeframes, missing intervals, median and maximum spread, and completeness.

Cleaning must not introduce future information. Allowed operations are duplicate removal, malformed-row removal, timestamp sorting and explicit missing-data handling. The system must not automatically forward-fill price data across market closures unless explicitly configured. Any imputation must be recorded.

Every dataset must have: dataset_id, source, symbol, timeframe, start_time, end_time, created_at, checksum, row_count, feature_version. If raw data changes, a new dataset version is created. Existing experiments continue to reference the original version. Large analytical datasets should use Parquet; DuckDB should query Parquet directly where practical.

### Tasks

- [x] Define the canonical OHLCV schema and optional tick schema (pydantic or equivalent)
- [x] Implement Parquet storage helpers and DuckDB access layer
- [x] Implement dataset versioning with id, checksum and full metadata
- [x] Build the data validator that detects all required problem classes
- [x] Generate the complete health report for every dataset
- [x] Implement the cleaning pipeline (no silent forward-fill)
- [x] Write unit tests using deliberately dirty synthetic data that exercise every validation rule

### Human Review Checklist

- [x] Validator rejects all classes of bad data listed in the requirements
- [x] Cleaning never introduces future information
- [x] Datasets are immutable once versioned
- [x] Health report contains every required field
- [x] Timestamps are timezone-aware and stored in UTC
- [x] No feature, label or model logic has been introduced

---

## M02 – Label Engine

**Branch:** `feat/m02-label-engine`

### Context for the Agent

Proof Lab’s core modelling idea is Predictive Setup Classification. The system does not forecast the next price. It estimates the probability that a predefined target will be reached before a predefined stop within a fixed horizon of bars.

For a long setup: entry = current price, target = entry + target_distance, stop = entry – stop_distance.  
For a short setup the inequalities are reversed. Distances must be configurable in pips, points, percentage or ATR multiples (pips and points are the initial priority).

The canonical classes are 1 = BUY, –1 = SELL, 0 = IGNORE. Internally the system must also preserve richer outcomes: TARGET_FIRST, STOP_FIRST, TIMEOUT, AMBIGUOUS.

Barrier evaluation starts at the entry timestamp, inspects future bars, and stops when the target is hit, the stop is hit, or the horizon expires. When both target and stop fall inside the same OHLC bar, an explicit ambiguity policy decides the outcome. Supported policies are: conservative (assume adverse barrier first – this is the default), optimistic, exclude, and (later) tick resolution. The chosen policy must be recorded in the experiment configuration.

After label generation a quality report is required: total samples, counts and percentages for BUY/SELL/IGNORE and for the richer outcomes, plus a warning if the class distribution is severely imbalanced. Labels must be fully deterministic given the same data and configuration.

Required test cases include target-first, stop-first, timeout, same-bar ambiguity, long and short setups, zero target, invalid stop, and insufficient future data.

### Tasks

- [x] Define the Setup configuration object (target, stop, horizon, direction, ambiguity policy)
- [x] Implement barrier evaluation for long setups
- [x] Implement barrier evaluation for short setups
- [x] Implement the ambiguity policies (conservative default, optimistic, exclude)
- [x] Produce and store the rich outcome metadata
- [ ] Generate the label quality report and imbalance warning
- [ ] Write exhaustive unit tests covering all required edge cases

### Human Review Checklist

- [ ] Labels are fully deterministic
- [ ] Conservative policy is the default
- [ ] Ambiguous bars are handled exactly as configured
- [ ] Rich metadata is preserved alongside the canonical classes
- [ ] Quality report matches the required content
- [ ] All critical test cases pass
- [ ] No feature or model code has been added

---

_End of current work._
