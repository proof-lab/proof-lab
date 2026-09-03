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

## M02 – Label Engine 🔄 IN PROGRESS

**Branch:** `feat/m02-label-engine`  
**Status:** Active – all tasks implemented, awaiting human review

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
- [x] Generate the label quality report and imbalance warning
- [x] Write exhaustive unit tests covering all required edge cases

### Human Review Checklist

- [x] Labels are fully deterministic
- [x] Conservative policy is the default
- [x] Ambiguous bars are handled exactly as configured
- [x] Rich metadata is preserved alongside the canonical classes
- [x] Quality report matches the required content
- [x] All critical test cases pass
- [x] No feature or model code has been added

---

## M03 – Feature Engine

**Branch:** `feat/m03-feature-engine`

### Context for the Agent

Once labels exist, the system needs quantitative features that describe market state at each point in time without ever looking into the future.

Features are organised into families:

- **Price**: return_1/2/3/6/12/24, range_1/3/6/12, body_size, upper_wick, lower_wick, distance_from_high/low, close_to_open/high/low
- **Momentum**: RSI, ROC, MACD, MACD_signal, MACD_histogram, momentum_3/6/12
- **Volatility**: ATR, ATR_percent, rolling_std, rolling_range, true_range, volatility_percentile
- **Trend**: EMA_fast/slow, EMA_distance, SMA_fast/slow, ADX, trend_slope (all parameters configurable)
- **Time**: cyclical encoding – hour_sin/cos = sin/cos(2π · hour / 24), dow_sin/cos = sin/cos(2π · day / 7)
- **Microstructure** (only when the required data exists, never faked from OHLCV): bid_ask_spread, spread_percentile, tick_count, tick_rate, short-term volatility/range, price acceleration, tick_volume_change

Every feature must carry metadata: feature_name, family, description, parameters, required_columns, lookback_period, uses_future_data, version. Every feature generator must declare its maximum lookback. The framework should reject suspicious future dependencies.

A feature is valid at time t only if all required information was available at t. Scaling and other preprocessing statistics must be fitted only on training data and then applied to validation and test data; they must never be fitted on the entire dataset. Warm-up rows (where the lookback is not yet satisfied) must be handled correctly.

The system must support experiment modes that compare feature families: Price only, Price+Volatility, Price+Momentum, Price+Volatility+Momentum, All Standard Features, All Standard + Microstructure.

The same feature implementations will later be reused for live inference; there must be only one code path.

### Tasks

- [ ] Create the feature metadata system and registry
- [ ] Implement the price feature family
- [ ] Implement the momentum feature family
- [ ] Implement the volatility feature family
- [ ] Implement the trend feature family
- [ ] Implement cyclical time features
- [ ] Build the feature pipeline with explicit lookback declarations and leakage guards
- [ ] Handle warm-up rows and enforce fit/transform separation for any scalers
- [ ] Write unit and property tests that prove absence of look-ahead bias for every family

### Human Review Checklist

- [ ] Every feature declares its maximum lookback
- [ ] No feature can access future data
- [ ] Scalers are fitted only on training data
- [ ] Metadata is complete for every feature
- [ ] Core families required by the MVP are present
- [ ] Tests demonstrate no leakage
- [ ] Microstructure features are not fabricated from OHLCV alone

---

_End of current work._
