# PROGRESS.md – Proof Lab Milestone Tracker

**This is the single source of truth for what the AI agent is allowed to work on right now.**

Before starting any work, the agent must have read `AGENTS.md`.

All permanent rules (branching strategy, commit format, how to mark tasks, what the agent may and may not do) live in `AGENTS.md`.  
This file only holds the milestones themselves.

## M00 – Foundation ✅ COMPLETE

**Branch:** `feat/m00-foundation`  
**Status:** Merged into `development`

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

## M01 – Data Engine ✅ COMPLETE

**Branch:** `feat/m01-data-engine`  
**Status:** Merged into `development`

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

## M02 – Label Engine ✅ COMPLETE

**Branch:** `feat/m02-label-engine`  
**Status:** Merged into `development`

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

## M03 – Feature Engine ✅ COMPLETE

**Branch:** `feat/m03-feature-engine`  
**Status:** Merged into `development`

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

- [x] Create the feature metadata system and registry
- [x] Implement the price feature family
- [x] Implement the momentum feature family
- [x] Implement the volatility feature family
- [x] Implement the trend feature family
- [x] Implement cyclical time features
- [x] Build the feature pipeline with explicit lookback declarations and leakage guards
- [x] Handle warm-up rows and enforce fit/transform separation for any scalers
- [x] Write unit and property tests that prove absence of look-ahead bias for every family

### Human Review Checklist

- [x] Every feature declares its maximum lookback
- [x] No feature can access future data
- [x] Scalers are fitted only on training data
- [x] Metadata is complete for every feature
- [x] Core families required by the MVP are present
- [x] Tests demonstrate no leakage
- [x] Microstructure features are not fabricated from OHLCV alone

---

## M04 – Model Engine 🔄 IN PROGRESS

**Branch:** `feat/m04-model-engine`  
**Status:** Active – agent is working here

### Context for the Agent

Proof Lab trains multiple models so their combination can later be evaluated. Before the main ensemble is judged, simple baselines are required so we can measure whether complexity actually adds value.

Required baselines: Random Classifier, Majority Classifier, Logistic Regression, Simple Rule Strategy, XGBoost-only.

Core models:

- **XGBoost** (“Rule-Maker”): configurable tree depth, learning rate, number of estimators, subsampling, column sampling, regularization, class weights. Hyperparameter tuning may use only training/validation data.
- **PyTorch Neural Network**: configurable architecture (hidden layers, units, dropout, learning rate, batch size, epochs, weight decay). Early stopping must use validation data only. Initial shape is Input → Dense → ReLU → Dropout → Dense → ReLU → Dropout → Output.
- **SVM** (“Statistician”): configurable kernel, C, gamma, class weights, probability estimation. Feature scaling must be performed through a pipeline fitted only on training data.

All models share a common interface. Every trained model is stored as an artifact that must contain at least: model weights, preprocessor, feature schema, feature order, training metadata. The blind test set must never be used for tuning or early stopping.

A minimal training pipeline skeleton should execute: Load Dataset → Validate → Generate Labels → Generate Features → Remove warm-up rows → Chronological Split → Fit Preprocessors on Training Only → Train models → Persist Artifact.

### Approved M04 implementation decisions

- **SVM probability estimation (M04 compatibility exception):** Fit the SVM and its scaler on an earlier training subpartition. Fit only the SVM-specific probability-estimation mechanism on a later training subpartition after purging every earlier sample whose complete label horizon reaches that subpartition. Do not refit the SVM or scaler on the probability-fitting rows. The ordinary validation partition and blind period are unavailable to probability fitting. This replaces the native mechanism's shuffled internal folds with an explicit chronological holdout. Record both subpartitions, the purge, and the probability method in the artifact.
- **M04 versus M05:** The exception is restricted to making the SVM's required native-style probability interface functional. It does not introduce a reusable calibration framework, model-wide post-processing, calibration-method comparison or selection, or claims that probabilities are formally calibrated. Formal Platt/isotonic calibration and its evaluation remain deferred to M05. The existing human review item about calibration is interpreted subject only to this explicitly approved SVM exception; its checkbox remains for the human reviewer.
- **Setup directions and split eligibility:** Train separate models for each explicitly configured setup direction. Retain a sample only when its complete future horizon lies within its permitted partition, even when a barrier happens to be reached sooner. Apply this rule at training/validation boundaries and inside SVM training. This prevents later observations from influencing earlier samples or their eligibility through outcomes.
- **Blind-period isolation:** M04 training accepts explicit chronological boundaries and loads only observations before the blind boundary. Do not generate blind labels or features, evaluate blind predictions, select parameters from the blind period, or produce blind metrics. The blind period remains untouched.
- **Simple Rule baseline:** Add a separate deterministic task with explicitly supplied indicator thresholds and direction. Learn no thresholds and apply no probability fitting or implicit calibration. Any probability-shaped output is a one-hot encoding of the deterministic action, not an estimated probability of setup success.

### Tasks

- [x] Define a common model interface
- [x] Implement Random and Majority baselines
- [x] Implement Logistic Regression baseline
- [x] Implement the XGBoost model with the required configurability
- [x] Implement the configurable PyTorch neural network with validation-only early stopping
- [x] Implement the SVM with probability estimates and training-fitted scaling
- [ ] Implement the deterministic Simple Rule baseline with explicitly configured thresholds
- [x] Implement artifact saving and loading that captures all required components
- [ ] Create the minimal training pipeline skeleton that respects chronological order and preprocessor rules

### Human Review Checklist

- [ ] All models implement the same interface
- [ ] Tuning and early stopping cannot see the blind test set
- [ ] Artifacts contain everything needed for later inference
- [ ] Baselines are present and functional
- [ ] No ensemble or calibration logic has been added yet
- [ ] Training remains free of future information

---

_End of current work._
