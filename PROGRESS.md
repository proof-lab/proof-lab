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
- [x] Implement the deterministic Simple Rule baseline with explicitly configured thresholds
- [x] Implement artifact saving and loading that captures all required components
- [x] Create the minimal training pipeline skeleton that respects chronological order and preprocessor rules

All M04 tasks are committed and ready for human review.

### Human Review Checklist

- [x] All models implement the same interface
- [x] Tuning and early stopping cannot see the blind test set
- [x] Artifacts contain everything needed for later inference
- [x] Baselines are present and functional
- [x] No ensemble or calibration logic has been added yet
- [x] Training remains free of future information

---

## M05 – Ensemble & Calibration

**Branch:** `feat/m05-ensemble-calibration`

### Context for the Agent

Individual models are available. They must now be combined and their outputs turned into well-calibrated probabilities.

The ensemble must support hard voting (required for compatibility) as well as probability averaging and weighted probability averaging; stacking may be added later. The application must not assume hard voting is inherently superior.

Raw probabilities must be optionally calibrated with Platt scaling or isotonic regression. Calibration is fitted only on training/validation data; the blind test set remains untouched. Calibration quality should be measured with Brier score, log loss, calibration curve and expected calibration error.

“Confidence” is defined strictly as the calibrated probability of the predicted class. It must never be merely the number of models that agree. Model agreement may be displayed separately.

Every prediction must conform to this schema:

```json
{
  "timestamp": "...",
  "symbol": "EURUSD",
  "prediction": "BUY",
  "probabilities": {
    "BUY": 0.72,
    "SELL": 0.08,
    "IGNORE": 0.2
  },
  "model_votes": {
    "xgboost": "BUY",
    "neural_network": "BUY",
    "svm": "IGNORE"
  }
}
```

### Approved M05 implementation decision

M05 keeps separate long and short ensembles, continuing the M04 one-model-per-configured-direction choice. Long ensembles emit BUY or IGNORE with SELL probability zero; short ensembles emit SELL or IGNORE with BUY probability zero in the required three-class output schema. M05 does not train a joint long/short three-class model. Formal Platt/isotonic calibration uses chronologically valid pre-blind data only; the blind period remains untouched. This formal M05 calibration is distinct from the narrow M04 native-style SVM probability-estimation exception.

### Tasks

- [x] Implement hard voting
- [x] Implement probability averaging
- [x] Implement weighted probability averaging
- [x] Implement Platt scaling calibration
- [x] Implement isotonic regression calibration
- [x] Produce the exact ensemble prediction schema shown above
- [x] Ensure confidence is the calibrated probability of the predicted class
- [x] Write tests for all combination and calibration methods

### Human Review Checklist

- [x] Calibration is fitted only on train/validation data
- [x] Blind test set is never used for calibration
- [x] Output schema matches the specification exactly
- [x] Confidence is never just vote count
- [x] Model votes remain available separately
- [x] Methods are selectable via configuration

---

## M06 – Validation Framework ✅ COMPLETE

**Branch:** `feat/m06-validation-framework`  
**Status:** Human review approved – ready for merge into `development`

### Context for the Agent

Random train/test splits are forbidden for the primary evaluation path. Proof Lab requires chronological splitting and walk-forward validation. A final blind test period (default: the last two years of data) is held out completely. It may never be used for feature selection, hyperparameter selection, model selection, threshold selection, strategy selection, calibration or manual tuning.

Walk-forward may be expanding or rolling; the exact periods are configurable. Because labels look into the future, overlapping observations can create dependence between train and test samples. The validation engine must therefore support purging and embargo periods. The embargo must be at least as long as the maximum target horizon where appropriate.

A leakage detector is required. The test suite must contain deliberately constructed datasets in which leakage would produce an obvious performance improvement; the detector must identify those cases.

Every experiment receives a unique ID (format PL-YYYY-XXXXXX) and must record: experiment_id, dataset_id, full strategy/feature/model/validation/execution configuration, random seed, software version, git commit, created_at, results and artifact locations. Reproducibility also requires recording Python version, library versions, dataset checksum, feature version and label version.

The system must track the number of experiments performed and warn when many experiments have been run against the same validation period. The blind test set should be protected from casual repeated inspection; revealing full blind-test results may require explicit confirmation.

### Tasks

- [x] Implement chronological train / validation / blind-test splitting
- [x] Implement expanding and rolling walk-forward generators
- [x] Implement purging
- [x] Implement embargo logic (length ≥ label horizon)
- [x] Build the leakage detector and prove it catches synthetic leaking cases
- [x] Create the experiment registry with full reproducibility metadata
- [x] Protect the blind test set from casual inspection and multiple-testing abuse

### Human Review Checklist

- [x] Random splitting is impossible on the primary evaluation path
- [x] Blind test period is fully isolated
- [x] Embargo length is at least the label horizon
- [x] Leakage detector works on the constructed examples
- [x] Reproducibility metadata is complete
- [x] No backtesting or risk logic has been added

---

## M07 – Backtesting Engine ✅ COMPLETE

**Branch:** `feat/m07-backtesting-engine`  
**Status:** Completed and reviewed

### Context for the Agent

The backtesting engine is independent of model training. Its architecture is:

Model → Predictions → Signal Engine → Backtester → Execution Simulator → Portfolio → Metrics

The backtester must never modify the predictions it receives. A prediction becomes a trade only when it satisfies configured conditions (prediction direction, minimum calibrated probability, risk checks, regime filter, news blackout, etc.).

Position sizing is risk-based: risk_amount = account_equity × risk_per_trade; position_size = risk_amount / stop_loss_value. Position size must respect broker limits.

The execution cost model must support spread, commission, slippage, swap and execution delay. Spread modelling supports historical spread, fixed spread, spread multiplier and stress spread, with Normal / Conservative / Stress scenarios. Slippage is configurable (initially fixed pips). Commission may be per lot, per unit, per transaction or percentage. Swap is modelled when trades can remain open across financing periods.

Each simulated order must contain: order_id, timestamp, symbol, side, requested_price, fill_price, quantity, stop, target, spread, commission, slippage, status, exit_reason.

Required metrics:

- Returns: Total return, Annualized return, CAGR
- Risk: Maximum drawdown, Average drawdown, Drawdown duration, Volatility, VaR, CVaR
- Risk-adjusted: Sharpe, Sortino, Calmar
- Trading: Trade count, Win rate, Loss rate, Average win/loss, Profit factor, Expectancy, Average holding time
- Costs: Total spread, commission, slippage, swap, total execution costs

The equity curve is generated from the actual simulated account balance and must support gross equity, net equity and drawdown.

### Tasks

- [x] Implement the signal engine that turns predictions into trade decisions under the configured filters
- [x] Implement the full order lifecycle
- [x] Implement spread modelling (historical, fixed, multiplier, stress)
- [x] Implement commission, slippage and swap cost models
- [x] Implement portfolio accounting and the equity curve
- [x] Calculate the complete metrics suite listed above
- [x] Write tests covering entry, exit, stop, target, costs, partial fills and lifecycle edge cases

### Human Review Checklist

- [x] Backtester never alters incoming predictions
- [x] All cost components are independently inspectable
- [x] Equity curve is derived from simulated balance
- [x] All required metrics are present and correct
- [x] Position sizing respects broker limits
- [x] Core lifecycle and cost tests pass

---

## M08 – Proof Engine & Robustness ✅ COMPLETE

**Branch:** `feat/m08-proof-robustness`  
**Status:** Completed and reviewed

### Context for the Agent

The Proof Engine is the central trust mechanism of Proof Lab. Its purpose is to answer: “Does this strategy work outside the data used to develop it?”

It must produce a scorecard (Net Return, Profit Factor, Sharpe, Sortino, Max Drawdown, Expectancy, Win Rate, Trade Count, Total Costs), an equity curve with drawdown, global feature importance (gain, split importance, optional SHAP), and a clear separation between global importance and per-trade local explanation.

Robustness testing is mandatory for any strategy that reaches this stage:

- Parameter sensitivity (nearby target/stop values). A strategy that only works at exact parameters must receive a robustness warning.
- Spread stress and slippage stress
- Year-by-year performance
- Regime performance (high/low volatility, trending/ranging, and optionally high/low spread, volume, session)
- Trade-order Monte Carlo (minimum 1 000 simulations, preferred 10 000) reporting median return, 5th/95th percentile return, median/95th percentile drawdown, probability of loss and probability of ruin

The engine assigns an explicit Proof Status — NOT PROVEN, WEAK, PROMISING or ROBUST — based on predefined rules, never on an arbitrary score. A strategy should only be considered validated when there is no known leakage, the blind test is completed, minimum trade count is satisfied, net performance is positive, risk limits are acceptable, robustness tests pass and execution stress tests pass. Thresholds are configurable.

Every completed experiment should produce a research report containing configuration, dataset information, feature set, model architecture, validation methodology, performance, risk, costs, robustness, regime analysis, feature importance, warnings and an evidence-based conclusion. Explicit research warnings (LOW TRADE COUNT, HIGH PARAMETER SENSITIVITY, HIGH CLASS IMBALANCE, HIGH OUT-OF-SAMPLE DEGRADATION, PERFORMANCE DEPENDS ON LOW SPREAD, POSSIBLE OVERFITTING, INSUFFICIENT DATA, MODEL DRIFT DETECTED, etc.) must be raised when appropriate.

### Tasks

- [x] Build the Proof Engine scorecard
- [x] Produce equity-curve and drawdown data
- [x] Calculate global feature importance
- [x] Implement parameter sensitivity tests
- [x] Implement spread and slippage stress tests
- [x] Implement Monte Carlo trade-order reshuffling (≥ 1 000 runs)
- [x] Implement regime performance analysis
- [x] Define and apply the explicit Proof Status rules
- [x] Generate the full research report with warnings

### Human Review Checklist

- [x] Proof Status is determined by explicit rules
- [x] Monte Carlo produces the required percentile statistics
- [x] Sensitivity analysis flags fragile parameter choices
- [x] Regime analysis covers the required cases
- [x] Report is complete and evidence-based
- [x] Appropriate warnings are generated

---

## M09 – Risk Engine ✅ COMPLETE

**Branch:** `feat/m09-risk-engine`  
**Status:** Completed and reviewed

### Context for the Agent

Risk management operates independently of model predictions. The model may say BUY while the risk engine says REJECT; the risk engine always wins.

Required limits: max risk per trade, max open positions, max total exposure, max leverage, max symbol exposure, max daily loss, max weekly loss, max consecutive losses. Maximum daily loss (example: 3 % of start-of-day equity) disables new trades when breached.

The kill switch must: stop new orders, cancel pending orders, optionally close open positions according to configured policy, persist its state, and produce an audit event.

Trading must automatically pause when any of the following occurs: market data is stale, broker connection is lost, model artifact is invalid, feature calculation fails, unexpected spread occurs, risk limits are exceeded, news blackout is active, model confidence is below threshold, system clock is invalid, or a duplicate signal is detected.

Position sizing remains risk-based as defined earlier. All of these controls must be testable in isolation from any live broker.

### Tasks

- [x] Implement risk-based position sizing
- [x] Implement the full set of exposure, loss and consecutive-loss limits
- [x] Implement maximum daily and weekly loss handling
- [x] Implement the kill switch with the required behaviour and audit event
- [x] Implement the safety-condition checks that automatically pause trading
- [x] Write tests proving that risk rules can reject any model signal

### Human Review Checklist

- [x] Risk engine can reject BUY/SELL signals
- [x] Kill switch persists state and emits an audit event
- [x] Daily loss limit disables new trades
- [x] All required limits are present and enforced
- [x] No live broker code has been introduced

---

## M10 – Paper Trading 🔄 IN PROGRESS

**Branch:** `feat/m10-paper-trading`  
**Status:** Active – agent is working here

### Context for the Agent

Paper trading must precede live trading. The architecture is:

Live Market Data → Feature Engine → Model → Risk Engine → Paper Execution → Portfolio

The live feature engine must reuse the exact same feature implementations used during training. There must never be a separate “live RSI” versus “training RSI”. Every model artifact must include feature schema, feature order, preprocessing pipeline, model, calibration model and strategy parameters; the inference engine must reject an artifact if required features are missing.

Live market data handling must detect stale ticks, missing ticks, duplicate ticks, out-of-order ticks and abnormal spreads. Paper trades are recorded with the same schema as live trades.

A strategy moves through explicit approval states: RESEARCH → VALIDATED → PAPER_TRADING → APPROVED → LIVE_ENABLED → SUSPENDED → RETIRED. A strategy must never become live-enabled automatically. Live trading remains disabled by default.

### Tasks

- [x] Build the live market-data consumer with staleness, gap, duplicate and quality checks
- [ ] Ensure live feature calculation reuses the identical training code path
- [ ] Implement inference that loads a complete model artifact and rejects incomplete ones
- [ ] Implement paper execution and paper portfolio accounting
- [ ] Record paper trades with the full trade schema
- [ ] Write an integration test of the complete paper-trading loop

### Human Review Checklist

- [ ] Feature code path is identical to training
- [ ] Incomplete artifacts are rejected
- [ ] Paper trades are fully recorded with the same schema as live trades
- [ ] No real broker orders can be sent from this code
- [ ] Bad or stale data pauses trading

---

_End of current work._
