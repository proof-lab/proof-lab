# Proof Lab

**A research-first platform for turning trading ideas into tested, risk-controlled strategies.**

---

## What it is

Proof Lab is a quantitative research and algorithmic trading platform.  

It does not try to predict the next price move.  
It answers a more specific question:

> Given the current market state, how likely is this predefined setup (target, stop-loss, and time horizon) to succeed?

Every strategy is then forced through chronological validation, realistic cost modelling, robustness tests, and paper trading before it is ever allowed near live capital.

The heart of the system is the **Proof Engine** — the component that decides whether a strategy has actually earned the right to be trusted.

---

## The problem it solves

Most trading systems fail for the same reasons:

- They look profitable in a backtest and fall apart on new data  
- They were tuned until the historical results looked good  
- Costs, slippage, and realistic execution were ignored or underestimated  
- There was no clear separation between research, validation, and live trading  
- Risk controls were an afterthought  
- It was hard to know *why* a strategy stopped working

Traders and researchers end up with strategies that look strong on paper and weak in reality, with little evidence about when or why they fail.

Proof Lab exists to close that gap.

---

## Why it is built this way

Financial markets are not static. A strategy that worked in one regime can fail in the next. Impressive backtests are easy to produce and easy to believe. Real robustness is harder.

Proof Lab is built on a simple belief:

> A strategy is not valuable because it produces an impressive backtest.  
> A strategy is valuable only to the extent that its statistical behaviour survives rigorous validation, realistic execution assumptions, robustness testing, and controlled live observation.

That belief drives every design choice:

- Research integrity comes before interface polish  
- The system must make leakage and look-ahead bias difficult, not easy  
- The final stretch of data is held out and protected from casual tuning  
- Costs and execution assumptions are treated as first-class concerns  
- Risk controls can override any model signal  
- Live trading stays off until explicit, audited approval  
- The same logic used in research is reused in paper and live trading so there is no hidden gap between “what we tested” and “what we run”

The goal is not to claim that any strategy will be profitable.  
The goal is to make it possible to know, with much higher confidence, whether a strategy has been properly tested — and when it should be stopped.

---

## How it solves the problem

Proof Lab turns a trading idea into a controlled research and execution pipeline:

1. **Define the setup clearly**  
   Target, stop-loss, and time horizon are explicit. The system then labels historical data according to whether that setup would have succeeded.

2. **Describe the market without looking ahead**  
   Features are generated with strict rules against future information. The same feature code is later reused for live decisions.

3. **Train and combine models carefully**  
   Multiple models are trained and combined. Probabilities are calibrated. Confidence has a defined statistical meaning.

4. **Validate the hard way**  
   Chronological splits, walk-forward testing, and a protected blind period are used instead of random shuffling. Leakage checks and reproducibility metadata are required.

5. **Test under realistic conditions**  
   Backtests include spread, commission, slippage and other costs. Robustness tests, stress scenarios, Monte Carlo analysis and regime breakdowns are part of the process.

6. **Enforce risk independently**  
   Position sizing, loss limits, exposure limits and a kill switch sit outside the model. The risk engine can reject any signal.

7. **Paper trade before going live**  
   Strategies must run against live data in paper mode. Only after explicit approval can live trading be enabled — and it remains off by default.

8. **Keep everything inspectable**  
   Experiments are versioned, artifacts are complete, and the Proof Engine produces clear reports and status rather than opaque scores.

The result is a path from idea → evidence → controlled deployment, with as little self-deception as possible along the way.

---

## Status

Proof Lab is under active development.  
The quantitative engine is being built before the interface.  
Live trading remains disabled by default.

---

## License

See `LICENSE`.
