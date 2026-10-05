# holedigger

Search market data for exploitable structure — dislocations, forced-flow fingerprints,
microstructure holes — instead of forecasting prices, with the multiple-testing problem
treated as the central engineering constraint rather than a footnote.

**v0 status: data audit + parquet layer + surrogate null factory + selection-bias floor +
one calibration look on real data. No cost model, no sweeps, no candidates yet.**

## Why the framing

A forecasting loss optimizes a conditional mean; whatever edge exists lives in tails and
conditionals. What ML actually buys here is hypothesis throughput, amortized scoring, and
**calibrated nulls** — not rows. The thing that kills naive versions of this project is
arithmetic, not modelling: the best of N pure-noise rules has an expected t-stat of
~sqrt(2 ln N), so the best-of-N noise strategy looks like

$$ SR_{ann} \approx \frac{\sqrt{2\ln N}}{\sqrt{\text{years}}} $$

Every rule you test raises the bar your best rule must clear. Breadth is a cost, not an edge.

## What v0 measured (on the real feed)

### 1. The data is not what the filenames say

`src/audit.py` measures per-month resolution and density. Reality on this broker feed:

| symbol | raw rows | curated rows | honest M1 from | anomalous months |
|---|---|---|---|---|
| EURUSD | 2,175,532 | 2,124,393 | 2020-12-28 | 2023-07 |
| GBPUSD | 2,191,523 | 2,140,736 | 2020-12-28 | — |
| XAUUSD | 2,090,491 | 2,082,703 | 2020-11-12 | — |
| US500 | 2,054,896 | 2,047,386 | 2020-11-11 | — |
| NAS100 | 2,086,793 | 2,078,839 | 2020-11-12 | — |
| USDX | 1,662,560 | 1,645,458 | 2021-08-02 | — |
| XTIUSD | 1,712,052 | 1,704,162 | 2021-11-01 | — |
| XAGUSD | 2,086,355 | 2,078,485 | 2020-11-12 | — |

One bar per trading day through Sep 2019 (server-side decimation on deep walk-back
windows), hourly from Oct 2019 through Oct 2020, and uniform M1 only from Nov–Dec 2020.
EURUSD has a real outage month (2023-07: 15 active days, half the bars). ~15.9M curated
M1 bars total.
Per-symbol reports in `artifacts/audit/`, hashes and parity in `data/manifest.json`.

### 2. The selection-bias floor (`src/noise_floor.py`)

Effective span ~5.9 years:

| rules tried | E[max t] | SR of best noise rule | 95th pct |
|---|---|---|---|
| 1,000 | 3.26 | 1.33 | 1.58 |
| 10,000 | 3.86 | 1.58 | 1.80 |
| 100,000 | 4.40 | 1.80 | 2.00 |
| 1,000,000 | 4.86 | 2.00 | 2.20 |

Validated against direct simulation of iid rule backtests (`tests/test_noise_floor.py`).
Note the identity $t \approx SR\sqrt{Y}$: sampling frequency does not change this. M1 data
buys microstructure access and capacity, not statistical evidence.

### 3. First calibration look (`scripts/first_look.py`)

Toy family: sign of the lagged M5 return, momentum/reversion, lags 1–60, three symbols,
406,035 aligned bars, 5.76 years, family size 48. No costs, no holdout, no mechanism.

- Real family max |t| = **5.79**. Permutation null (200 worlds, `mean_block=1`, which
  destroys serial structure of every horizon while preserving marginals and lag-0
  cross-correlation): mean 2.22, p95 3.04, max 3.46. z = 7.9, 100th percentile.
  The analytic floor for N=48 (2.19) matches the empirical null (2.22) — the two
  independent calibration tools agree.
- The winner is lag-1 reversion, replicated across all three symbols:
  t = −5.59 (EURUSD), −5.79 (GBPUSD), −3.42 (XAUUSD).
- The lag-1 probe (restricting to exact 5-minute adjacency, 99.3% of rows) keeps it:
  t = −5.11 / −5.21 / −3.24 — so it is real bar-to-bar structure, not gap-fill.
- What it pays: **0.025 / 0.029 / 0.036 bp per trade**, gross, before any cost.

Compare with the IntradayMomentum study's own numbers (`../ctrader/calgo/IntradayMomentum`):
breakeven there sat at 0.03–0.5 bp per side. This effect is below even the most generous
breakeven, and ~10–40× below realistic retail spreads. Statistically overwhelming,
economically dead — the exact shape of the queue: both verdicts are required, and the
second one is not optional.

## Layout

```
src/audit.py        what the data actually is; uniform_start, anomalous months, integrity
src/ingest.py       CSV -> parquet (+raw/curated split, sha256, read-back parity)
src/surrogates.py   null factory: joint time-index permutation / Politis-Romano bootstrap
src/noise_floor.py  selection-bias floor: E[max t] over N trials, validated
src/registry.py     append-only hypothesis registry; evaluate-before-register raises
scripts/first_look.py  end-to-end calibration smoke on real data
tests/              contract tests for the statistical core (plain python, no pytest)
```

## How to run

```powershell
python -m src.audit                    # ~20s, writes artifacts/audit/
python -m src.ingest                   # ~30s, writes data/parquet/ + manifest
python -m src.noise_floor --years 5.9  # selection-bias floor table
python -m scripts.first_look           # toy family vs surrogate null

python -m tests.test_registry; python -m tests.test_surrogates; python -m tests.test_noise_floor
```

Requires Python 3.13 + numpy/pandas/pyarrow (global install is fine; see `requirements.txt`).
Price data comes from `../ctrader/history` and is never modified.

## Design rules this repo encodes

1. **Audit before modelling.** Anything derived (features, labels, regimes) inherits
   resolution and outage artifacts silently. This project's first finding was that its own
   data was 40% decimated by row count.
2. **Register before evaluating.** `src/registry.py` refuses to record an evaluation for an
   unregistered hypothesis id. The number of things tried must be reconstructible from the
   log, not from memory.
3. **Calibrate against a null that contains no holes.** A candidate's bar is the null-max
   distribution of the *same search procedure*, not zero. Block length is an explicit,
   recorded choice sitting between the noise scale and the candidate horizon.
4. **Every survivor needs a mechanism story and a measured cost model.** Neither exists yet;
   that is the next phase, not a solved problem.

## Next

1. Cost model: log live bid/ask through `ctbridge` into a spread tape per symbol/session;
   measured spreads become part of every evaluation.
2. Mechanism-prioritized generators (session opens, rollover and swap windows, dividend/index
   adjustments, calendar) registered with an explicit family budget.
3. Symbol-holdout splits: select on 6 symbols, verify the same rule on 2 unseen ones.
4. Mask audited `anomalous_months` inside feature builders rather than dropping the series.
5. Scale the factory: block-length policy per family, more worlds, and full-world surrogates
   with synthetic OHLC for path-dependent rules.

## Field notes

One bug during the v0 build silently replaced every `rows` count in the audit report with
120 (a shadowed variable) — caught by reading the numbers, not by an exception. That is the
class of failure this project exists to make impossible to ship: every search produces a
tradable-looking rule, every bug produces a plausible number.
