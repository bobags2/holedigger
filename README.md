# holedigger

Search market data for exploitable structure — dislocations, forced-flow fingerprints,
microstructure holes — instead of forecasting prices, with the multiple-testing problem
treated as the central engineering constraint rather than a footnote.

**Status: data audit + parquet layer + null factory + selection-bias floor + broker cost model
(Fusion Markets, 8/8 symbols) + three registered families (645 hypotheses — empty, correctly)
+ live spread tape (recording; analyzer ready). No live-money candidates yet.**

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

### 4. Cost probe (`scripts/cost_probe.py`)

Broker commissions converted to bp of notional (last prices 2026-09-30):

| | XAUUSD | EURUSD | GBPUSD |
|---|---|---|---|
| $ / 1.00 lot | 4.96 | 2.51 | 2.97 |
| bp per side | 0.119 | 0.222 | 0.224 |
| round trip (if quoted per side) | 0.239 | 0.443 | 0.448 |

All eight symbols are now wired: XAGUSD 3.69 $/lot → 0.244 bp round trip; XTIUSD, US500,
NAS100 and USDX are zero-commission — for those the spread is the *entire* cost (and index
CFD spreads are the broker's revenue, unlike raw FX, so the tape matters most there).

Reversion edge per trade by timeframe (exact-adjacency rows): M1 0.008 · M5 0.038 · M15 0.096 ·
M30 0.138 · H1 0.192 · H4 0.36 bp — the edge grows with horizon, but the t-stat collapses
(H4: t < 2, gold flips negative) and per-day gross flattens at ~3–8 bp/day.

Conditioning on prior-move size (M5, top-5% moves, where a limit-order variant would have to
live): best cell in the entire grid is GBPUSD at 0.221 bp/trade against 0.224 bp round-trip
commission under the most generous cost reading — break-even before spread. Everything else
loses 0.05–0.44 bp per trade. Commission is invariant to order type, so limit orders cannot
rescue this family; the spread-*earning* variant (passive fills, adverse selection) is a
different business and needs quote data to evaluate — see Next.

### 5. First registered sweep: fixed session windows (`src/window_sweep.py`)

552 candidates = 8 symbols × 69 windows (entry on hours 00–19 UTC; holds 1/2/4/8h; exit capped
20:00 UTC so no trade crosses the daily break). Statistic: |t| of the mean per-trade log return,
calibrated as a family against 200 per-symbol permutation worlds. Every candidate registered in
`artifacts/registry.jsonl` before evaluation.

- null family max |t|: mean 3.20, p95 3.85, max 4.53 (analytic E[max] reference 3.05)
- real family max |t| = **3.16 → 46th percentile. The family is empty.**

What the calibration buys: the top rows look genuinely attractive in isolation — XAGUSD
10:00–11:00 short pays 2.86 bp/trade (t = −3.16, negative in both halves), XAUUSD 00:00 +2h pays
2.27 bp (net of commission +2.03) — and all of it is inside what 552 tries on shuffled worlds
produce by chance. A coherent-looking cluster exists at 03:00 UTC (dollar up; majors, gold and
indices down; 7 of 8 symbols aligned), but its sign alignment was chosen *after* looking at the
results, so it is not evidence yet — the correct next step is a pre-registered pooled test with
a joint surrogate that preserves cross-symbol correlation. Full results:
`artifacts/window_sweep.json`.

### 6. Pooled re-test of the 03:00 cluster — pre-registered, negative

The apparent 03:00 UTC cluster (dollar-strength direction, all eight symbols) was flagged in
advance as post-hoc, then tested by `src/pooled_test.py` on the common 5-min grid: equal-risk
oriented basket (USDX long, seven shorts), joint row permutations preserving cross-symbol
correlation, family = all 69 windows.

- candidate 03:00+1h: t_pooled = **1.01** (69.5th percentile of its own null)
- family max |t_pooled| = 1.88 → **17th percentile** (null mean 2.36, p95 3.17)

Per-symbol magnitudes were unchanged when restricted to the panel period (EURUSD −2.60,
USDX +2.85), so the cluster was not a regime artifact — it was inside what chance produces.
Construction note: the common grid attenuates measurements (entry/exit bars shift when any
symbol is missing a 5-min bucket), which is why the decisive negative remains the original
family null (46th percentile) computed on each symbol's own bars. Thread closed; no
re-testing of the same hypothesis with new statistics.

### 7. Pair dislocation reversion — pre-registered, negative

Six pairs with mechanism stories (sibling indices, metals, European majors, gold/oil vs the
dollar index), one fixed rule: rolling 2-day beta from levels OLS (trailing window, no
look-ahead), 5-day spread z-score, enter |z| ≥ {2, 3}, exit |z| ≤ {0, 0.5} or 1-day timeout,
one position per pair. 24 candidates, joint-permutation null (contemporaneous co-movement
preserved, serial structure destroyed).

- null family max |t|: mean 2.13, p95 3.03
- real family max |t| = **1.70 → 23.5th percentile. The family is empty.**

Shape notes: where there is any signal it points at *continuation*, not reversion —
US500/NAS100 dislocations keep going (reversion edge −3.8 bp/trade, t = −1.7; the momentum
side would clear its zero commissions comfortably, at still-noise |t|). XTIUSD/USDX
dislocations are large (+23 bp/trade) and hopelessly noisy (n = 76, t = 0.6, avg |β| 5.3).
Most trades run to the timeout instead of reverting to the exit level — at this timescale,
dislocations persist. Results: `artifacts/pair_sweep.json`.

## Layout

```
src/audit.py        what the data actually is; uniform_start, anomalous months, integrity
src/ingest.py       CSV -> parquet (+raw/curated split, sha256, read-back parity)
src/surrogates.py   null factory: joint time-index permutation / Politis-Romano bootstrap
src/noise_floor.py  selection-bias floor: E[max t] over N trials, validated
src/registry.py     append-only hypothesis registry; evaluate-before-register raises
src/costs.py        broker cost model (commission table, break-even spread)
src/window_sweep.py family 1: session windows + null calibration + registry
src/pooled_test.py  pre-registered pooled dollar-strength test (joint surrogate)
src/pair_sweep.py   family 3: pair dislocation reversion (rolling beta/z, sequential sim)
scripts/first_look.py  end-to-end calibration smoke on real data
scripts/cost_probe.py  edge vs costs by timeframe and move-size bucket
scripts/tape_analysis.py  spread distributions from a live tape
tests/              contract tests for the statistical core (plain python, no pytest)
```

## How to run

```powershell
python -m src.audit                    # ~20s, writes artifacts/audit/
python -m src.ingest                   # ~30s, writes data/parquet/ + manifest
python -m src.noise_floor --years 5.9  # selection-bias floor table
python -m scripts.first_look           # toy family vs surrogate null
python -m scripts.cost_probe           # edge vs costs by timeframe and move-size bucket
python -m src.window_sweep --worlds 200  # family 1: session windows (~30s)
python -m src.pooled_test --worlds 200   # pre-registered dollar-strength re-test (~20s)
python -m src.pair_sweep --worlds 200    # family 3: pair dislocations (~2 min)
python -m scripts.tape_analysis          # analyze newest ../ctrader/tape/*.csv

python -m tests.test_registry; python -m tests.test_surrogates; python -m tests.test_noise_floor; python -m tests.test_window_sweep; python -m tests.test_pooled; python -m tests.test_pair_sweep

# live spread tape (run from ../ctrader; refresh the token first if stale):
# python -m ctbridge.tape --config config.json --minutes 60
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

1. Analyze the live tape (`scripts/tape_analysis.py`) → per-symbol/hour spread distributions;
   wire measured spreads into `src/costs.py`, then re-rank the session-window near-misses.
2. Symbol-holdout splits: select on 6 symbols, verify the same rule on 2 unseen ones.
3. Project-level multiplicity: 645 hypotheses are registered; when a candidate survives its
   family null, the registry is what lets the cross-family selection bias be priced
   (White/Hansen style) instead of asserted away.
4. Mask audited `anomalous_months` inside feature builders rather than dropping the series.
5. Scale the factory: block-length policy per family, more worlds, full-world surrogates
   with synthetic OHLC for path-dependent rules.

## Field notes

One bug during the v0 build silently replaced every `rows` count in the audit report with
120 (a shadowed variable) — caught by reading the numbers, not by an exception. That is the
class of failure this project exists to make impossible to ship: every search produces a
tradable-looking rule, every bug produces a plausible number.
