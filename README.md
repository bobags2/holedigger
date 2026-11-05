# holedigger

Search market data for exploitable structure — dislocations, forced-flow fingerprints,
microstructure holes — instead of forecasting prices, with the multiple-testing problem
treated as the central engineering constraint rather than a footnote.

**Status: two sources live. cTrader mid bars (28 instruments, audited and ingested) + MT5
tick data (10 instruments, 350 days each, ~430M ticks → mid bars with per-bar spreads).
Six registered families, ~2,300 hypotheses, **zero survivors** — including the two live
candidates the MT5 spread data killed at their actual trading hours (copper −4.8 bp/trade,
lead −15 to −26 bp/trade). No live-money candidates yet.**

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
NAS100 and USDX are zero-commission — for those the spread is the *entire* cost.

Spreads are now measured (live tape, 2026-10-05 16:21–17:21 UTC, 44,369 quotes; medians;
a round trip pays the spread once):

| symbol | spread bp | commission bp | **total round trip** |
|---|---|---|---|
| NAS100 | 0.26 | 0.00 | **0.26** |
| US500 | 0.39 | 0.00 | **0.39** |
| XAUUSD | 0.19 | 0.24 | **0.43** |
| EURUSD | 0.00 | 0.44 | **0.44** |
| GBPUSD | 0.00 | 0.45 | **0.45** |
| USDX | 1.86 | 0.00 | **1.86** |
| XTIUSD | 3.23 | 0.00 | **3.23** |
| XAGUSD | 3.28 | 0.24 | **3.52** |

The cheap instruments are the FX majors, gold and the two equity indices. The four
"zero-commission" symbols are *not* the cheap ones — oil, silver and USDX are the expensive
ones (the broker earns in the spread there). Full distribution:
`artifacts/tape_summary.json`.

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

With measured costs the top rows split cleanly: the XAGUSD and USDX candidates (2–3 bp gross
edges) are **cost-dead** — silver alone pays 3.5 bp round trip — while the XAUUSD (+1.8, +0.8),
NAS100 (+1.7, +1.3) and EURUSD (+0.1–0.4) rows clear their measured costs and die on the
family null alone.

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

### 8. Expanded universe (28 symbols): first family above its null

28 instruments (8 original + 11 softs + 9 energy/metals), 1,932 candidates, 200 worlds:
family max |t| = **6.80 vs null p95 3.99** (z = 10.8) — the first family in the project to
beat its own null. Two instruments drove it:

- **XPBUSD (lead)**: a smooth London-morning ramp, entirely inside 06:00–08:00 UTC
  (06→08 = +8.33 bp, t = 6.80; exact decomposition 04→08 = 04→06 + 06→08; halves +4.2/+5.9;
  every year positive). Measured spread **13.05–13.09 bp** (taped at 16:00 UTC; live check at
  06:00 UTC inside the ramp: 13.05) and the feed ticks ~1/minute. Net = **−4.7 bp/trade.
  Dead on cost**, in both directions, at the actual trading hours.
- **XCUUSD (copper)**: an early-Asia fade localized to 03:00→05:00 UTC: short = **+4.46 bp/
  trade**, t = −6.51, ~1 trade/day, halves −5.8/−3.3, every year negative. Measured spread
  **2.50 bp in two sessions** (15:53–16:53 UTC and 06:45 UTC; p90 2.67). Net ≈ **+2.0 bp/trade**
  (~+5%/yr on notional), no rollover crossing. **First candidate to pass both the null and the
  measured cost gate**; registered as `forward_test_v1` (remaining gates: spread sample at
  03:00–05:00 UTC; demo execution/fill check).

Also measured: the expanded universe's cost wall — copper 2.5 and Brent 2.9 bp are the cheap
instruments; softs 6–20 bp; natgas 23, platinum 35, palladium 59 bp. Edges must be that large
to survive there.

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

## Expanding the universe (softs, energy, metals)

Next candidates: soft commodities (coffee, sugar, cocoa, grains, cotton) and further
energy/metals (natural gas, copper, platinum/palladium). Priors that justify pre-registered
families there: physical seasonality, thin books and slower arbitrage, roll/adjustment
mechanics, and session-bounded hours (gap structure). The trade-off runs in reverse too:
wider spreads and shorter data depth — both measured before anything is believed.

Runbook. Never guess symbol names — `symbols` first, after any tape stops so you don't hold
two live connections:

```powershell
# from ../ctrader
python -m ctbridge.bridge --config config.json symbols --filter XCU   # locate copper etc.
python -m ctbridge.bridge --config config.json backfill --symbols <names> --days 2200

# from holedigger
python -m src.audit  --symbols <names>       # depth / resolution / outages per symbol
python -m src.ingest --symbols <names>       # parquet + manifest + read-back parity
python -m src.window_sweep --symbols <names> --worlds 200
python -m src.pair_sweep --pairs "XCUUSD/XNGUSD,..." --worlds 200

# costs: zero-commission instruments are spread-only — tape them before ranking
python -m ctbridge.tape --config config.json --symbols <names> --minutes 60
python -m scripts.tape_analysis              # per-symbol / per-hour spread distributions
```

Multiplicity: every added symbol grows each family's candidate count and therefore its
null-max bar (~sqrt(2 ln N)). That is correct behaviour, and more symbols also mean stronger
cross-symbol replication evidence. Add commissions and lot sizes to `src/costs.py` when the
broker's figures are known; unknown costs fall back to gross break-even. Exotic CFDs often
have much shorter M1 depth than majors — the audit's `uniform_start` will say how short.

## MT5 data source (tick-level, real spreads)

`src/mt5.py` is a dependency-free client for the MetaTrader 5 MCP server
(`HOLEDIGGER_MT5_URL`, key in `HOLEDIGGER_MT5_KEY` — never commit it). It upgrades the
data layer in three ways over the cTrader feed: **per-bar spread** (ask-bid at every
minute), **true mid prices** (cTrader bars were broker-side), and **tick counts** as a
liquidity measure. Same account/broker, same symbol universe.

```
python -m src.mt5_ingest --symbols EURUSD XAUUSD ... --days 350 --budget 250   # resumable
python -m scripts.mt5_audit --symbols ... --hours 3 5 6                        # market quality
python -m src.window_sweep --source mt5 --symbols ...                          # per-trade spread costing
```

The ingest is **day-chunked, state-resumable and budget-bounded**: it stops cleanly
before a host timeout and continues on the next invocation (`data/mt5_state_<SYM>.json`
holds progress, `data/mt5_manifest.json` the terminal clock). Tick responses are split
adaptively so a 100k-row limit never silently truncates — NAS100 alone is ~460k ticks/day.

Clock: terminal-local timestamps are **UTC+3** (measured by cross-source shift scan against
the cTrader UTC-labelled parquet: correlation peaks at -3h). MT5 hour 06:00 = UTC 03:00 —
which is exactly the copper fade window, so the sweep's hour labels carry over with that
offset in mind. Cross-source agreement was validated: 63% identical 5-min closes, 4 bp rms
difference (same price stream, different tick aggregation; no systematic bid/mid offset).

Spread costing (`--source mt5`) prices each trade with the spread *at its own entry and
exit bars* (half a spread each side), instead of a per-symbol session median. Confirmed
on live ticks: EURUSD bid==ask on 99.8% of ticks (0.00 bp), XAUUSD ~0.17 bp stable across
all hours — so the majors' cost is essentially commission-only, while the exotics'
spreads (copper 2.5, silver 5.0, oil 3.1, palladium 59 bp) are now measured historically
instead of taped once.

### 9. MT5 re-run: same experiments, tick-derived data

Ten instruments (350 days each; NAS100 59), 690 candidates, 200 permutation worlds
(`--source mt5`). Family max |t| = **3.82 vs null p95 3.87** (94.0th percentile, z = 1.77):
no survivor. The candidate that carried the top of the family is the same one on both runs —
EURUSD 00:00–01:00 *terminal* (21:00–22:00 UTC, the rollover hour), gross +0.87 bp, t = 3.82,
n = 248, halves +4.21/+1.24 — and it clears on the six-symbol family (p95 3.66) but not on the
ten-symbol one (p95 3.87). Adding instruments raised the floor past the candidate, which is
the multiplicity discipline working exactly as designed. Its measured cost at that window:
spread 0.54 bp + commission 0.44 bp → **net −0.12 bp. Dead** on cost as well.

The two cost questions left open by the cTrader universe are now answered with 350 days of
tick history, and both go the wrong way:

- **Copper fade** (terminal 06–08h = UTC 03–05h): gross short edge 2.6 bp vs per-trade
  spread cost **7.4 bp** → net **−4.8 bp/trade**. Copper's spread is 5.0 bp in Asia hours
  against 2.7 bp in London — the edge sits exactly where the spread is widest.
- **Lead ramp** (terminal 09–11h = UTC 06–08h): gross edge ±2 bp vs per-trade spread cost
  **17–28 bp** → net **−15 to −26 bp/trade**.

Two generalizable findings. First, on this broker the thin-session hours are both where
session-window edges concentrate *and* where spreads are widest, so per-trade costing is
not a refinement, it is the whole question. Second, the lead ramp — the project's strongest
signal at t = 6.8 on six years of cTrader bars — does not replicate on ten months of MT5
mid ticks (opposite sign, t ≈ −1). Basis or regime, unresolved; uninvestable either way at
those spreads. Cost-clearing candidates do exist (XAUUSD 13+2h nets +5.2 bp of spread *and*
commission, XTIUSD 17+2h +16.5 bp, EURUSD 18+1h +1.0 bp) but all sit below the family null
(gross t = 1.4–2.6 vs p95 3.87), which is the correct verdict — and which is the argument
for mechanism-first families: at N = 1 the two-sided bar is ~2.0, so a pre-registered single
hypothesis on a candidate like these would be decidable; at N = 690 it is not.

## Equities universe (MT5 Technology sector, 105 names)

The broker's **Technology** sector (105 instruments) exists only on MT5 — the cTrader feed has
no equities. Two things had to change to make them usable:

1. **Clock**: MT5 stamps terminal-local (UTC+3). The ingest now subtracts the offset and stores
   UTC, so one window grid covers both asset classes and the equity session (13:30–20:00 UTC)
   actually falls inside the default 00–20h entry range. Verified on the pilot: Apple's bars
   start exactly 13:30 UTC.
2. **Quality measure**: equities carry overnight gaps (earnings, dividends, regime volatility) —
   the audit now reports a gap profile, because that is the noise floor any session-window edge
   has to clear. In the current regime the median overnight gap is ~1% (p95 ~2.7%), which is
   larger than most session-window effects, so candidates must clear it by construction.

Costs from the pilot: median spread 4.0 bp for the large caps (Apple 4.1, NVIDIA 3.7), 9 bp for
Robinhood, 17 bp for Atlassian — declining through the session. Commissions are not yet in the
model for equities (spread dominates; needs the user's per-share numbers).

The universe is one correlated factor, so 105 names ≈ one bet repeated: the permutation null
remains valid and conservative (per-symbol permutation gives independence the real cross-section
does not have), but a "survivor" in 6,555 candidates means little, and the mechanism-first
discipline matters more here than anywhere.

## Equities run (105 MT5 Technology names)

Full universe, 200-day window, tick-derived mid bars. 5.4M bars; history varies by name
(most have ~198 days; Apple's ticks start only 2026-06-29; Palantir/SpaceX ~145).

**Cost map** (`scripts/mt5_audit --mt5-done`): every name's spread declines through the
session — e.g. Accenture 14.7 bp at 14:00 UTC → 7.8 bp at 19:00 — and the universe splits
~10× in cost, from Berkshire 3.2 / Visa 3.4 / Alphabet-A 3.5 bp to NIO 105 / Snapchat 90 /
Lucid 92 bp at the wide end. Overnight gaps median 0.5–1.8% are the noise floor.

**The finding** (sector-level test, calibrated inside the same permutation null): the
18:00→20:00 UTC window (13:00–15:00 ET) drifts down sector-wide — z = **−6.07 vs null p95
2.64, 100th percentile, n = 105**. The same shape appears in the overlapping 16:00–20:00
window (z = −5.53). This is the first candidate in the project to clear a family null
decisively.

**The cost verdict** (`scripts/sector_test.py` — per-day, per-name, spread measured at the
trade's own bars, then an equal-weight portfolio):

| basket | gross bp/day | spread cost | net bp/day | net t |
|---|---|---|---|---|
| all 105 | +3.35 | 11.8 | −8.46 | −3.74 |
| cheapest 24 (ex-ante, by measured spread) | +3.15 | 4.66 | −1.18 | −0.70 |
| cheap-24, first half / second half | +4.0 / +3.7 | — | −0.36 / −0.66 | ≈ −0.2 |

So the effect is real and stable across halves, and the cost of executing it consumes it
almost exactly: the cheap basket nets to zero, the broad basket loses. Only a post-hoc
handful clear (NVIDIA +8.7, Micron +7.3, Booking +5.5 bp/trade net) — and selecting names
after seeing the P&L is not evidence, so that stays a lead rather than a candidate.

## Clock verification and the FX/metals re-run

Clock: the cTrader feed's UTC labels are **true UTC**, established two independent ways —
MT5 equity bars land at exactly 13:30 UTC after subtracting the 3h terminal offset (the US
cash open is an external reference), and cross-source the MT5 series aligns to the cTrader
series at exactly a −3h shift. The ingest now normalizes MT5 to UTC, so both sources
share one clock and every window label means what it says.

Sector-level re-run on the FX/metals universe (10 instruments, UTC clock, 200 worlds):

| window | sector z | null p95 | percentile |
|---|---|---|---|
| 17:00 +1h | −3.22 | 2.13 | 98.5% |
| 13:00 +1h | −2.97 | 2.21 | 99.0% |
| 02:00 +1h | −2.87 | 2.00 | 99.0% |
| 12:00 +2h | −2.39 | 2.09 | 98.0% |

Priced as portfolios (`scripts/sector_test.py`):

- 17+1h on the five cheapest (EURUSD, XAU, XAG, XCU, XTI): gross **+1.46 bp/day**, cost 4.3 →
  net **−2.26 bp/day**; halves +2.14/+0.69 gross (decaying), net −2.3/−2.3.
- **Copper 03:00+2h UTC** (the cTrader finding): gross **+2.64 bp**, decaying across halves
  (+4.11 → +1.34), cost **7.39** → net **−4.75 bp/trade**, both halves strongly negative.
  So the clock was not the failure — the spread was, and the gross edge itself was fading.

Verdict across all three universes: equities showed the largest real effect (z = −6.07) and it
was cost-consumed; FX/metals show the same shape at a third the magnitude and are also
cost-consumed. The pattern is consistent: microstructure effects at these horizons are real,
small, and priced.

## Event-driven study (calendar-triggered)

Design note — the null that matters here: comparing event windows against the **same clock
time on days with no event for that currency**. The intraday drift measured earlier is
therefore inside the null, and only event-specific behaviour counts as signal.

First result (EURUSD, high-impact USD events, +30 min, ~9 months):

| | mean | t |
|---|---|---|
| event windows (n=292) | +0.16 bp | +0.36 |
| clock-time controls (n=284) | −1.84 bp | −5.13 |
| **incremental** | **+2.0 bp** | — |

Reads as: the event hour is ~2 bp *less negative* than the same hour on quiet days — a real
incremental effect sitting on top of a strongly negative baseline hour. +2.0 bp is near
EURUSD's cost (zero spread + commission), so this is a lead, not a candidate: the standard
barriers apply — more symbols, cross-instrument replication, and the measured spread.

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
4. **Every survivor needs a mechanism story and a measured cost model.** The cost model
   exists (8/8 symbols; break-even spread reported per candidate). No survivor has needed to
   pass it yet — three families, zero survivors, and that is the honest state.

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
