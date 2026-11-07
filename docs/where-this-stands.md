# Where the project stands, and why it stopped here

Written after the third round of external review. The prior rounds found real defects in the
inference layer; this one found that the two surviving results did not survive either. With
both withdrawn, the honest state is **zero survivors across eight families**, and the numbers
say that is a property of the search space rather than a failure of the machinery.

## The arithmetic that settles it

| quantity | measured value |
|---|---|
| cross-sectional ranking, gross per leg | **+0.453 bp** |
| equity round-trip spread, ~1.4 bp | the cost of taking that leg |
| conditional family, median drift difference | +3.1 to +7.9 bp per leg (and it did not clear a calibrated null) |
| effect size 137 days can certify at t=2 | **3.4 bp** (per-day sd ≈ 20 bp) |
| effect size 250 days can certify at t=2 | **2.5 bp** |

So the search space is looking for a ≥ 2.5 bp edge to *certify*, while the effects it can find
are ~0.45 bp per leg and the cost of taking one leg is ~1.4 bp. Every family in this repo has
finished in the same place: an effect that is real, replicable, and smaller than the toll.

Public price features at 5-minute-and-longer horizons on liquid CFDs are the tightest market
in existence — that is where the arbitrage capital already is, and the costs already include
it. The repo's data is 140–250 days per instrument, which cannot certify the small effects
that could survive those costs, and no amount of inference repairs the power gap.

## What the machinery did prove

This is the part worth keeping, and it is not a null result.

The inference layer caught, in order:

1. **A mean estimator reporting a fake +26 bp** on fat-tailed data (dislocation study).
2. **A pooled z calibrated against the wrong correlation structure** — the original sector
   test, ~100% false positives on pure noise.
3. **A DST-blind clock**, which had ~32% of bars an hour early and silently invalidated every
   hour-of-day claim.
4. **A control sample of 284 draws from 9 distinct days**, which manufactured the event
   study's baseline.
5. **A null that could not destroy its own hypothesis** (whole-day permutation against a
   fixed clock-slot effect).
6. **A conditional null shuffling correlated observations** — 70% false positives, the same
   defect as (2) in a new place.
7. **A ranking null that destroyed beta structure** — 42% false positives.
8. **Direction chosen in-sample**, and a markout whose horizon was not in minutes.

Every one was found by a test that asked the null a question with a known answer. That is the
transferable asset here: `tests/test_null_calibration.py` is a pattern, not a one-off, and any
future search in this repo — or in a different market where the costs are not this tight —
should carry the same discipline.

## What would actually move the needle

Not more families of the same shape. In order of expected value:

1. **A venue where costs are not the binding constraint.** The user trades NQ and GC futures
   through Rithmic; CME spread and commission structure differs from Fusion CFDs by more than
   the entire measured effect. Re-costing the same candidates on the real venue is a data
   problem, not a research one, and it is the single highest-value thing available.
2. **More history — not available on this venue, checked.** The MT5 terminal advertises
   ``data_available_from: 2024-01-02``, but the same terminal returns **0 ticks for every
   queried date before 2025-11-01** (bisected 2024-01-02, 2024-06-01, 2025-01-01,
   2025-06-01, 2025-10-01, 2025-11-01) and only 4 daily bars for all of 2024 H1. The
   advertised range is not retrievable, so the 350-day panel is the venue's real limit and
   the power gap cannot be closed by pulling more from this broker. Power scales with √days,
   so a venue serving 4 years would quarter the certifiable effect size — that requires a
   different data source, not a longer pull.
3. **A different search space.** Tick-level microstructure (order-flow imbalance, queue
   dynamics, passive-fill economics) has effects measured in fractions of a basis point but
   trades at sizes where the cost is also fractions of a basis point. This repo can *reach*
   that data (`src/tick_ingest` retains it) and its honest finding so far is that a passive
   fill loses at every horizon — which is a real answer to a real question, not a null one.
4. **Model the predictable.** `src/cost_model` already predicts spread with 90-95% accuracy
   out-of-sample. Cost is the one target in this project with dense labels and real signal.

## The rule this repo now encodes

> A null that has not been checked on data with a known zero answer is not a null, and a
> result that has not survived a time-split is not a result.

Eight families, zero survivors, three withdrawn findings, and a testing discipline that found
every defect. That is the honest deliverable.
