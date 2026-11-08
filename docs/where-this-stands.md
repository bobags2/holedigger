# Where the project stands, and why it stopped here

Written after the third round of external review. The prior rounds found real defects in the
inference layer; this one found that the two surviving results did not survive either. With
both withdrawn, the honest state is **zero survivors across eight families**, and the numbers
say that is a property of the search space rather than a failure of the machinery.

## The arithmetic that settles it

Written after the third round of external review. The prior rounds found real defects in the
inference layer; this one found that the two surviving results did not survive either. With
both withdrawn, the honest state was **zero survivors across eight families**.

That conclusion has since been revised in one place, and the revision is recorded here rather
than left to a diff: the fourth review found that the ranking book's **cost was drawn from the
wrong session** (42.98 bp against a measured median equity entry+exit of ~10 bp, a 3.6×
overstatement). Re-costed on the session the book actually holds over, the book's net is
**+6.99 bp/session at percentile 1.000 against its own null**, and +4.23 bp beta-neutralised.
So this is the first candidate in the repo's history to clear its own null, and the "every
family finishes below the toll" claim above is now **too strong**.

**Which of those two states is right, and how it will be settled.** The three numbers below
are the honest current position:

| quantity | measured value | status |
|---|---|---|
| ranking book, net after measured cost | **+6.99 bp/session** | clears its own null (4,000 worlds) |
| ranking book, beta-neutralised net | **+4.23 bp/session** | survives the transform the reviewer asked for |
| ranking book, gross t | **+1.92** | **weakly certified — the gross is the weak link, not the cost** |
| ranking book, data span | **2026 only** | one calendar year; the +6.99 bp is not a multi-year claim |
| equity round-trip spread, measured | ~10 bp median entry+exit | the toll the book pays per leg |

The two open checks that decide whether +6.99 bp is real, both from the same review and both
unfinished: (a) whether the gross is concentrated in the opening bar — the book trades the
first and last bars of the session, where equity CFD spreads are widest, so a mid-session
entry/exit must be priced before anyone acts on this; and (b) a second calendar year. Until
those land, treat +6.99 bp as **a candidate that cleared its null, not a validated edge**.

The rest of the arithmetic still holds and still explains why everything *else* finished below
the toll:

| quantity | measured value |
|---|---|
| cross-sectional ranking, gross per leg | +0.453 bp (the ranking correlation, not the book) |
| equity round-trip spread, median entry+exit | ~10 bp |
| conditional family, median drift difference | +3.1 to +7.9 bp per leg (did not clear a calibrated null) |
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
9. **A "corrected" null that was not a p-value** — the real statistic was computed on RAW
   returns while its null residualised them, so the percentile compared two different
   functions. Measured at **100% false positives** at market AR(1)=0.3; 5.0% after applying
   one shared ``beta_neutralise`` to both sides. The calibration test asserting that null
   contained the *same asymmetry*, which is why it passed at 0% while production failed.
10. **A cost drawn from the wrong session** — entry/exit spreads read from the previous
    session's rows *and* from per-symbol arrays indexed with the union-day array, which
    overstates the toll 3.6× and buried the one result that was positive.

Every one was found by a test that asked the null a question with a known answer. That is the
transferable asset here: `tests/test_null_calibration.py` is a pattern, not a one-off, and any
future search in this repo — or in a different market where the costs are not this tight —
should carry the same discipline.

## What would actually move the needle

Not more families of the same shape. In order of expected value:

1. ~~**A venue where costs are not the binding constraint.**~~ **Moot — CFD-only, verified.**
   This item used to recommend re-costing the candidates on NQ/GC futures through Rithmic.
   The account is CFD-only (confirmed 2026-11-07), so that is not available and the note is
   left here only so it is not re-proposed. What survives the correction is the *shape* of
   the question, and `src/venue_cost.py` is the answer to it: the cost ledger now prices two
   venues (MT5 panels and the cTrader live tape), and four pairs that were dead or
   unpriceable on MT5 clear on cTrader. Pick the cheaper venue per instrument.
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

## The fourth review's outcome: no alpha, and a risk framework instead

The fourth external review found that the cTrader position P&L was computed from the **first
closing deal only** (`gp.iloc[0]`), on a deals file that also contained 6,000 duplicate rows.
Corrected, the account's net is **−602,934** across 3,473 positions, every instrument that
matters is negative, and the balance path ran 112,508 → 54 with a peak of 246,253. That removed
the last positive-looking result in the repo and confirmed what eight families of price features
had already said: there is no directional edge here.

What it left is the most robust finding in the project, and it is not about direction.
**The losses are process.** The worst ten trades are 279% of NAS100's net, 153% of silver's, 96%
of gold's, and 51% of the entire oil loss is ten trades out of 734. A cap is enforceable at
entry time with no forecast, so `src/risk_rules.py` builds one.

**The rule that looked like it worked — and is withdrawn.** Cap each position at 3x that
instrument's measured venue toll (`src/venue_cost`):

| | net | days positive | worst day |
|---|---|---|---|
| as measured | −602,934 | 13.1% | −103,332 |
| ~~capped at 3× toll~~ | ~~+1,210,466~~ | ~~66.7%~~ | ~~−5,794~~ |

**Why it is withdrawn, and it is not a subtlety.** That evaluation floored each position's FINAL
P&L at the cap. A stop fires when the price path TOUCHES the level, so the floor kept every
position that dipped below the cap and later recovered — precisely the positions a real stop
would have cut. The property test settles it: on a synthetic zero-skill book with negative
expectancy by construction, the floor scores **+2,120** where the path-aware evaluation scores
**−798**. A rule that turns a known-no-skill book positive is measuring its own arithmetic, not
removing loss, and the day-clustered null could not catch it because the artifact is in how the
number is built.

The level was also unsound on its own terms: on the real book the 0.44 bp gold cap is **touched
on 59.9% of positions**, which makes it a toll multiplier, not a stop. Stop distance has to come
from volatility over the holding horizon and from account risk.

**What replaced it — SIZE, which is the actual mechanism of ruin here.** The framework never
compared notional with equity:

| | measured |
|---|---|
| mean notional vs median balance | 4,043,835 / 6,272 = **645× leverage** |
| median per-position loss, % of balance | **7.0%** |
| p95 per-position loss, % of balance | **233%** |
| R1's spec stop (37.9 bp), % of balance per position | **244%** |
| size at 0.5% risk / 10 bp stop / $1,125 equity | **$5,625 notional** |
| size actually traded | **4,043,835 — 717× oversized** |

The account drawdown stop, run on the event-ordered equity path the day limit should have used,
fires: **101% drawdown, trough −1,297** (trigger set at 25%). That is what explains the account,
and it needs no alpha and no cap — it needs size tied to equity. `src/risk_rules.leverage_profile`,
`.size_for` and `.kill_switch` are built and tested for exactly that.

**What this is not.** A counterfactual on history in which the rules did not exist. The forward
test is the point: freeze `artifacts/risk_spec.json`, run it on demo for 20 sessions, and judge
on the shape above rather than the total. A max-hold rule is deliberately absent — these trades
had no max hold, so evaluating one conditions on the outcome.
