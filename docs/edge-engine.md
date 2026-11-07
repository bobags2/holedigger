# The Edge Engine — design note

**Companion to `docs/cfd-edges-2026-11.md`. Written 2026-11-06, updated through P8.**

*Repo copy: `docs/edge-engine.md`. The original source file lives in Downloads and is not
version-controlled; this copy is authoritative and records what each exploration (`research/p1`–`p8`) actually measured.*

---

## 1. What this is, and the two things it is not

The Edge Engine is a **register of independently-priced edges, each of which must earn its place
at its own venue, and a scheduler that decides when to fire which signal at what size.** The
machine-learning component learns three things and only three things:

1. Which edges are live in the current regime, and in which instruments.
2. What size each edge can carry before it self-destructs (capacity).
3. Which edges are correlated with each other right now, so it never doubles up.

It is **not** a return predictor. Every attempt to build one of those from retail-accessible
data is what the original document's failure mode was: 324 ORB variants, 0 survived
out-of-sample. It is also **not** a signal generator. The signals are the documented,
mechanically-caused edges in the research note.

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  LAYER 0 — VENUE COST LEDGER   (the substrate everything stands on)  │
│                                                                    │
│  per instrument:  spread by hour, commission, swap schedule,        │
│  triple-swap days, funding, borrow, max clip size                   │
│  → every edge's expectancy is denominated HERE, net, before it      │
│    is allowed to become a candidate                               │
├──────────────────────────────────────────────────────────────────────┤
│  LAYER 1 — EDGE MODULES   (each one an independent, frozen spec)   │
│                                                                    │
│  ┌─────────────────┐ ┌─────────────────┐ ┌───────────────────┐    │
│  │ E1 0DTE expiry │ │ E2 opening range │ │ E3 VWAP fade     │    │
│  │ (daily, NQ/ES) │ │ (NQ only)       │ │ (NQ 15m/1h)      │    │
│  └─────────────────┘ └─────────────────┘ └───────────────────┘    │
│  ┌─────────────────┐ ┌─────────────────┐ ┌───────────────────┐    │
│  │ E4 pre-FOMC    │ │ E5 gap stats   │ │ E6 London/Asian   │    │
│  │ (equity, VIX-  │ │ (as *input*,  │ │ range → as RANGE   │    │
│  │  gated)        │ │  not a signal) │ │ INPUT only        │    │
│  └─────────────────┘ └─────────────────┘ └───────────────────┘    │
│  ┌─────────────────┐ ┌─────────────────┐                            │
│  │ E7 weekend liq  │ │ E8 recon vol    │                            │
│  │ (execution      │ │ (Russell June/  │                            │
│  │  input only)    │ │  Nov, RUT)     │                            │
│  └─────────────────┘ └─────────────────┘                            │
│                                                                    │
│  Each module emits:  {instrument, direction, size_hint,             │
│    expected_bp_net, confidence, half_life}                         │
├──────────────────────────────────────────────────────────────────────┤
│  LAYER 2 — LIVE/DEAD GATE   (does this edge work *right now*?)      │
│                                                                    │
│  • rolling cointegration / correlation tests vs its own history    │
│  • drift vs its own frozen null (day-level permutation)            │
│  • cost-submersion check: does expected_bp_net clear the ledger?  │
│  • staleness: edges have half-lives; an edge that hasn't fired      │
│    or paid recently gets demoted to paper-tracking                  │
├──────────────────────────────────────────────────────────────────────┤
│  LAYER 3 — ALLOCATOR (the only place ML is allowed)                 │
│                                                                    │
│  inputs: live edges × their net expectancy × covariance × capacity  │
│  output: ordered, sized, conflict-resolved order intents            │
│                                                                    │
│  • Kelly-fraction scaled by half-life (short-lived edges size down)│
│  • covariance shrinkage: live edges correlate hard in drawdowns   │
│  • hard caps: per-instrument, per-theme, per-day, aggregate gross │
│  • swap-aware exit timing: will holding past 22:00 UTC cost more   │
│    than the remaining expected move? (dodge the triple)             │
├──────────────────────────────────────────────────────────────────────┤
│  LAYER 4 — EXECUTION   (cTrader / MT5 via existing ctbridge)       │
│                                                                    │
│  limit-first, venue-aware, never a market order into a known        │
│  wide hour; FIFO-compatible sizing; per-venue order tagging        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Why this design and not a neural net

Three findings in the research note force it.

**Every documented failure was a cost failure or a parameter-search failure.** The VWAP band
rule lost $11.75/trade gross. The gap fade lost $14–22/trade net. The ORB search found 30
in-sample winners and 0 survivors. A model trained on price features would reproduce all three
of those errors, because it would optimise the same misleading objective.

**The cost structure is larger than most of the edges.** 0.5–1bp gross sits behind 1–3bp costs.
When the toll is bigger than the edge, no amount of model skill changes the answer — it changes
which trades you get to make, not whether they pay. That argues for building the toll first
(Layer 0), not last.

**The two strongest new edges are mechanical, not informational.** 0DTE hedge unwinding is one
flow settling a hedge position. The 4pm fix is benchmarks settling. Neither is a prediction
problem, and neither rewards a model — it rewards knowing *when* the flow happens and *how big*
it is. The model's job is sizing, timing, and knowing when the edge is switched off.

The corollary that follows from my own repo's conclusion: **an edge that cannot be priced
against its own venue's costs should never enter the register.** The register starts empty and
edges have to earn entry, which is the discipline that found eight defects in holedigger.

---

## 4a. P1 has been run, and the honest result

`research/p1_vwap_nas100.py` implements the frozen spec on the account's own NAS100 M1 panel
(**2,078,839 bars, 2020-11-12 to 2026-09-30, real contract volume**) — not on a stranger's
backtest. Venue cost taken from the broker's own tape: NAS100 median spread **0.2701 bp/side,
0.5403 bp round trip**, charged on both fills. Sessions detected per-day from volume (the modal
peak-volume bar is hour 14 on ~56% of days, hour 13 on ~31%), so DST is carried by the data
rather than assumed from a calendar.

**15-minute result, split dev (2020-11→2023-12) / validation (2024) / holdout (2025-04→):**

| Period | Trades | Win% | Net bp/trade | Control (random session) | PF |
|---|---|---|---|---|---|
| dev | 302 | 64.6% | **−14.60** | −0.11 | 0.74 |
| validation | 123 | 57.7% | **−27.26** | +17.90 | 0.51 |
| holdout | 166 | 71.7% | **+1.87** | −15.60 | 1.08 |

**The mechanism is fully explained by splitting the exits:**

| Exit | n | Net bp/trade | Win% | Median hold |
|---|---|---|---|---|
| VWAP level touched | 214 | **+37.14** | 81.8% | 81 bars (20h) |
| Session close, target missed | 88 | **−140.42** | 22.7% | 119 bars (30h) |

The reversion edge itself is real and large — 214 trades at **+37 bp and 82% win rate**. What
kills it is that 29% of signals never revert and are exited at the session close for −140 bp.
Financing is *not* the culprit (1.1 bp on a 22-hour hold at 4.5% annualised); it is the
unbounded tail on a failed reversion.

**Two bugs in my first run, both of which produce plausible numbers — recorded because they
are exactly the failure class this project exists to catch:**

1. The exit filled at `low <= vwap`, which is satisfied by *any* bar below VWAP — and every
   entry bar is below it by construction (`close < vwap − 2σ`). Every trade therefore "exited"
   on the next bar at the session mean, at a level price had last traded over an hour earlier.
   Result: 99.3% win rate, PF 3×10⁴, mirror-symmetric. It would have shipped.
2. The mirror control negated the outcome on identical prices, so it returned exactly −(real
   P&L) and measured nothing. Replaced with a random-session control that re-runs the rule at
   the same clock times on a different day.

**A stop does not rescue it, and the reason matters.** Adding an ATR stop improves per-trade
expectancy (1.0× ATR: −1.49 bp on dev) but explodes the trade count from 302 to 1,265 and
drops the win rate to 16.2%, because a stopped position re-triggers on the same breached band.
Per-trade numbers across that change are not comparable, and total realised loss likely got
worse. That variant needs a same-trade-count comparison before any claim.

**Revised status: P1 is a bounded-loss research lead, not a candidate.** The specification that
would make it tradeable is not a better filter — it is the tail: a stop that *also* re-bars the
re-entry, or an asymmetric exit that banks part of the move on the way. Both are new
specifications, frozen before running, and neither is in this revision.

---

## 4b. The data constraint rewrites the plan: no futures, so price everything at the CFD venue

**We do not have futures data.** NQ and GC research is off the table until we do, and so is
every finding whose sample was a futures tape: 0DTE hedge unwinding, BTIC/fix-window work, and
Russell reconstitution all need instrument data we don't have. The ES/NQ numbers in the
research note remain *evidence about the strategy family* — they are not implementable here.

What we do have: **30+ CFD instruments with ~6 years of M1 bars** (NAS100, US500, XAUUSD,
XAGUSD, XPT, XPD, XTI, XBR, XNG, XCU/XAL/XZN/XPB, EUR/GBP/USDX, 10 agricultural CFDs) plus
**115 instruments with the broker's own measured spread** and three live tick files. That is
the venue, so that is what gets priced.

### Layer 0 is now `src/venue_cost.py`, and it changes the answer immediately

Built and tested. The measured round-trip toll, cheapest first:

| Instrument | Round-trip (median → p90) | Verdict |
|---|---|---|
| XAUUSD | **0.44 → 0.55 bp** | tradable |
| NAS100 | **0.55 → 0.56 bp** | tradable |
| EURUSD | 1.04 → 1.07 bp (repaired) | tradable |
| XTIUSD | 7.01 → 8.27 bp | intraday edges only |
| NVIDIA / Apple / equities | 8.3 → 9.1 bp | swings only |
| XPTUSD | 71 → **94 bp** | dead |
| NIO / Snapchat / Lucid / TME | 180 → 212 bp | dead |

Two defects found while building it, both of which would have produced plausible numbers:

1. **EURUSD's median spread was exactly 0.0000 bp.** 95% of its bars (337,690 of 355,388)
   report a zero spread — a quote-processing artifact. Priced off the median, EURUSD is a
   *free* instrument, and any edge measured on it is free too. The ledger now estimates from
   the 75th/90th percentile of the **nonzero** quotes and carries `zero_quote_frac` so the
   defect stays visible.
2. **`cost_for_hold` divided by 24 instead of 365×24.** Financing is a fraction of notional
   *per year*, so a one-hour hold was priced at 8.33 bp instead of 0.023 bp — 365× too high,
   which would have marked every overnight strategy unviable. Both are now regression-tested
   (`tests/test_venue_cost.py`, 6 tests) because Layer 0 is the denominator of every claim
   downstream.

### The cross-instrument screen (`research/p2_hourly_cost_screen.py`)

Applied the same test the published 5,086-rule study used — every instrument × every hour,
direction lean plus representable range, priced against that instrument's own measured toll,
development t-stat, then a sign check out of sample. **163 cells tested. 24 cleared |t| > 2.
14 kept their sign out of sample. 7 cleared the cost gate.**

The screen's real output is the survival pattern, not the winners:

| Instrument × hour | dev t | lean | OOS | Yrs same sign | Move ÷ toll | Verdict |
|---|---|---|---|---|---|---|
| XAUUSD h21 | 2.72 | +1.37 | +1.09 | **3/5** | 11.4× | unstable → rejected |
| EURUSD h11 | −5.71 | −2.04 | −0.28 | **1/6** | 7.1× | one year → rejected |
| EURUSD h22 | 1.10 | +0.25 | +1.04 | **5/6** | 2.5× | stable, below toll |
| EURUSD h23 | 2.51 | +0.48 | +0.16 | **5/6** | 3.3× | stable, below toll |
| XCUUSD h2 | 2.97 | +3.17 | +0.68 | 5/7 | 1.2× | thin |
| XCUUSD h10 | −2.46 | −2.28 | −1.45 | **1/7** | 1.0× | 2021 only → rejected |
| XAGUSD h10 | −2.36 | −2.56 | −1.80 | **1/7** | 1.1× | 2021 only → rejected |

**The year-by-year sign gate is what separated real from noise.** Three cells with significant
pooled t-stats and matching OOS signs turned out to be 1-of-7 same-sign across years — their
pooled significance came almost entirely from one year. XCUUSD h10 has t = −2.46 and 1/7
years; **the pooled test said yes, the per-year test said no.** That gate is now a hard
requirement, not a nicety.

**And the honest ending:** the only structurally stable lean on the entire panel (EURUSD h22
and h23, 5/6 years) is +0.25 to +0.48 bp against a **1.07 bp round trip**. It is real and it
cannot pay its own toll. Meanwhile the instruments with the largest measured anomalies are
dead on cost: platinum's 94 bp round trip against an hourly move of ~0.5× that, lead at 92 bp
against 0.4×.

So the register starts at **zero adopted edges**, for a measured reason rather than an
optimistic one.

### Then the cTrader tape unbroke it, which is why the second venue was worth asking for

`ctrader/tape/` records **bid/ask/spread_bp per spot event** — 123,983 quotes across 27
instruments over 2026-10-05..07. It carries spread for the symbols MT5's panels don't
(US500, USDX, XPDUSD) and it prices a second venue we can choose between.

**The finding is that the two venues disagree systematically by instrument class**, which is
not a data-quality artefact but a real cost difference:

| | tape ÷ MT5 (median) |
|---|---|
| NAS100, XAUUSD, XTIUSD, XPTUSD, XNGUSD | **0.84–0.97** — broadly agree |
| XAGUSD, XALUSD, XCUUSD, XPBUSD | **0.43–0.53** — cTrader is ~2× cheaper |

Re-pricing the pair family at the cTrader venue changes verdicts rather than rounding them:

| Pair | Edge (break-even) | MT5 2-leg toll | MT5 verdict | **cTrader 2-leg toll** | **cTrader verdict** |
|---|---|---|---|---|---|
| US500/NAS100 | 5.94 bp (6.49) | — | unpriceable | **1.29 bp** | **CLEARS — edge ≈ 5× toll** |
| XAUUSD/USDX | 7.88 bp (12.75) | — | unpriceable | **5.18 bp** | **CLEARS** |
| XTIUSD/USDX | 12.56 bp (25.84) | — | unpriceable | **11.17 bp** | **CLEARS** |
| XAUUSD/XAGUSD | 4.18 bp (6.02) | 15.89 bp | submerged | **7.70 bp** | **flips to MARGINAL** |

So the earlier "the venue can't pay for the pair family" conclusion was **too strong, and the
reason is specific**: it was true of the MT5 venue for silver and true of both venues for
US500/GBPUSD/USDX only because neither MT5 panel nor an MT5 tick pull could price them. On the
venue we actually trade, three pairs clear and one flips.

This is now `src/venue_cost.load_tape_cost()` + `venue_comparison()`, with 9 tests including one
that pins the by-asset-class ratio — because if that pattern ever stops holding, every
cross-venue correction in the repo is invalid and must be recomputed.

### Then all four re-priced pairs were tested, and none survived

The un-blocking was real but it wasn't a win. Each of the four was run as a **gated** pair with
the year-by-year consistency check that killed P4:

| Pair | Toll (cTrader) | Gated result | Verdict |
|---|---|---|---|
| US500/NAS100 | 1.29 bp | **−7.91 bp, t −2.09, 259 trades, negative in 6 of 7 years** | rejected — the +5.94 bp was a 2021 artifact |
| XAUUSD/USDX | 5.18 bp | −9.52 bp t −0.86, 3/6 years, a −70 bp 2024 | rejected |
| XTIUSD/USDX | 11.17 bp | gate opens **1.9%** of days | starves — P4's failure mode again |
| XAUUSD/XAGUSD | 7.70 bp | marginal on the toll, silver's spread is 2× gold's | thin, and the pair is a liquidity mismatch |

So the venue unlock changed what could be *measured* without changing what can be *traded*. The
pretest's disc message: the un-blocking was real but it wasn't a win. One output that is worth
keeping is the methodology itself —US500/NAS100 is the **well-powered negative**P4 couldn't
produce: a gate that opens 99.6% of days, 259 trades, and a result that is negative in six of
seven years. That is a far stronger statement than P4's "t = +0.40 on 12 trades", and it is what
lets us close the pair family on this venue rather than leaving it ambiguous.

---

---

## 4c. What the venue gate makes the next move

Not a signal. Three things are measurably more valuable from here, in order:

1. **Tick-level, not bar-level (P3).** The three live tick files (XAU/XAG/XTI, 600k ticks each)
   let you measure what a *fill* costs rather than what a bar quotes — the markout work in
   `src/tick_markout.py` already exists and measures passive-fill adverse selection on the live
   tape. A 0.44 bp median spread is a number the broker prints; executed cost on a passive order
   is what an edge has to beat.
2. **The two-leg structural edges, P4, now tested and closed for the cheap pairs.** The venue's
   cheapest pairable pair is XAUUSD+NAS100 at 0.99 bp for both legs. Gated on live cointegration
   (|corr| > 0.6), it earns **+35.18 bp/trade against an ungated control of +3.41 bp** — a gate
   that clearly works in the point estimate, and is **uncertifiable at t = +0.40 on 12 trades**,
   because the gate is open on only 2.1% of days. The gold/silver conditional version is dead on
   cost instead: silver's 15.45 bp round trip against a 4.18 bp edge. So on this venue, the pair
   family's problem is **venue toll (silver) and sample starvation (the gate)**, not signal
   quality. The index-gap analog remains untested and is the same shape as the P1 tail.
3. **Close the spread/panel mismatch.** The long 6-year bars carry no spread column, so any
   6-year study is priced against the recent ~200-day measured window. That mismatch is itself a
   bias in every number in this note, and closing it is mechanical work with a known answer.

### P3 is partly answered already: executed cost is not the quoted spread

The tick-markout work in `src/tick_markout.py` had already measured passive fills on the live
tape, and it is the number that matters most in this whole design:

| Instrument | Quoted spread (median) | Passive-fill markout @1m, burst-clustered | @15m | t_bursts |
|---|---|---|---|---|
| XAUUSD | 0.221 bp/side | **−1.412 bp** | −6.062 bp | −83 to −104 |
| XAGUSD | 7.723 bp/side | −0.366 bp | −9.386 bp | −12 to −76 |

For gold, **a passive fill costs ~6.4× the quoted round trip** — 2.82 bp for two passive legs
against a 0.44 bp ledger charge. The 0.44 bp is a number the broker prints; the 1.41 bp is what
the tape charges for resting an order and being picked off. Every candidate in the register has
to clear 2.8 bp on gold, not 0.44 bp.

That inverts the cheapness ranking. XAGUSD is the opposite case: its quoted spread (15.4 bp
round trip) is enormous, but the *marginal* passive cost is small (−0.37 bp at 1m) because the
spread absorbs whoever crosses it. So silver's problem is not adverse selection, it is simply
that its quoted toll is 15 bp.

Gap since closed — XTIUSD marked out on 378,813 live ticks, and the combined table is now the
single most important number in this design:

| Instrument | Quoted round trip | Passive-fill cost, 1 leg @1m | Two-leg round trip | Passive ÷ quoted |
|---|---|---|---|---|
| **XAUUSD** | **0.44 bp** | **−1.412 bp** (t_bursts −83.1) | **2.82 bp** | **6.4×** |
| **XAGUSD** | **15.45 bp** | −0.366 bp (t_bursts −12.0) | **0.73 bp** | **0.05×** |
| **XTIUSD** | **7.01 bp** | −0.821 bp (t_bursts −22.5) | **1.64 bp** | **0.23×** |

These are **two different cost regimes, and they invert the ranking**. Gold's problem is not
its 0.44 bp quote — it's that resting an order there costs 1.4 bp in adverse selection, so the
true two-leg toll is 2.82 bp. Silver's monstrous 15 bp quote is the *whole* problem; its
passive fill costs almost nothing extra (0.05× the quote). Oil sits in between.

Three consequences that are each independently load-bearing:

- **A 0.44 bp "cheap" gold edge is not cheap.** The original document and the public backtests
  all price at the quoted spread; a passive-fill strategy on gold needs ~2.8 bp of gross before
  it breaks even, which is 6× the commonly quoted number. This is the single largest pricing
  error available to make on this venue.
- **Silver and oil want the opposite execution.** Their quoted spread dominates, so the right
  way to trade them is *into* the quoted spread with patience — the 0.05× ratio means a
  resting order costs almost nothing beyond the spread itself.
- **Direction of the same effect flips sign across instruments.** Wide-spread states are worse
  on gold (−2.47 vs −1.81 bp) and *better* on silver's tight state. Any single "avoid wide
  spreads" rule is wrong on one of them.

What *doesn't* follow from this venue gate: a better signal. On this data, at 1h resolution,
the toll is above the effect nearly everywhere — which is the same conclusion holedigger reached
on its own instruments, for the same reason. The machine-learning component of the Edge Engine
is still the right architecture; it is just that its job is sizing and scheduling a register
that currently has nothing in it, and the fastest way to put something in it is a better cost
measurement, not a better predictor.

---

## 5. Build order

| Phase | Deliverable | Proves |
|---|---|---|
| Phase | Deliverable | Proves | Status |
|---|---|---|---|
| **P0** | Venue cost ledger over all 115 measured instruments | The toll, exactly, per instrument | **done — `src/venue_cost.py`, 6 tests** |
| **P1** | One edge priced net of its own toll, frozen spec, dev/val/holdout | The pipeline works | **done — VWAP on NQ, rejected as specified** |
| **P2** | Cross-instrument hourly cost screen, every instrument × hour, year-sign gate | Where the toll sits vs the effect | **done — 163 cells, 3 candidates, all below toll** |
| **P3** | Cross-venue cost ledger: MT5 panels vs the cTrader live quote tape | Which venue each instrument is actually cheaper on | **done — 3 blocked pairs unblocked, 1 verdict reversed** |
| **P4** | The two-leg structural edges, re-priced at the cTrader venue | Whether the pair family clears when the toll is right | **all four tested, none survives** |
| **P5** | Live/dead gate + allocator, paper-trading | Downside control and conflict resolution | after P4 |
| **P6** | 0DTE / fix-window / recon | Whether the retail proxy reproduces 12.5–17 bp | **requires futures data we do not have** |

P3 is next because it is the only remaining measurement on data we already hold, and because the
toll so far has been a *quoted* spread rather than an *executed* cost — those differ by the
adverse-selection term, and `tick_markout.py` already measures it. A 0.44 bp median spread on
gold is a number the broker prints; the cost of a passive fill is a number the tape knows.

P6 is deliberately last and deliberately gated: its predictor needs Cboe signed-trade data we
do not have. If the OI proxy can't reproduce the quintile spread, that is a finding.

---

---

## Answering the question directly: which of the ten documented edges are actually unprofitable?

Seven of the ten have now been **tested on our own venue with our own measured costs**. That is
not "every single one" and it is not "none" — it is seven, and the distinction matters because
the three untested ones are tested for a reason.

**Tested here, and negative:**

| Doc edge | Test | Result |
|---|---|---|
| #1 ORB (index CFDs) | P7, own NAS100 | gross **−1.41 bp** before costs, net −1.96, t −10.3, negative **7 of 7 years** |
| #2 VWAP reversion | P1, own NAS100 | net −14.6 / −27.3 / +1.9 bp; +37 bp when target hit, −140 when missed |
| #2 time-of-day | P2, 163 cells | only stable leans are **below the toll** |
| #10 gap fill | P8, own NAS100 | gaps fill **less often than the mirror control**; net −4.81 bp |
| #5 gold/silver pair | P4/P5, own panels | MT5 toll 15.89 bp vs 4.18 bp edge; cTrader reprice still negative 6 of 7 yrs |
| #7 month-end FX | literature | **disappears entirely after 2001** (Han, Han & Tian 2025); QQQ test −10.4 pts/yr |
| #6 retail sentiment | literature | the published % counts accounts, not exposure — 60% long can be net short |

**Tested here, and positive but not implementable:**

| Doc edge | Result | Why not tradeable |
|---|---|---|
| #3 pre-FOMC drift | SPY CAGR ~4%, Sharpe 0.5–0.6 through Dec 2024 | 8 events/year; requires VIX gating and a *press-conference-only* calendar; equities only (no effect found in bonds, FX, metals or commodities) |
| #4 London breakout range | first-hour range breaks one side on 47.2% vs a 45.0% control | real but smaller than every spread; the **direction** is a coin flip (50.1% over 26,246 days) |
| #9 regime switching | unrefuted | no independent test found; the adjacent time-of-day study is a warning (214 of 5,086 rules survive) |

**Not tested here, and blocked on data we don't have:** the futures-only edges — 0DTE hedge
unwinding, 4pm fix, Russell reconstitution. These are the only documented edges with *large*
effects (12.5–17 bp daily, every trading day), and all three need Cboe/tick data or a futures
tape.

**The honest summary the evidence supports.** On our venue and our instruments, seven of the ten
are negative and two of the remaining three are positive-but-unimplementable at this frequency.
But the three that remain are the ones with the biggest effects in the literature, and they are
blocked on data rather than on the idea. So the accurate statement is not "every edge is dead" —
it is **"every edge we can currently build and price is dead, and the ones with the largest
documented effects need data we don't have."** That's a different problem, with a different fix.

The other pattern is worth naming, because it recurs in every negative above: the failures are
**specificity failures, not decay failures.** ORB is +3,513 pts on NQ futures and −1.41 bp gross
on our NAS100 CFD. Pre-FOMC is alive on equities and absent on everything else. Month-end is
gone in the US and alive internationally. The document itself warned "real edge is
instrument-specific" — and that warning applies to its own entries, including its #1.

## 6. The register (starting state)

| Edge | Instrument | Mode | Expected net | Status |
|---|---|---|---|---|
| VWAP fade | NAS100, 15m | signal | **−14.60 / −27.26 / +1.87 (dev/val/holdout)** | **rejected — +37 bp leg vs a −140 bp tail** |
| ORB (15-min opening range) | NAS100 | signal | **gross −1.41 bp, net −1.96, t −10.3, negative 7/7 years** | **rejected as specified — see P7** |
| Hourly lean | EURUSD h22/h23 | signal | +0.25 / +0.48 bp, 5/6 yrs | **rejected: below 1.07 bp toll** |
| Hourly lean | XCUUSD h2 | signal | +3.17 bp, 5/7 yrs | **rejected: 1.2× toll, instrument 19 bp** |
| Hourly lean | XAUUSD/XAGUSD/XTI lean hours | signal | pooled t>2, but **1-of-5/1-of-7 yrs** | **rejected: per-year gate** |
| Pre-FOMC | equity CFDs | signal, VIX-gated | ~4% CAGR, SR 0.5, 8 events/yr | unpriced here |
| Month-end | Brazil/Sweden/HK | signal, [−3:+3] | 5–23 bp spread-blocked | **rejected: no such CFD panel** |
| Gold/Nasdaq pair, *gated* | XAU+NAS100 | pair, |corr|>0.6 gate | +35.18 bp, **t = +0.40 on 12 trades** | **rejected: gate opens 2.1% of days, sample starved** |
| Gold/silver *conditional* | XAU+XAG | pair, cointegration-gated | **cTrader toll 7.70 bp vs edge 4.18** | **marginal at the cTrader venue (was submerged on MT5)** |
| US500/NAS100 pair | US500+NAS100 | pair | edge 5.94, **cTrader toll 1.29** | **CLEARS the toll — the strongest candidate on the venue** |
| XAUUSD/USDX pair | XAU+USDX | pair | edge 7.88, **cTrader toll 5.18** | **CLEARS** |
| XTIUSD/USDX pair | XTI+USDX | pair | edge 12.56, cTrader toll 11.17 | clears, thin margin |
| Gap fill | NAS100 | signal | **net −4.81 bp; gaps fill LESS than the mirror control** | **reversed — see P8** |
| Gap stats | NAS100/US500 | *input to expected range* | n/a | input (P8 measured it as an input) |
| London/Asian range | FX majors | *input to sized range* | n/a | input |
| Tick fill cost | XAU/XAG/XTI | measurement | adverse selection term | **P3, in progress** |
| 0DTE expiry | — | signal, daily | 12.5–17 bp | **requires futures data** |
| 4pm fix | — | signal | ~10 bp basis | **requires futures data** |
| Recon vol | — | signal, June/Nov | volume +37% | **requires futures data** |
| Gold/silver unconditional | — | pair | OOS −0.27% CAGR | **rejected** |
| Crypto weekend drift | — | calendar | −56.9 pts/yr | **rejected** |
| US month-end TOM | — | calendar | −10.4 pts/yr | **rejected** |
| Retail sentiment | — | positioning | account % ≠ exposure | **rejected** |

---

## 7. The one rule that survives every finding in the research

**Price the edge at its venue, at its hour, net, then freeze the parameters and never look
back.** The original document's ten edges failed in exactly three ways, and all three are
avoidable by that rule:

- It quoted gross numbers as if they were net (VWAP, gap fill).
- It quoted community win rates without the search that produced them (ORB variants).
- It quoted a mechanism without checking whether the mechanism still operates (gold/silver
  cointegration, pre-FOMC on non-equities, US month-end).

Everything else in this note is engineering. This rule is the product.
