# CFD Algorithmic Trading Edges — 2026 Update

**Revision of the ten-edge report originally written as `cfd_research.md` (in Downloads,
not version-controlled), written 2026-11-06.**

*Repo copy: `docs/cfd-edges-2026-11.md`. The original source file lives in Downloads and is not
version-controlled; this copy is authoritative and carries the tests run against it in
`research/p1`–`p8`.*

This update re-tests all ten edges in the original report against research published after it
was written, and adds five new ones. The headline: **five of the ten original edges are now
contradicted or materially weakened by post-document evidence**, and the pattern behind each
failure is the same. Two were gross-positive and cost-buried (VWAP bands, gap fill). Two were
parameter-instability deaths (ORB variants, London breakout). One was a regime change nobody
priced (gold/silver cointegration). The original report's failure mode was not bad research —
it was quoting win rates from community backtests that never charged costs or held out data.

Everything below names its source and states whether the number is gross or net.

---

## Part A — Verdict on the original ten

| # | Edge | Original claim | 2026 verdict |
|---|---|---|---|
| 1 | Opening Range Breakout | "Still working. Won't be arbed away" | **Works on NQ, fails on ES, and 0/30 variants survived honest out-of-sample** |
| 2 | VWAP Reversion | "Structural — institutions all use VWAP algos" | **Refuted as written: lost money *before* costs on ES 5-min** |
| 3 | Pre-FOMC Drift | "Still documented as alive" | **Alive but narrower: equities only, press-conference meetings only, VIX-conditional** |
| 4 | London Breakout | "Community-verified, WR 55-60%" | **Direction is a coin flip (50.1% over 26,246 days); the *range* is the useful part** |
| 5 | Gold/Silver Ratio | "Structural — shared mining supply dynamics" | **Cointegration detected only 20% of the time; break in late 2017; walk-forward OOS was −0.27% CAGR** |
| 6 | Retail Sentiment | "Documented contrarian edge" | **Downgraded: the published % counts accounts, not exposure — 60% long can be net short** |
| 7 | Month-End FX Rebalancing | "BIS working papers, Quantpedia" | **Dead in US equities (disappears after 2001); survives only in the 7-day window on international/EM** |
| 8 | Grid Trading | "15-30% annual, DD 8-12%" | **Unchanged, but 2026 financing is its dominant cost and swap-free is not a legal workaround** |
| 9 | Regime Switching | "VHF filter adds 30-50%" | **Unfalsified but unproven; the adjacent evidence says most filters fail (4.2% survival)** |
| 10 | S&P 500 Gap Fill | "70%+ fill rate on smaller gaps" | **Fill rates real, trade dead: −$14 to −$22 per trade net, all three versions buried for Costs** |

### 1. ORB — instrument-specific, and the parameter search is the killer

Crabel's own 2026 century study (84 futures markets, 1923–2025) measures the base rule's decay
directly: Sharpe averages above six in the 1970s–1990s, **2.92 in the 2000s, 0.91 in the
2010s**. The sign never flipped in eleven decades, but the *size* is a third of what the folklore
assumes.

The instrument split is stark. 5-minute ORB on NQ, 2024-10 to 2026-10 (CE Market Solutions):
515 trades, 35.7% win rate, +3,513 pts gross ($70,270 per contract), profit factor 1.128 — but
max drawdown 3,114 pts. The identical rule on ES over the same window: 515 trades, 29.5% win
rate, **PF 0.980, −99.75 pts**. Same rule, same period, opposite outcome.

The decisive number comes from a 324-variant study on NQ (3 range lengths × 3 entry timeframes
× 12 setups × 3 targets, 2016-2026, 1,050,582 fills): **30 versions were clearly positive
in-sample and 0 of those 30 stayed clearly positive out-of-sample.** Three of the thirty were
the random-direction control.

**What to change:** trade it on NQ only. Write the parameters down and freeze them before
touching data. Treat the first 5-minute candle as the entry — not setup time (see gap-fill
timing below for why: 25.5% of ES fills land in the opening 5-minute bar). Crabel's own
contraction/expansion framing (condition on yesterday being a narrow-range day) is the part of
his book with the most recent support.

### 2. VWAP Reversion — the doc's exact rules lost money before costs

An ES 5-minute backtest (1,737 sessions, 2020–2026, $30.76 round trip) killed all three
Investopedia formulations:

| Variant | Gross/trade | Net/trade | Winners |
|---|---|---|---|
| With the VWAP (cross) | +$9.06 | −$21.70 | 20.5% |
| Against the VWAP | −$9.06 | −$39.82 | 72.5% |
| **Bands (the doc's rule)** | **−$11.75** | −$42.51 | 22.4% |

The band version **lost before costs**. 75% of trades hit the stop, and the entries "did worse
than every run of the control." A separate VWAP lab reached the same conclusion from the other
direction: the −2.5σ lower-band fade looked like the standout in a frictionless MFE/MAE study,
then lost money out-of-sample once fills, slippage and commission were modelled. Their verdict
is the one to quote to anyone proposing it: *"A theoretical edge that disappears once friction
is applied is worse than no edge at all, because it would have shipped."*

The wrinkle that makes this a *timeframe* failure rather than a *concept* failure: the same
VWAP 2σ fade is gross-positive on slower bars. NQ 15-minute: 358 trades, 65.1% winners,
+2,325 pts, PF 1.148. NQ 1-hour: 209 trades, 62.7% winners, **+3,219 pts, PF 1.428**. NQ
5-minute: 472 trades, PF 0.972. One instrument, one rule, three timeframes, and the fastest
one is the losing one.

**What to change:** 15-minute or 1-hour only. Never 5-minute. And drop the "institutions pull
price back to VWAP" rationale — that mechanism is not what the data supports.

### 3. Pre-FOMC Drift — alive, but only in a narrower window than the doc says

The decline is documented in stages. Ben Dor & Rosa (2019, *Journal of Fixed Income*): the 24h
pre-FOMC equity return fell from **0.5% per meeting to roughly 0.1% after 2011** and was no
longer significant — and there was **no evidence of the effect in Treasuries, TIPS, currencies,
precious metals, or commodities**. Boguth, Gregoire & Martineau (2019, JFQA): the drift
**disappeared entirely for announcements without a press conference**. Knox &
Vissing-Jorgensen (2025, Federal Reserve FEDS 2026-023): it weakened further even for
press-conference meetings, and the decline is explained by *reduced uncertainty* — the
post-ZLB indicator loses significance once VIX enters the regression.

QuantSeeker's test through December 2024 confirms it is still there: SPY CAGR ~4%, Sharpe
0.5–0.6, trading only 5% of days. The useful new finding is **conditionality**: splitting by
entry VIX quartile, the drift is strong in high-volatility environments and close to zero in
low-volatility ones. It also flatlined 2016–2019, so recency is not proof.

**What to change:** equities only — do not run this on gold, oil, or FX CFDs, the 2019 paper
tested those and found nothing. Trade only meetings *with a press conference*. Gate on VIX
level. Expect a ~0.5 Sharpe, not a goldmine.

### 4. London Breakout — the range is real, the direction is not

Two independent large-sample tests agree. First London hour direction vs. the rest of the
morning, 26,246 days across 13 markets (CRTLAB, 2017–2026): **50.1% same-way, with the control
(the hour *before* London) at 50.1%.** Following it averaged **0.00R ± 0.01**. Fading it
averaged −0.07R. EUR/USD 51.4%, GBP/USD 50.3%, gold 49.5%.

A second study (23 years hourly, 19 markets) on whether the London hour even moves more:
median movement in the window is **89% of a normal hour**. EUR/USD moves 90% of its normal
size in its own open hour. The one genuinely loud market is GER40 at 123% — because that hour
is also the Frankfurt cash open.

But the same 68,966-session study found the range half is load-bearing: only **0.32% of
sessions leave the first hour's initial balance unbroken**. A session breaks one side on a
median 47.2% vs. a 45.0% control — real, but smaller than every spread on the board.

**What to change:** stop trading the breakout direction. Use the first-hour range as a
*volatility input* — it predicts how far the session will travel, which is the axis that
survives a holdout. That is the honest and more useful read of the same data.

### 5. Gold/Silver Ratio — the doc's strongest claim is now the weakest

Caporale, Fons Palomares & Gil-Alana (2026, CESifo), 2010-01-04 to 2025-11-28: standard
cointegration tests **fail on the full sample and the pre-break subsample**; only the
post-break subsample yields one. PELT and Bai-Perron both detect a **structural break in late
2017**. Local Whittle estimates show high persistence and slow adjustment; FCVAR finds error
correction only in the post-break period.

Bank of America's rolling 5-year regressions: cointegration in only **20% of monthly
regressions since 1996** — "the gold-silver pair never makes the cut." Critically, the
cointegration episodes cluster in 1997, 2002, 2007–08 and 2015–16, and **absent during COVID,
Russia-Ukraine, and the 2024–26 rally**. Their cause is structural: central-bank gold buying is
price-insensitive, while silver is driven by industrial demand and harder to store. The ratio
moved 60:1 → 70:1 in a single month in 2026, which is the opposite of a mean-reverting input.

And the walk-forward evidence is unambiguous. A 25-year cointegration pairs backtest with
expanding-window hedge ratio, correlation filter, vol targeting and 1bp/leg costs finished at
**−0.27% CAGR out-of-sample**.

**What to change:** do not run this as an unconditional pair. The version that survives is
*conditional on the cointegration test itself*: trade only when the rolling correlation /
cointegration statistic is currently positive, which is roughly 20% of the time and clusters
in crisis regimes. Even then, the expected return is near zero and the drawdown profile is
worse than buy-and-hold.

### 6. Retail Sentiment — the published number is not what it looks like

The doc's rule (">70% long → go short") is measuring the wrong quantity. A IG sentiment reading
**counts accounts, not exposure**. The worked example from the measurement literature: 60 accounts
long at 5,000 EUR each and 40 accounts short at 20,000 EUR each gives 60% long by accounts and
**27% long by size**, with a net position of −500,000 EUR. Separately, the long share can rise
from 60% to 75% with *no new longs* — 20 shorts simply closed. A percentage can rise while its
numerator stays fixed.

IG's own guide now says it plainly: *"a useful input rather than a reliable standalone signal"*
and the changes matter more than the level. The usable combination from the 2026 practitioner
literature is extreme level (>80%) **plus divergence** — price making higher highs while the
sentiment reading is flat or falling — plus a rising 20-day average and still-positive 24-hour
change. That last combination is the "last wave of momentum" pattern.

**What to change:** require divergence, not just extreme level. Read the change, not only the
level. Accept that account share ≠ positioning. And note the practical problem: there is no
clean free API for MT5/cTrader — IG publishes it on their platform and market-data pages, and
you need an account or a scraper to get it in.

### 7. Month-End FX Rebalancing — dead in the US, alive internationally on a wider window

Han, Han & Tian (2025, *Finance Research Letters*): the turn-of-month effect **disappears
entirely after 2001**, and the Ogden liquidity explanation no longer holds; transaction-cost
declines let arbitragers trade against it. An independent 2025 thesis (2006–2023, replicating
McConnell & Xu) finds the same disappearance. A 20-year QQQ test of the plain [last day + first
three] rule returned **−10.4 points/year versus the benchmark, p = 1.000**.

But the mechanism survived in a different window. The Etula et al. (2020, *RFS*) explanation is
T+3 settlement: institutions are net sellers on days 0−8 to 0−4 raising cash, then buyers on
0−3 to 0+3. Testing that **7-day [−3:+3] window** rather than the classical 4-day one, most
equity ETFs still show a significant 5–12 bp/day effect, and international markets are far
stronger — Brazil 23 bp/day, Sweden and Hong Kong also significant. The US
[0:3] window is indistinguishable from zero, and some sectors now show an *inverse* effect.

**What to change:** drop the US version. Use the [−3:+3] window on international equity CFDs
(Brazil, Sweden, Hong Kong), where the effect is both larger and less arbitraged. Note this is
the least accessible version for a retail CFD book — these are the instruments with the
widest spreads, which eats most of a 23bp edge.

### 8. Grid Trading — same rules, but 2026 financing is the whole game

The doc's structural advice (fixed lot, never martingale, basket stop, ADX suspension, ECN
spread cap) is still correct. What has changed is the cost environment. The 2025–26 Fed cut
cycle narrowed G10 rate differentials to the point that **classic positive-swap pairs went from
5–8% annualised credits to 1–2%, below typical broker markups** — most major pairs are now
debit-heavy in *both* directions. Overnight fees consume 15–30% of gross swing profits on
positions held 3–7 days, and a grid holds longer than any other strategy on this list: a full
week is **8× the single-day swap** (1+1+3+1+1+1+1), because Wednesday's rollover covers the
weekend value-date gap.

One thing that is *not* available: swap-free accounts as a financing workaround. IC Markets
publishes swap-free holding fees ($11/lot EURUSD, $71/lot XAUUSD, $4/lot US500 after up to five
grace days, with a triple-swap night burning three grace days), and their terms **explicitly
prohibit** the use: §5.1–5.2 bans carry trades, arbitrage on swap differentials, cash-and-carry,
and any "strategy that generates insufficient spread revenue relative to the financing costs
incurred by the Company," with remedies up to retroactive charging and account termination.

**What to change:** model swap as a first-class cost, not a footnote. The published per-instrument
holding-fee schedules make this arithmetic exact, so there is no excuse for a grid backtest that
omits it.

### 9. Regime Switching — unproven, and the adjacent evidence is a warning

No direct refutation found, but the surrounding literature should lower confidence. A
5,086-rule time-of-day study (19 markets, 23 years, every hour × weekday, realistic spread
charged, last 30% held out) found **307 net positive and 214 that held up out of sample — 4.2%.**
The kill mechanism was uniform: of 400 liquid FX weekday×hour buckets, **only 12.5% move more on
average than a 1.0-pip spread**. The rest cannot pay the toll. Direction was free to measure;
trading it was not.

A VHF or ADX threshold is exactly the kind of parameter that study would bury.

**What to change:** if you build this, evaluate the filter as *part of the signal* and report
net-of-cost expectancy, not the hit rate of the classifier. A regime filter that is right 70% of
the time and removes all the trades that could pay the spread has made the strategy worse.

### 10. Gap Fill — the statistics are real, the trade is dead

An ES 2020–2026 backtest (1-minute, 1,737 sessions, $30.76 round trip, frozen spec before any
gap was computed) measured both halves:

| Gap size | Same-day fill |
|---|---|
| 0.10–0.25% | **73.8%** |
| 0.25–0.50% | 52.3% |
| 0.50–1.00% | 32.8% |
| ≥1.00% | 19.1% |

Then it traded all three versions: **GF-1X −$14/trade net (+$17 gross), GF-SMALL −$22, GF-NS
−$21.** All three buried, cause *Costs*. None beat a control that moved each trade to a random
day's open at the 95th percentile. The reason is arithmetic: the target *is* the gap (median
13.75 points on small gaps) and the round trip is about 0.6 of a point — cost is a large share
of a small target, and the gaps that don't fill cost as much as the fills earn.

A separate 6,984-gap study adds the control that matters: small gaps fill 82% of the time, but
**a mirror level the same distance on the other side is reached 77.5% of the time.** Gaps fill a
little more than chance — a 2.9-point lean, not a magnet.

The genuinely useful new finding is **timing**: 25.5% of ES gap fills and **34% of NQ fills
happen in the first 5-minute candle**; 82%/86% are complete by 11:00 CT; the lunch hour adds
4–5% and the afternoon 6–7%. Median fill time ~8:57 CT for ES, ~8:47 for NQ.

**What to change:** stop trading it as a standalone. Use gap *statistics* — fill probability by
size and by opening location (inside-range gaps fill ~70%, above/below-range ~43–47%) — as an
input to expected range and to whether a session is more likely to trend or fade. If you do
trade the fade, the edge is concentrated entirely before 8:30 CT.

---

## Part B — Five new edges

### 11. 0DTE hedge unwinding at settlement — the strongest genuinely new lead

Amayaa, Garcia-Ares, Pearson & Vasquez (August 2026, SSRN 7330098), using proprietary Cboe
signed-trade data and Algoseek quotes: when a cash-settled index option expires in the money,
the option's delta disappears but the market maker's **underlier hedge position does not**, so
it must be liquidated in the run-up to settlement. This is a channel the prior literature
missed entirely (it studied hedge *rebalancing*, driven by gamma).

Measured on SPXW: sorting days by the aggregate options-market-maker delta at 3:30 pm, the
**high-delta days average 17 bps more than the low-delta days over the final 30 minutes**
(3:30→4:00 pm); 12.5 bps using 3:45 pm delta over the final 15 minutes. Volatility over the
same window is significantly related to the *absolute* delta. And because SPX options expire
daily, this happens **every trading day**. The paper's separation of channels is the useful
part: hedge *unwinding* moves price *levels*; hedge *rebalancing* moves volatility, and only
modestly (median −0.08 pp of annualised daily vol, max +2.3 pp, which the authors call "not
large" because changes that size happen anyway on ~20% of days).

**Why this is the best new lead:** the mechanism is mechanical, not behavioural. Mechanically
scheduled order flow is the class of edge least likely to be arbitraged away, because the
participants generating it are not trying to make money on it — they are settling a position.
It is also daily-frequency, which solves the pre-FOMC problem (8 events/year can't produce a
statistically meaningful sample in a reasonable time).

**What is required and what is hard:** the paper's predictor is the OMM delta, which comes from
Cboe signed-trade-by-capacity data. The retail reconstruction path is 0DTE open interest by
strike, which Cboe publishes, from which net customer delta can be approximated and the OMM
hedge inferred. **The first deliverable of any project here is validating the proxy against the
17 bp / 12.5 bp magnitudes before a single trade.** If your OI proxy can't reproduce the
quintile spread, you don't have the trade.

### 12. The 4pm WM/Refinitiv fix — mechanical, huge, and dated

Every day at 16:00 London an enormous aggregated FX order prints at a single published rate.
It is a benchmark, so the flow is not discretionary — it happens whether or not anyone is
trading it. The FTSE Russell (2026) study measures the cost of being anywhere near it: taking
the FX observation one hour away from the 4pm fix produces an average absolute WGBI basis of
just under 10 bp, 95th percentile 18–26 bp, drifting to 15–20 bp / 35–40 bp at five hours out.
CME sells BTIC specifically to let institutions trade the fix directly, and the case study
walks a €250m rebalance executed against it.

The event-study opportunity is the April 3, 2025 fix outage: an LSEG technical disruption on a
tariff-announcement day delayed the fix to ~8pm, and index providers defaulting to the previous
day's fix produced **deviations up to 3.5% from fair index levels**. Solactive has since
proposed a synthetic fix (time-weighted average of the ±2.5-minute window, which it measures
as within 0.01% of the true fix).

**Honest assessment:** the *magnitude* of fix-window price impact is well documented; the
*retail-sized predictability* of it is not, by anyone. Mark this as speculative-but-mechanical.
The testable retail version is EUR/USD around 15:45–16:05 London, and it must be priced against
the rollover window (21:00–22:00 UTC), where spreads widen sharply — one of the time-of-day
studies explicitly flags that bucket rather than celebrating its statistic.

### 13. Russell reconstitution — now twice-yearly, and the naive version is wrong

From 2026 the Russell reconstitution moves from annual to **semi-annual**: fourth Friday of
June, plus the **second Friday in November**. $11trn is benchmarked to the indices; $217.2bn
traded in the closing moments of recon day 2025, and E-mini Russell traded 237K contracts on
the day, 37% above the week's ADV.

The 2025 LSEG post-mortem is the part that matters and it contradicts the naive trade: **the
pricing-pressure logic does not apply on recon day.** Because the preliminary lists ship from
late May and updates run weekly, traders front-run — adds get overbought and deletes oversold,
so on the day itself *additions decline and deletions rally*. Methodology changes (the Nasdaq
closing cross, 2.5bn shares executed in 0.878 seconds) further compressed it.

**What to change:** don't buy adds and sell deletes on recon day. The documented, tradable
pattern is a **volatility and volume prediction** for the week *before* the effective date, and
a documented *reversal tendency* on the day. For a CFD book this maps to US2000 (Russell 2000)
in the first three weeks of June and the first two weeks of November — with a hard caveat that
the spreads on a US2000 CFD are far wider than RTY futures.

### 14. Crypto weekend effect — the liquidity asymmetry is real, the return one isn't

Three convergent papers. Kaiser/Mueller (2024, *Finance Research Letters*): testing 10 majors
2013–2024 with bootstrap methods, there is **no robust return seasonality in Bitcoin after
2015**, the famous positive Monday effect is a post-2015 statistical artifact, January and
Halloween are unreliable — and the **only stable finding is significantly lower weekend trading
activity, robust across assets and time**. Sovbetov (December 2026, *QREF*): the Saturday
premium is broad-based, stronger under thin weekend liquidity, and concentrated in speculative,
lower-priced, liquidity-sensitive tokens; institutional-grade assets show Monday and Wednesday
effects instead; the anomaly weakened during COVID and re-emerged post-Terra. A 151-weekend
measurement through November 2026: BTC weekend +1.27% → Monday −1.96% → Friday-to-Monday
−0.71%, and **larger weekend gains are erased less often** (1 of 23 for gains ≥2%, against 17
of 59 for gains <2%). A plain long-the-weekend rule across 18 pairs returned **−56.9 pts/year
versus the benchmark** over 2017–2025.

**What to change:** use weekend liquidity as an **execution input**, not a directional signal.
Size down and expect worse slippage when weekend turnover is thin (measured at ~0.35× weekday
in the sample above). Don't run a weekend-drift calendar rule — the direction is not there, and
the costs are charged on the way in and out.

### 15. Venue cost asymmetry — boring, real, and nobody builds it

This one is not a market edge; it is an edge in the instrument you already trade, and it is the
only item on this list whose magnitude you can compute exactly in advance.

The 2026 swap literature publishes the numbers. Overnight fee = notional × effective annual
rate ÷ 365, where the effective rate is the benchmark plus a broker markup of roughly 0.5–1.0%
(premium ECN), 1.5–2.5% (standard retail), or 3–5% (offshore). Wednesday's rollover is 3×
because of the value-date gap. A position held Monday-to-Monday pays 8×. IC Markets' published
swap-free holding fees are $11/lot EURUSD, $71/lot XAUUSD, $4/lot US500. Closing before the
22:00 UTC rollover saves 80–90% of that night's charge at the cost of one spread.

Two consequences that pay for themselves immediately:

- **The same strategy has materially different net economics at different brokers**, and the
  difference is frequently larger than the strategy's gross edge. Recall that the VWAP band
  rule lost by $11.75/trade and the gap fade by $14–22/trade. A cost delta of that size is
  routinely available between venue fee schedules on the *same instrument*.
- **Holding-period cost is a strategy parameter, not an accounting detail.** A system that
  knows the rollover calendar can shift an entry or exit by one day to dodge a triple swap,
  and that decision is worth more than most signal refinements.

The publication also settles what you cannot do: swap-free accounts are contractually off-limits
as carry vehicles (see §8).

---

## Part C — What to avoid (additions to the original table)

| New trap | Why |
|---|---|
| Gross-only backtests on community sites | Both cost-killed edges in this revision (VWAP bands, gap fill) were gross-positive on the same data |
| In-sample parameter search on breakout/fade rules | 0 of 30 positive ORB variants survived out-of-sample; 3 of the 30 were the random control |
| Reading a win rate without its expectancy | Against the VWAP won 72.5% of trades and lost the most; with the VWAP won 20.5% and lost the least |
| Requoting a fill rate as an edge | 74% of small gaps fill; the trade lost money. And a same-distance mirror level gets hit 77.5% |
| Swap-free accounts as carry vehicles | Contract breach. Remedy is revocation plus retroactive charging |
| A weekend drift calendar rule on crypto | Direction unproven post-2015; −56.9 pts/yr vs benchmark on a 18-pair basket |
| Trading the Russell recon on the day | Adds overbought and deletes oversold by then; recon day often trades the *other* way |
| Gold/silver as an unconditional pair | Cointegration only ~20% of the time; walk-forward OOS −0.27% CAGR |

---

## Part D — Implications for a build

Four of the ten original edges needed only a *timeframe* change to become net-viable (VWAP
15m/1h, gap-fill timing, ORB NQ-only, month-end international). Two needed to stop being
trades and start being inputs (London range, gap statistics). Three needed to be dropped or
narrowed (US month-end, gold/silver unconditional, pre-FOMC on non-equities). One
(0DTE hedge unwinding) is genuinely new and mechanically defended.

The full build implications are in the accompanying note, but the single design rule this
revision supports is: **quote every edge in net-of-cost expectancy at its own venue and its own
time of day, and hold its parameters frozen.** Every failure above was a cost failure or a
parameter-search failure. None was a "the market changed and the effect vanished" failure — the
effects that did decay (ORB's Sharpe, pre-FOMC's magnitude) decayed *slowly and measurably*,
which is a different and much more tractable problem.
