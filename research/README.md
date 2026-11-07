# research/

Exploratory / pre-registered strategy investigations. Not covered by the permanent suite;
results are not promoted to `artifacts/` until they survive the same discipline as anything
else: a frozen specification written before the first result is read, a control that measures
something, and a cost charged at the instrument's own venue.

## Contents

| # | File | Question | Result |
|---|---|---|---|
| 1 | `p1_vwap_nas100.py` | Does the VWAP 2σ fade on the own NAS100 panel clear its toll? | **Rejected.** −14.60 / −27.26 / +1.87 bp (dev/val/holdout). The reversion leg earns +37.14 bp at 81.8% when the target is hit; the 29% that miss exit at session close for −140.42 bp. Financing priced out at 1.1 bp, so the tail is the mechanism, not the toll. |
| 2 | `p2_hourly_cost_screen.py` | Which instrument × hour combos clear their own toll? | **3 candidates, all below toll.** 163 cells → 24 with \|t\|>2 → 14 same-sign OOS → 7 past the cost gate → 3 after a year-by-year sign gate. EURUSD h22/h23 (5/6 yrs) lean +0.25/+0.48 bp against a 1.07 bp toll. |
| 3 | `p3_spread_inference.py` | Can a 6-year OHLCV panel's spread be inferred, so 6-year studies can be priced? | **Dead in principle.** Inference MAE does not beat a flat median (−36% on XAGUSD, −62% on EURUSD, +34% on XPTUSD which is mostly bias). Bar range explains R²=0.077 and \|ret\| R²=0.045 of spread variance — the information is not in OHLCV. |
| 4 | `p4_pair_gold_nasdaq.py` | Does gating a pair on live cointegration turn an unprofitable pair into a tradeable one? | **The gate works, the sample starves.** Gold/Nasdaq is the venue's cheapest pair (0.99 bp both legs): gated +35.18 bp vs an ungated control of +3.41 bp, but **t = +0.40 on 12 trades** because the gate opens on only 2.1% of days. Gold/silver conditional is dead on cost (15.89 bp toll vs a 4.18 bp edge). |
| 5 | `p5_pair_us500_nas100.py` | Does the US500/NAS100 pair clear once the cTrader toll prices it? | **Negative, and well-powered.** −7.91 bp net, t = −2.09, 259 trades, gate open 99.6% of days. Year-by-year: negative in **6 of 7 years**. The +5.94 bp in `pair_sweep.json` is a 2021 artifact. |
| 6 | `p6_weekend_crypto.py` | Weekend crypto: is it a direction anomaly or a liquidity effect? | **Activity effect confirmed; direction NOT ESTIMABLE from this sample.** BTC weekend 9,091 ticks/hr vs weekday 17,857 (0.51×), ETH 3,694 vs 10,000 (0.37×) on 1.3M/1.26M ticks. Direction is untestable here: BTC fell 7.42% over the 13-day window with daily ±500 bp and one weekend day missing. Reconciles Kaiser/Müller (activity only) with Sovbetov (Saturday premium): activity is robust, direction needs another sample. |
| 7 | `p7_orb_nas100.py` | The document's #1 edge: does ORB clear the toll on our NAS100? | **Gross ≈ 0.** With the published rule (no stop, no target, held to session close): 1,805 trades, gpp −0.010 bp, 48.3% win, net −0.558, t −11.16, 0/7 years. Gold +0.172 bp gross, 1/7 years. Two implementation defects first — peak-hour instead of the volume step, and a stop+target the published rule does not have. |
| 8 | `p8_gap_fill_nas100.py` | Does NAS100 gap fill pay, with the mirror control the doc never ran? | **Reversed, not merely negative.** Tradeable gaps fill *less* often than a same-distance mirror level on the other side: 10–25 bp fills 44.3% vs mirror 57.5%; 25–50 bp fills 8.0% vs mirror 42.0%. Sub-10 bp gaps "fill" 97.3% while a mirror level is hit 98.0% — anything near the open gets touched. The fade: net −4.81 bp, 3/6 years. |
| 9 | `p9_ranking_null_calibration.py` | Does the *shipped* ranking null hold at nominal alpha? | **It did not — 100% FP at market AR(0.3), 5.0% after the fix.** The real statistic used raw returns while the null residualised; `beta_neutralise` now applies to both sides. The shipped calibration test contained the same asymmetry and passed at 0% while production failed at 100%. |
| 10 | ranking-book re-cost (fix in `src/ranking_book.py`, no script) | Is the book's cost charged on the session it actually holds over? | **No — cost was drawn from the wrong session. 42.98 → 12.02 bp, net −23.97 → +6.99 bp, percentile 1.000.** Beta-neutralised net +4.23 bp. The first result in this repo to clear its own null; limits recorded in the commit. |

## Why the negative results stay

A frozen spec that fails is the deliverable. Nine defects in this directory produced
*plausible* numbers before the real result, and all nine are recorded rather than quietly
fixed:

1. A look-ahead exit fill — 99.3% win rate, PF 3×10⁴, mirror-symmetric (P1).
2. A mirror control that returned exactly −(P&L) and measured nothing (P1).
3. t-stats of −57,082 from a bp-vs-log-return unit mismatch (P2).
4. A reconstructed per-trade return from an invented constant, then from mismatched sds, then
   from price ratios instead of log-return differences (P4).
5. A session detected from the peak-volume hour instead of the volume step, so the "opening
   range" was built from 30–60 minutes of pre-open tape (P7).
6. An exit rule with a stop and target the published strategy does not have (P7).
7. A test harness that appended a test after `if __name__`, so it was never registered — 9
   tests passed while a 10th existed and was silently never executed (P6/P7 harness).
8. A calibration test that applied a different transform to the real statistic than to the
   null, so it passed at 0% while the production pipeline failed at 100% (P9).
9. A cost drawn from the wrong session and both its columns, overstating the toll 3.6× (item 10).

## Cross-venue addendum

`src/venue_cost.py` reads the cTrader live quote tape (`ctrader/tape/`, 123,983 quotes /
27 instruments) as well as the MT5 panels, which prices symbols MT5 does not cover (US500,
USDX, XPDUSD) and gives a second venue to choose between. The two disagree systematically by
asset class — 0.84–0.97× on gold/indices/energy, 0.43–0.53× on silver/aluminium/copper/lead.
That unblocked four pairs from pricing; all four were then tested as gated pairs with the
year gate, and none survived.

Crypto is in the ledger too, from the only market that runs on a weekend: BTCUSD 6.62 bp and
ETHUSD 29.4 bp round trip, i.e. 15× and 67× gold's 0.44 bp — by a wide margin the most
expensive instruments on the venue, which is why the weekend activity finding is an execution
input and not a strategy basis.

## Units sanity check (reviewer's item)

`p7`'s t of −11.16 on 1,805 published-mode trades with net −0.558 bp implies a per-trade SD of
**2.13 bp** (`|net| * sqrt(n) / |t|`), which is the right magnitude for a 6.5-hour NAS100
session. The ~8 bp figure the reviewer derived came from the stopped-mode run, which is
superseded (its stop and target are not in the published rule). The bp/log-return unit bug that
produced t = −57,082 is fixed and is item 3 above.

| 11 | `p11_range_model.py` | Is next-period RANGE predictable, and does predicted range clear the toll? | **Strongly yes, and it reframes the whole register.** Out-of-sample R² **0.38–0.70** across the panel (directional families measured ~0.45 bp effects). Opportunity = predicted range ÷ measured toll: **XAUUSD 31.6×** (p90 54.3×), XTIUSD 4.15×, XCUUSD 1.65×, XNGUSD 0.96×, XPTUSD 0.35×. Not an alpha claim — a sizing/filtering input. |
