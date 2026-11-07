# research/

Exploratory / pre-registered strategy investigations. Not covered by the permanent suite;
results are not promoted to `artifacts/` until they survive the same discipline as anything
else: a frozen specification written before the first result is read, a control that measures
something, and a cost charged at the instrument's own venue.

## Contents

| File | Question | Result |
|---|---|---|
| `p1_vwap_nas100.py` | Does the VWAP 2σ fade on the own NAS100 panel clear its toll? | **Rejected.** −14.60 / −27.26 / +1.87 bp (dev/val/holdout). Reversion leg earns +37.14 bp at 81.8% when the target is hit; the 29% that miss exit at session close for −140.42 bp. Financing priced out at 1.1 bp. |
| `p2_hourly_cost_screen.py` | Which instrument × hour combos clear their own toll? | **3 candidates, all below toll.** 163 cells → 24 \|t\|>2 → 14 same-sign OOS → 7 past cost gate → 3 after a year-by-year sign gate. EURUSD h22/h23 (5/6 yrs) lean +0.25/+0.48 bp against a 1.07 bp toll. |
| `p3_spread_inference.py` | Can a 6-year OHLCV panel's spread be inferred, so 6-year studies can be priced? | **Dead in principle.** Inference MAE does not beat a flat median (−36% on XAGUSD, −62% on EURUSD, +34% on XPTUSD which is mostly bias). Bar range explains R²=0.077 and \|ret\| R²=0.045 of spread variance — the information is not in OHLCV. |

## Why the negative results stay

A frozen spec that fails is the deliverable. Three of this directory's defects produced
*plausible* numbers — a 99.3% win rate with PF 3×10⁴ from a look-ahead fill, a mirror control
that returned exactly −(P&L) and measured nothing, and t-stats of −57,082 from a bp-vs-log-return
unit mismatch. None would have been caught by reading the output. That is the same failure class
the permanent suite exists to catch, and the record of it is worth more than the results it
replaced.
| 4 | `p4_pair_gold_nasdaq.py` | Does gating a pair on live cointegration turn an unprofitable pair into a tradeable one? | **The gate works, the sample starves.** Gold/Nasdaq is the venue's cheapest pair (0.99 bp both legs): gated +35.18 bp vs an ungated control of +3.41 bp, but **t = +0.40 on 12 trades** because the gate opens on only 2.1% of days. Gold/silver conditional is dead on cost (15.89 bp toll vs a 4.18 bp edge). |
| 5 | `p5_pair_us500_nas100.py` | Does the US500/NAS100 pair clear once the cTrader toll prices it? | **Negative, and well-powered.** −7.91 bp net, t = −2.09, 259 trades, gate open 99.6% of days. Year-by-year: negative in **6 of 7 years**. The +5.94 bp in `pair_sweep.json` is a 2021 artifact. |

**Cross-venue addendum.** `src/venue_cost.py` now also reads the cTrader live quote tape
(`ctrader/tape/`, 123,983 quotes / 27 instruments), which prices symbols the MT5 panels do not
cover (US500, USDX, XPDUSD) and is a second venue we can choose between. The two disagree
systematically by asset class — 0.84–0.97× on gold/indices/energy, 0.43–0.53× on
silver/aluminium/copper/lead. That unblocked four pairs from pricing; all four were then tested
as gated pairs with the year gate, and none survived (`p5_pair_us500_nas100.py` is the template:
US500/NAS100, −7.91 bp, t −2.09, negative in 6 of 7 years).
| 6 | `p6_weekend_crypto.py` | Weekend crypto: is it a direction anomaly or a liquidity effect? | **Activity effect confirmed; direction NOT ESTIMABLE from this sample.** BTC weekend 9,091 ticks/hr vs weekday 17,857 (0.51×), ETH 3,694 vs 10,000 (0.37×) on 1.3M/1.26M ticks. The direction comparison is contaminated: BTC fell 7.42% over the 13-day window with daily ±500 bp, one weekend day missing (Nov 1 = 0 ticks), and daily returns far larger than the weekend effect ever claimed. Reconciles Kaiser/Mueller with Sovbetov: activity is robust, direction is not testable at this sample size. |
| 7 | `p7_orb_nas100.py` | The document's #1 edge: does ORB clear the toll on our NAS100? | **Negative, well-powered.** 15-min range/wide stop: 1,738 trades, 31.4% win, raw **−1.41 bp** before costs, net −1.96 bp, t −10.3, negative in **7 of 7 years**. 1,118 of 1,738 exit at session close — the drift never arrives. Stops: with a midpoint stop (the doc's literal rule) 1,149 of 1,738 stopped immediately, which is a *defect* not a result — a breakout retracing half its range is ordinary. |
| 8 | `p8_gap_fill_nas100.py` | Does NAS100 gap fill pay, with the mirror control the doc never ran? | **Reversed, not merely negative.** Tradeable gaps (10–50bp) fill *less* often than a same-distance mirror level on the other side: 10–25bp fills 44.3% vs mirror 57.5% (−13.2 pts); 25–50bp fills 8.0% vs mirror 42.0% (−34 pts). Sub-10bp gaps "fill" 97.3% but a mirror level is hit 98.0%. The fade itself: gross −4.27 bp, net −4.81 bp, t on 191 trades, 3/6 years. |
| 9 | `p9_ranking_null_calibration.py` | Does the *shipped* ranking null hold at nominal alpha? | **It did not — 100% FP at market AR(0.3), now 5.0% after the fix.** The real statistic used raw returns while the null residualised; `beta_neutralise` is now applied to both sides. The shipped calibration test itself contained the same asymmetry and passed at 0% while production failed. |
