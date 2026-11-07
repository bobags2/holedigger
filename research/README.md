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
