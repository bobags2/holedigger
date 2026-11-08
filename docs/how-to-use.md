# How to use what we built — from "I just deposited" onward

Written 2026-11-07. Everything below references files in this repo (`holedigger/`) and
`Z:/projects/ctrader/`. Every number quoted is measured, not estimated — see
`docs/where-this-stands.md` for where the numbers come from and what was withdrawn.

---

## The single thing to understand before you do anything

**What we built does not trade, and does not tell you what to trade.**

It answers two questions:

1. **How much should I put on this trade?** → `src/risk_rules.size_for`
2. **Should I stop trading today?** → `src/risk_rules.kill_switch`

That's the whole product. It's a guardrail, not a signal generator. We looked for a trading
edge for a long time and did not find one: eight families of price features, a meta-labeller on
your own 3,870+3,473 closed positions, cross-sectional ranking, pairs, time-of-day seasonality,
gap-fill, ORB — **zero survivors**. The mode's own history trades at **−1.82 bp per position**
(gross −$4,614, commission −$1,413, net −$6,029).

So depositing does not buy you "our strategy". It buys you: correctly-sized trades with a hard
stop, on whatever signal you supply.

---

## Step 1 — Deposit, then check the account isn't too small to use

```powershell
cd Z:\projects\holedigger
python -c "import sys; sys.path.insert(0,'.'); from src.live import risk_monitor as rm; import json, asyncio; from pathlib import Path
cfg=json.load(open('../ctrader/config.json'))['ctrader']
m=rm.RiskMonitor(spec_path=Path('artifacts/forward_spec.json'), account_cfg=cfg)
print('venue floors:', {k:v for k,v in rm.VENDUE_MIN_NOTIONAL.items() if v>100})
print('gold minimum notional:', rm.VENDUE_MIN_NOTIONAL['XAUUSD'])"
```

**The number that decides whether this works at all:**

| Equity | Gold max notional | Risk per trade | Verdict |
|---|---|---|---|
| $54 (now) | — | — | **DO NOT TRADE** |
| $100 | $500 | $0.50 | tradeable, but the floor bites fast |
| $1,125 | $5,625 | $5.63 | the sizing rule works properly |
| $5,000 | $25,000 | $25.00 | full range of instruments |
| $25,000 | $125,000 | $125.00 | platinum becomes tradable |

**Below ~$1,000 the honest answer from the monitor is `DO NOT TRADE` for several symbols**,
because the risk-correct size lands under the broker's minimum lot. That's not caution — it's
arithmetic: rounding up to the venue minimum would trade a size the risk rule never approved.

**Deposit at least $1,125, preferably $5,000+.** If you can only fund $500, trade XAUUSD and
NAS100 only — silver, oil and platinum will correctly refuse.

---

## Step 2 — Start the monitor and let it run

```powershell
cd Z:\projects\holedigger
python -m src.live.risk_monitor
```

It authenticates against account **47477688**, loads the broker's symbols, and logs every
60 seconds:

```
equity 1125.08 | halted False (-)
   XAUUSD   notional      5625.40  risk     5.63  cap  1.327bp
   NAS100   notional      5625.40  risk     5.63  cap  1.548bp
   XAGUSD   notional      2599.11  risk     5.63  cap 21.644bp cost_cap (from 5625)
   XPTUSD   DO NOT TRADE (sized notional 232.08 is below the venue minimum 380.00)
```

Four things this tells you:
- **`notional`** — the size you may put on. Never exceed it.
- **`risk`** — what that size loses if the 10 bp stop hits.
- **`cap ... cost_cap`** — silver's cost cap is shrinking it from $5,625 to $2,599 because its
  spread is 15× wider than gold's. **This is the useful part: cheap instruments get bigger.**
- **`DO NOT TRADE`** — don't force it. Either deposit more or pick a different symbol.

Leave it running while you trade. If it ever logs `KILL SWITCH`, **stop trading for the day** —
that means your account has dropped 25% from its peak, which on this account's own history
reached −101%.

---

## Step 3 — Trade with the numbers it gives you

On every trade, by hand or via the bot:

1. Look at the **current `notional`** line for your symbol.
2. Set your position size to **at most** that number.
3. Put a stop **10 bp** away (that's the stop the sizing assumes).
4. If a 10 bp stop would be less than a reasonable tick distance, don't trade that symbol.

**Why 10 bp?** It's the stop the sizing formula uses. It's also measured: gold's whole session
range is ~14.8 bp, so 10 bp is roughly two-thirds of a typical hour. A tighter stop gets hit by
noise (we measured: the 0.44 bp cap is touched on **59.9%** of gold positions — a stop that
tight just pays the toll repeatedly). A wider stop means smaller size.

**The arithmetic, so you can sanity-check it:** `notional = equity × 0.5% ÷ 10bp`. Lose on
every trade and you lose 0.5% of the account each time. Ten losses in a row is −5%, not −50%.

---

## Step 4 — Wire it into the bot so it's automatic (optional but better)

The monitor is the authority; the bot is a thin client of it. Two ways:

**cTrader cBot** (you already have `calgo/OfrFlagship`): have the cBot read
`artifacts/forward_spec.json` and call the same `size_for` / `kill_switch` functions, capping
every order it sends.

**MT5 Expert Advisor**: same, against `ELD`/`EA` in MetaTrader.

I have **not** built either, deliberately. Wiring an automated cap to a live order path is a
deployment decision with real consequences and you should make it after watching the monitor run
for a few days. If you want it, say so and I'll build the cBot — it's a small amount of code
because the logic already exists and is tested.

---

## Step 5 — Forward test on demo before real money

**This is the step that actually tells you something.** Run the monitor plus your trading rules
on a demo account for **20 sessions**, and record:

- net expectancy after costs, with a confidence interval
- max drawdown
- ruin probability (are you down 50%?)
- the worst 10 trades as a share of total P&L

The pass bar is in `artifacts/forward_spec.json`:

```json
"target": {
  "net_expectancy_ci_low_min": 0.0,   ← the lower bound must clear ZERO
  "max_drawdown_pct_max": 25.0,
  "ruin_prob_max": 0.05,
  "worst10_share_max": 0.5
}
```

**If the confidence interval's low end is above zero on demo, you have something.** If it
isn't, you don't — no matter how good the chart looks. On your own real history this same
measurement is **−1.82 bp per position**, so the real book does not pass.

---

## What this does *not* do, stated so you're not surprised

- **It does not tell you what to buy or sell.** No signal. Eight families tested, zero found.
- **It does not make a negative-expectancy strategy positive.** Measured: correctly-sized, a
  book at −1.82 bp/trade just loses more slowly.
- **It does not fix sizing by itself** if you override it. The measured failure on your account
  was **645× leverage** — $40,438 notional against a $62.72 median balance. The monitor gives
  you the number; you have to use it.
- **It does not replace a stop-loss in your platform.** The sizing assumes you actually put the
  10 bp stop on.

---

## The short version, taped to your monitor

1. Deposit ≥ $1,000 (≥ $5,000 if you want all instruments).
2. `python -m src.live.risk_monitor` — leave it running.
3. Trade at or below the `notional` it prints, with a 10 bp stop.
4. `KILL SWITCH` logged → stop for the day.
5. 20 demo sessions, and check the CI low end clears zero. That's the only thing that counts.

---

## If you want the next real step

The one untested place a signal could still live is **tick-level microstructure** — order-flow
imbalance, spread state, quote refresh — on the 1.58M ticks we already have. Everything else in
this repo bars is bars, and everything that reads bars has come back negative. That's the honest
remaining unknown, and it's the difference between "we looked everywhere except the last place"
and "we looked everywhere".

Ask and I'll build that test.
