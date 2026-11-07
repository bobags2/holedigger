# Null-calibration verification, and what it withdrew

Independent simulations run against the repo's own nulls, prompted by an external review.
Both "survivor" results from the new families are **not certified**, and both are withdrawn.

The tests below are the property that was missing from the repo: *the null's own false-positive
rate on synthetic data with a known answer of zero*. Every null in this project is now
expected to be checked that way, not just against a planted effect.

## 1. The conditional family's pooled null is invalid (70% false positives)

`src/conditional.py`'s pooled test shuffles **state labels across the pooled
window-observations** — roughly `n_days x n_windows` values, where one day contributes once
per *overlapping* window. Those observations are not independent, so the null treats
correlated data as independent and is far too tight.

Simulated: one market factor per day, clock-slot seasonality, spread state drawn
independently of returns. 20 trials, 200 worlds each, alpha = 0.05.

| null | false-positive rate | null p95 of \|diff\| |
|---|---|---|
| observation-shuffle (shipped) | **70%** | 5.76 bp |
| day-permutation (one label per day, applied to all its windows) | **5%** | 31.38 bp |

The shipped null is ~5x too tight. The reported "percentile 1.000 on XAUUSD / XAGUSD /
XTIUSD" is therefore not evidence, and the per-symbol sign matching each asset's trend in
the sample is consistent with leftover drift rather than a mechanism.

**Withdrawn.** The fix is a day-level permutation: one label per day, applied to every
window that day contributes.

## 2. The ranking family's within-session null is invalid (42% false positives)

`src/cross_section.py` shuffles the **feature across names within each session**. That
destroys beta structure, and the mean rank correlation then picks up market-sign persistence
times beta dispersion — a real quantity that has nothing to do with the feature predicting
anything.

Simulated: one-factor market, zero predictability by construction, beta dispersion at the
equity-like level (beta ~ U[0.5, 2.0]). 12 trials, 120 worlds each.

| market autocorrelation | within-session null (shipped) | session-shift null |
|---|---|---|
| 0.0 | **42%** | 8% |
| 0.3 | 17% | 8% |
| 0.6 | 17% | 8% |

So `rho = +0.035` is not certified. The miscalibration's *direction* matches the result,
which is the worst case — the whole effect could be this artifact.

**Withdrawn.** The fix is a session-shift null (pair each session's feature with a different
session's returns), plus beta-neutralised returns, plus an explicit check that the book's
P&L is not `(beta_long - beta_short) x market` after up-days. The ranking book is dead
regardless (`-24 bp` net, 39% hit rate), and extreme prior-return names are the widest-spread
names, so the feature selects its own cost.

## 3. The tick markout cannot support its t-statistics

Three separate defects, each independently sufficient:

1. `markout()` converts minutes to rows using the **median tick gap**, so "+30m" is not 30
   minutes on a bursty tape.
2. A "fill" is any tick where the quote moved — that selects momentum ticks, so the markout
   is conditioned on the very move it then measures.
3. The t-stat treats ~290k autocorrelated ticks from ~2.7 days as independent.

And a structural limit, not a bug: **this feed carries only bid and ask** — no last price, no
volume — so order flow is not measurable on it at all. No fill inference from this source can
be print-level.

**Downgraded to descriptive.** Sign and horizon ordering are reported; the t-statistics are
not. The 1–15 minute sign agreement across three symbols (passive fills lose) stands as an
observation, not as a certified result.

## Standing rule added

Every null in this repo now carries a calibration test on synthetic data with a known
zero answer, asserting the rejection rate is near nominal alpha. A null that has not been
checked that way is not a null.
