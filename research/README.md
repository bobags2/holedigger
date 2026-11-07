# research/

Exploratory / pre-registered strategy investigations.

**Everything here is exploratory.** Files in `research/` are not covered by the permanent test
suite, and their results are not promoted into `artifacts/` until they survive the same
discipline as anything else: a frozen specification written before the first result is read,
a control that measures something, and a cost charged at the instrument's own venue.

## Contents

- `p1_vwap_nas100.py` — VWAP 2σ fade on the account's own NAS100 M1 panel, dev/val/holdout
  split, real contract volume, measured venue cost. Result in the accompanying design note.
  **Rejected as specified**: the reversion leg earns +37.14 bp at 81.8% win rate when the
  target is reached, but 29% of signals never revert and exit at the session close for
  −140.42 bp, and the overall expectancy is −14.60 bp (dev) / −27.26 (validation).

## Why the negative results stay

A frozen spec that fails is the deliverable. Two of this file's defects produced *plausible*
numbers — a 99.3% win rate with a profit factor of 3×10⁴, and a mirror control that returned
exactly −(P&L) and therefore measured nothing — and neither would have been caught by looking
at the output. That is the same class of failure the permanent suite was built to catch, and
the record of it is worth more than the negative result it replaced.
