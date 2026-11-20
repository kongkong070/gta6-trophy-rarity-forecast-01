# GTA VI Trophy Rarity Forecast — gta6-trophy-rarity-forecast-01

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/kongkong070/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/kongkong070/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every GTA VI (PS5) trophy's Exophase earn rate at a case's
target time. Entry point: `main.py` (Python 3.13, standard library only,
no network, deterministic).

## Model

A trophy's rate is earners / players-tracked, and both sides move, so
they are forecast separately and combined:

* **Players tracked N(t):** power law in time since release,
  ln N = a + b·ln(1 + hours since 2026-11-19T00:00Z), fitted to the
  distinct refresh snapshots (first appearance of each distinct read),
  recency-weighted, slope shrunk toward a prior of ~1.0.
* **Earners E(t):** per trophy, ln E = a + g·ln(1 + hours), same
  recency weighting, slope shrunk toward the cross-trophy mean and
  damped over the extrapolation gap (adoption saturates).
* **Direct rate fit:** logit(rate) against ln(1 + hours), slope fitted
  on the last 24 h of snapshots when there are enough of them (the
  launch rush makes early slopes far steeper than the recent trend).
  Blended with the E/N forecast in log space; its weight grows with
  the number of distinct snapshots, so it is ~silent on the flat
  practice histories and dominant on a rich final history.
* **Flat-history priors do the work on practice:** with no refresh
  seen, N is forecast to grow ~6× to the next refresh while earners
  grow only ~30%, so rates are forecast to fall to ~20% of their last
  read instead of staying put — the dilution every early refresh
  produces. The 100% trophy is kept near 100% with a band below it.
* **Spread:** normal on the logit scale; σ combines a base refresh
  uncertainty (0.45), the prediction variance of the fits, and a
  horizon term (0.30 per √day), so a 4.5-day forecast is honestly
  much wider than a 90-minute one (σ ≈ 0.9 logit at the final).

## Local evaluation

Practice answers are not published, so they were reconstructed from
the one hard fact the rules give — the house baseline scores ~1.51 —
which pins the refresh's average rate factor at k ≈ 0.20 (earners
+20% while tracked players went 638 → 3,868). Three scenarios were
scored: S1 uniform k=0.20, S2 per-trophy earner growth 1.0–1.5 by
rate level, S3 earners unchanged (k=0.165).

| case | model pinball (S1/S2/S3) | baseline | typical miss | 90% coverage |
|------|--------------------------|----------|--------------|--------------|
| p00  | 0.270 / 0.285 / 0.308    | 1.50 / 1.57 / 1.69 | 0.03–0.07 pts | 0.98 |
| p01  | 0.257 / 0.272 / 0.295    | 1.50 / 1.57 / 1.69 | 0.03–0.06 pts | 0.98 |

Synthetic final-like cases (2 days of 30-min reads with 3-hourly
refreshes, saturating adoption curves, power-law and saturating
player growth, 4.5-day horizon): model pinball 0.18–0.19 vs house
baseline 0.33–0.46 and house curve baseline 0.23–0.25; typical miss
~1.0–1.3 pts. Held-out pseudo-cases cut from a synthetic history
(24 h → 36 h, 36 h → 48 h): model 0.143 / 0.117 vs baseline
0.241 / 0.156 (curve baseline wins one, loses the other badly,
0.496 — the model is the robust middle).

## Checks

* Two runs byte-identical; runtime ~0.07 s per practice case
  (limit 8 s); outputs ~2.8 KB (limit 65,536); 55 rows, 5
  non-decreasing quantiles in [0, 100]; single-read and null-rate
  edge cases handled.

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
