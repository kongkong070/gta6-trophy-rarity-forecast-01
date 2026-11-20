"""GTA VI trophy rarity forecast (gta6-trophy-rarity-forecast-01).

Forecasts every trophy's Exophase earn rate at a case's target time.

Model idea
----------
A trophy's rate is earners / players-tracked, and both sides move:

* players_tracked N(t) grows as Exophase picks up more players.  On the
  launch-night recording N is a staircase: flat between Exophase's own
  refreshes, then a jump.  The underlying growth is modelled as a power
  law in time since release, ln N = a + b * ln(1 + hours since release),
  fitted to the distinct refresh snapshots (first appearance of each
  distinct value), with the slope shrunk toward a prior when few distinct
  snapshots exist.
* earners E(t) = rate * N / 100 per trophy follow the same shape,
  ln E = a_i + g_i * ln(1 + hours since release), fitted per trophy with
  hierarchical shrinkage toward the cross-trophy mean slope.  Story /
  early trophies earn fast (large g), late grind trophies slowly.
* The forecast rate is E(target) / N(target).  When the history is
  completely flat (no refresh seen yet, as in the practice cases) the
  priors do the work: N grows with slope ~1 in log-log time while
  earners grow much more slowly, so rates are forecast to fall sharply
  at the next refresh -- the dilution every early refresh produces --
  instead of staying put.

Spread
------
Quantiles are normal on the logit scale around the forecast median.
The logit spread combines a base refresh-step uncertainty, the
prediction variance of the two log-log fits (which grows as the target
moves away from the fitted x-range), and a term that widens with the
horizon in days, so a 4.5-day forecast is honestly much wider than a
90-minute one.  The 100% trophy is handled separately: it stays near
100% with a one-sided-ish band below it.

Only the standard library is used.  No network, no randomness, no
clock reads: the same input always produces byte-identical output.
"""

import json
import math
import os
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# Release of GTA VI: 2026-11-19T00:00:00Z.  Time is measured in hours
# since then; x = ln(1 + hours) is the regressor for both fits.
RELEASE_EPOCH = 1795046400  # 2026-11-19T00:00:00Z as a Unix timestamp

# Priors used when the history has too few distinct snapshots to fit.
PRIOR_N_SLOPE = 1.0       # players-tracked grows ~linearly in (1+t) early
PRIOR_N_SLOPE_W = 2.0     # pseudo-count weight of that prior
PRIOR_E_SLOPE = 0.15      # earners grow slowly in log-log time early on
PRIOR_E_SLOPE_W = 2.0
MAX_N_SLOPE = 1.6
MAX_E_SLOPE = 2.0

BASE_SIGMA = 0.45         # logit spread: refresh-step / model uncertainty
HORIZON_SIGMA = 0.30      # extra logit spread per sqrt(day of horizon)
RESID_SIGMA = 0.45        # assumed residual std when a fit has <3 points


def parse_time(value):
    """ISO-8601 UTC timestamp -> Unix seconds (stdlib only)."""
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    from datetime import datetime
    return datetime.fromisoformat(text).timestamp()


def clamp_pct(p):
    return min(99.99, max(0.01, float(p)))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(l):
    return 100.0 / (1.0 + math.exp(-l))


def round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def ols(xs, ys, weights=None):
    """Least squares y ~ a + b*x.  Returns (a, b, resid_var, sxx, xbar, n)."""
    n = len(xs)
    if weights is None:
        weights = [1.0] * n
    sw = sum(weights)
    xbar = sum(w * x for w, x in zip(weights, xs)) / sw
    ybar = sum(w * y for w, y in zip(weights, ys)) / sw
    sxx = sum(w * (x - xbar) ** 2 for w, x in zip(weights, xs))
    if sxx <= 1e-12:
        return ybar, 0.0, 0.0, 0.0, xbar, n
    sxy = sum(w * (x - xbar) * (y - ybar)
              for w, x, y in zip(weights, xs, ys))
    b = sxy / sxx
    a = ybar - b * xbar
    resid = sum(w * (y - (a + b * x)) ** 2
                for w, x, y in zip(weights, xs, ys))
    dof = max(1, n - 2)
    return a, b, resid / dof, sxx, xbar, n


def shrink(value, prior, weight, n):
    return (n * value + weight * prior) / (n + weight)


def distinct_snapshots(history, n_trophies):
    """Collapse consecutive identical reads to refresh snapshots.

    Returns a list of (x, t_hours, players, rates) where x =
    ln(1+hours since release) at the FIRST read showing that distinct
    value -- the refresh that produced it -- and rates are the read's
    rates.
    """
    snaps = []
    prev_key = None
    for read in history:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"), tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        t_hours = (parse_time(read["at"]) - RELEASE_EPOCH) / 3600.0
        x = math.log(max(0.5, 1.0 + t_hours))
        snaps.append((x, t_hours, read.get("players_tracked"), list(rates)))
    return snaps


def recency_weights(t_hours_list):
    """Weights for the fits: recent snapshots matter most, because
    adoption slopes flatten as the launch rush passes.  Half-life
    about 25 hours."""
    t_last = max(t_hours_list) if t_hours_list else 0.0
    return [math.exp(-(t_last - t) / 36.0) for t in t_hours_list]


def forecast_case(case):
    trophies = case["trophies"]
    history = case["history"]
    if not history:
        raise ValueError("case has no history")
    horizon_days = case["horizon_minutes"] / 1440.0
    target_hours = (parse_time(case["target"]) - RELEASE_EPOCH) / 3600.0
    x_target = math.log(max(0.5, 1.0 + target_hours))
    n_tr = len(trophies)

    snaps = distinct_snapshots(history, n_tr)
    last = history[-1]
    last_rates = list(last["rates"])
    last_n = last.get("players_tracked") or 0

    # ---- players-tracked fit: ln N ~ a + b*x on distinct snapshots ----
    n_xs, n_ys, n_ts = [], [], []
    for x, t_h, players, _rates in snaps:
        if players and players > 0:
            n_xs.append(x)
            n_ys.append(math.log(players))
            n_ts.append(t_h)
    if len(n_xs) >= 2:
        a_n, b_n, var_n, sxx_n, xbar_n, _ = ols(
            n_xs, n_ys, weights=recency_weights(n_ts))
        b_n = shrink(b_n, PRIOR_N_SLOPE, PRIOR_N_SLOPE_W, len(n_xs))
        b_n = min(MAX_N_SLOPE, max(0.0, b_n))
        # Re-anchor the intercept at the last snapshot so the level is
        # exact at the most recent refresh, keeping the fitted slope.
        a_n = n_ys[-1] - b_n * n_xs[-1]
        resid_n = math.sqrt(var_n) if var_n > 0 else RESID_SIGMA
        lev_n = ((x_target - xbar_n) ** 2 / sxx_n) if sxx_n > 1e-12 else 1.0
    else:
        b_n = PRIOR_N_SLOPE
        base_x = n_xs[0] if n_xs else math.log(max(0.5, 1.0))
        base_y = n_ys[0] if n_ys else math.log(max(1, last_n))
        a_n = base_y - b_n * base_x
        resid_n = RESID_SIGMA
        lev_n = 1.0
    log_n_target = a_n + b_n * x_target
    # Never forecast fewer tracked players than the last read, and cap
    # runaway growth at 8x the last count.
    log_n_target = min(math.log(max(1, last_n) * 8.0),
                       max(math.log(max(1, last_n)), log_n_target))

    # ---- per-trophy earner fits: ln E ~ a + g*x ----------------------
    # First pass: raw slopes, to get the cross-trophy mean for shrinkage.
    raw = []
    for i in range(n_tr):
        xs, ys, ts = [], [], []
        for x, t_h, players, rates in snaps:
            r = rates[i] if i < len(rates) else None
            if r is None or players is None or players <= 0:
                continue
            earners = max(0.5, r * players / 100.0)
            xs.append(x)
            ys.append(math.log(earners))
            ts.append(t_h)
        if len(xs) >= 2:
            _a, g, _v, _s, _xb, _n = ols(xs, ys, weights=recency_weights(ts))
            raw.append((i, xs, ys, ts, g))
    if raw:
        mean_g = sum(g for _i, _x, _y, _t, g in raw) / len(raw)
        mean_g = min(MAX_E_SLOPE, max(0.0, mean_g))
    else:
        mean_g = PRIOR_E_SLOPE
    prior_g = shrink(mean_g, PRIOR_E_SLOPE, 1.0, len(raw))

    horizon_sigma = HORIZON_SIGMA * math.sqrt(horizon_days)
    rows = []
    for i, trophy in enumerate(trophies):
        r_last = last_rates[i] if i < len(last_rates) else None
        if r_last is None:
            r_last = 0.5
        # The 100% trophy: every tracked player earned it, so it stays
        # at the top; only a band below it is honest.
        if r_last >= 99.99:
            qs = [96.0, 98.5, 99.5, 99.9, 99.99]
            rows.append({"id": trophy["id"], "q": qs})
            continue

        xs, ys, ts = [], [], []
        for x, t_h, players, rates in snaps:
            r = rates[i] if i < len(rates) else None
            if r is None or players is None or players <= 0:
                continue
            earners = max(0.5, r * players / 100.0)
            xs.append(x)
            ys.append(math.log(earners))
            ts.append(t_h)
        if len(xs) >= 2:
            a_e, g_e, var_e, sxx_e, xbar_e, _n = ols(
                xs, ys, weights=recency_weights(ts))
            g_e = shrink(g_e, prior_g, PRIOR_E_SLOPE_W, len(xs))
            g_e = min(MAX_E_SLOPE, max(0.0, g_e))
            # Adoption slopes flatten with time (a trophy's earners
            # saturate as the reachable players earn it), so damp the
            # slope over the extrapolation gap to the target.
            gap = max(0.0, x_target - xs[-1])
            g_e = g_e / (1.0 + 0.25 * gap)
            a_e = ys[-1] - g_e * xs[-1]
            resid_e = math.sqrt(var_e) if var_e > 0 else RESID_SIGMA
            lev_e = ((x_target - xbar_e) ** 2 / sxx_e) if sxx_e > 1e-12 else 1.0
        else:
            g_e = prior_g
            gap = max(0.0, x_target - (xs[-1] if xs else 0.0))
            g_e = g_e / (1.0 + 0.25 * gap)
            base_x = xs[0] if xs else math.log(max(0.5, 1.0))
            base_y = ys[0] if ys else math.log(max(0.5, r_last * max(1, last_n) / 100.0))
            a_e = base_y - g_e * base_x
            resid_e = RESID_SIGMA
            lev_e = 1.0
        log_e_target = a_e + g_e * x_target
        log_rate = log_e_target - log_n_target
        median_en = clamp_pct(100.0 * math.exp(log_rate))

        # Direct fit: logit(rate) against x over the same snapshots,
        # recency-weighted and anchored at the last snapshot.  This is
        # the net drift of the rate itself (adoption minus dilution);
        # it is the strongest signal once several refreshes have been
        # seen, and uninformative when the history is flat, so its
        # weight grows with the number of distinct snapshots.
        r_xs, r_ys, r_ts = [], [], []
        for x, t_h, _players, rates in snaps:
            r = rates[i] if i < len(rates) else None
            if r is None:
                continue
            r_xs.append(x)
            r_ys.append(logit(r))
            r_ts.append(t_h)
        # Fit the slope on the last 24h of snapshots when there are
        # enough of them: the launch rush makes early slopes far
        # steeper than the recent trend, and the recent trend is what
        # the next days follow.
        if len(r_xs) >= 2:
            t_last_r = r_ts[-1]
            win = [(x, y) for x, y, t in zip(r_xs, r_ys, r_ts)
                   if t_last_r - t <= 24.0]
            if len(win) >= 4:
                wx = [p[0] for p in win]
                wy = [p[1] for p in win]
                _ar, b_r, _vr, _sr, _xbr, _nr = ols(wx, wy)
            else:
                _ar, b_r, _vr, _sr, _xbr, _nr = ols(
                    r_xs, r_ys, weights=recency_weights(r_ts))
            b_r = shrink(b_r, 0.0, 2.0, len(r_xs))
            med_direct = clamp_pct(inv_logit(r_ys[-1] + b_r * (x_target - r_xs[-1])))
        else:
            med_direct = r_last
        w_direct = min(0.6, len(snaps) / (len(snaps) + 6.0))
        median = clamp_pct(math.exp(
            (1.0 - w_direct) * math.log(median_en)
            + w_direct * math.log(med_direct)))

        # Spread: base refresh uncertainty + fit prediction variance
        # (both fits, growing with distance from the fitted range) +
        # a horizon term.
        fit_var = (resid_e ** 2) * (1.0 + lev_e) + (resid_n ** 2) * (1.0 + lev_n)
        sigma = math.sqrt(BASE_SIGMA ** 2 + fit_var + horizon_sigma ** 2)
        sigma = min(2.5, max(0.35, sigma))
        med_logit = logit(median)
        qs = []
        for z in Z:
            q = inv_logit(med_logit + z * sigma)
            qs.append(round4(min(99.99, max(0.01, q))))
        # Keep quantiles non-decreasing after rounding.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = sorted(INPUT_DIR.glob("*.in"))
    for path in inputs:
        with open(path, "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_path = OUTPUT_DIR / (path.stem + ".out")
        text = json.dumps(result, separators=(",", ":"), sort_keys=False)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
