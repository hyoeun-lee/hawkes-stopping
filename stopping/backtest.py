"""Backtest of execution-timing rules for a buyer with a deadline, on saved NBBO changes.

A buyer must buy one unit within H seconds of a decision time t. Rules:

- now:   buy at t;
- wait:  buy at t + H;
- model: buy at the first check time s in [t, t + H] at which the expected signed
         displacement of the mid over the remaining horizon (s, t + H] is >= 0 (the price is not
         expected to fall further), otherwise at t + H. Check times are t, every event of the
         model's definition in (t, t + H), and a grid of n_grid points. This is a one-step
         lookahead rule, not the optimal stopping rule, so its value is a lower bound.

Cost = quoted NBBO ask at execution (+ execution latency) minus the ask at t, in ticks.
Positive = paid more than at t.

Two latencies:
- info_lag: the state at s uses only events at or before s - info_lag. E1m events are
  confirmed only after their new state has lasted h (1 ms), so clean rules use info_lag = h.
- exec_lat: an order decided at s executes at the ask quoted at s + exec_lat.

Decision sets: "grid" (every grid_step seconds), "after_up" and "after_down" (just after each
event of the model's definition, plus info_lag).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hawkes_taq import events as ev
from hawkes_taq import hawkes as hk

from .drift import Accumulators, SignedDrift

START, END = hk.SESSION_START, hk.SESSION_END
HORIZONS = (0.1, 1.0, 10.0, 60.0)


@dataclass
class DayData:
    label: str
    ev_t: np.ndarray      # model events, seconds from session start
    ev_eps: np.ndarray    # +1 / -1
    ev_size: np.ndarray   # ticks
    ask_t: np.ndarray     # NBBO state start times, seconds from session start
    ask: np.ndarray       # best ask in ticks


def load_day(path, definition="E1m", min_state=0.001):
    """One saved NBBO file (columns t, bid, ask) -> DayData for the given event definition."""
    st = ev.build_states(pd.read_csv(path), END)
    if definition == "E0":
        e = ev.events_e0(st)
    elif definition == "E1m":
        e = ev.events_e1(st, min_state=min_state)
    else:
        raise ValueError(definition)
    e = e.sort_values("t").reset_index(drop=True)
    return DayData(label=path.name[:8],
                   ev_t=e["t"].to_numpy() - START,
                   ev_eps=np.where(e["type"].to_numpy() == "up", 1.0, -1.0),
                   ev_size=e["size"].to_numpy() / 2.0,
                   ask_t=st["t"].to_numpy() - START,
                   ask=st["ask"].to_numpy().astype(float))


def _price(t_arr, v_arr, s):
    j = np.searchsorted(t_arr, s, side="right") - 1
    return v_arr[max(j, 0)]


def run_day(day, drift, decisions, horizons=HORIZONS, n_grid=20, info_lag=0.0, exec_lat=0.0):
    """Costs of now / wait / model for each decision time and horizon.

    Returns {H: dict of arrays now, wait, model, waited (fraction of H waited by model)}.
    """
    acc = Accumulators(day.ev_t, day.ev_eps, drift.beta)
    ask = lambda s: _price(day.ask_t, day.ask, s)  # noqa: E731
    out = {}
    for H in horizons:
        n = len(decisions)
        now, wait, model, waited = (np.zeros(n) for _ in range(4))
        for q, t in enumerate(decisions):
            a0 = ask(t)
            now[q] = ask(t + exec_lat) - a0
            wait[q] = ask(t + H + exec_lat) - a0
            i0 = np.searchsorted(day.ev_t, t, side="right")
            i1 = np.searchsorted(day.ev_t, t + H, side="left")
            checks = np.sort(np.concatenate([[t], day.ev_t[i0:i1] + info_lag + 1e-9,
                                             t + H * np.arange(1, n_grid) / n_grid]))
            checks = checks[checks <= t + H]
            tau = t + H
            for s in checks:
                if drift.expected(acc.state(s - info_lag), t + H - s) >= 0:
                    tau = s
                    break
            model[q] = ask(tau + exec_lat) - a0
            waited[q] = (tau - t) / H
        out[H] = {"now": now, "wait": wait, "model": model, "waited": waited}
    return out


def decision_sets(day, grid_step=30.0, margin=61.0, info_lag=0.0):
    T = END - START
    late = day.ev_t < T - margin
    shift = max(info_lag, 1e-6)
    return {"grid": np.arange(60.0, T - margin, grid_step),
            "after_up": day.ev_t[(day.ev_eps > 0) & late] + shift,
            "after_down": day.ev_t[(day.ev_eps < 0) & late] + shift}


def backtest(days, drift, horizons=HORIZONS, info_lag=0.0, exec_lat=0.0, grid_step=30.0,
             log=None):
    """Per-day mean costs, one row per (day, decision set, horizon)."""
    rows = []
    for day in days:
        for name, dec in decision_sets(day, grid_step, info_lag=info_lag).items():
            res = run_day(day, drift, dec, horizons, info_lag=info_lag, exec_lat=exec_lat)
            for H in horizons:
                r = res[H]
                rows.append({"day": day.label, "set": name, "H": H, "n": len(dec),
                             **{k: float(v.mean()) if len(v) else np.nan for k, v in r.items()}})
        if log:
            log(f"done {day.label}")
    return pd.DataFrame(rows)


def summarize(per_day):
    """Event-weighted means over days, with day-clustered standard errors."""
    out = []
    for (name, H), g in per_day.groupby(["set", "H"]):
        w = g["n"].to_numpy(dtype=float)
        row = {"set": name, "H": H, "n": int(w.sum())}
        for col in ("now", "wait", "model"):
            v = g[col].to_numpy()
            m = float(np.average(v, weights=w))
            row[col] = m
            row[col + "_se"] = float(np.sqrt(np.sum((w * (v - m)) ** 2)) / w.sum())
        row["waited"] = float(np.average(g["waited"], weights=w))
        out.append(row)
    return pd.DataFrame(out)
