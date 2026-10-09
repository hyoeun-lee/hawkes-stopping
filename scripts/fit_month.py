"""Fit the symmetric K-exponential Hawkes model to one stock-month and save the fit.

Uses hawkes-taq's event construction and fitter. Output: <out>/<SYM>_<DEF>_K<K>_<MONTH>.pkl,
containing {"fit": HawkesFit, "size": mean move size in ticks}. Fits are TAQ-derived and stay
out of the repository.

    python scripts/fit_month.py --nbbo nbbo/2017Q2 --sym INTC --month 201704 --definition E1m
"""

import argparse
import os
import pickle
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stopping  # noqa: E402,F401  (locates hawkes-taq)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hawkes_taq import events as ev  # noqa: E402
from hawkes_taq import hawkes as hk  # noqa: E402
from hawkes_taq.pipeline import HALF_DAYS  # noqa: E402


def month_files(roots, sym, month):
    files = []
    for r in roots:
        files += [f for f in Path(r, sym).glob(f"{month}*.csv.gz") if f.name[:8] not in HALF_DAYS]
    return sorted(files, key=lambda f: f.name)


def build_days(files, definition, min_state=0.001):
    days, sizes = [], []
    for f in files:
        s = ev.build_states(pd.read_csv(f), hk.SESSION_END)
        e = ev.events_e0(s) if definition == "E0" else ev.events_e1(s, min_state=min_state)
        e["t_fit"] = ev.spread_ties(e["t"])
        d = hk.day_from_events(e, label=f.name[:8])
        if len(d.t):
            days.append(d)
            sizes.append(e["size"].to_numpy() / 2.0)
    return days, float(np.mean(np.concatenate(sizes)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nbbo", nargs="+", required=True)
    p.add_argument("--sym", required=True)
    p.add_argument("--month", required=True)
    p.add_argument("--definition", choices=["E0", "E1m"], default="E1m")
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--delta", type=float, default=None,
                   help="dead time, seconds (default 0.001 for E1m, 0 for E0)")
    p.add_argument("--out", default="fits")
    a = p.parse_args()

    delta = a.delta if a.delta is not None else (0.001 if a.definition == "E1m" else 0.0)
    days, size = build_days(month_files(a.nbbo, a.sym, a.month), a.definition)
    print(f"{a.sym} {a.month} {a.definition}: {len(days)} days, "
          f"{sum(len(d.t) for d in days)} events, mean size {size:.3f} ticks", flush=True)
    starts = hk.full_starts(a.K)
    t0 = time.time()
    f = hk.fit(days, K=a.K, starts=starts, delta=delta)
    print(f"{(time.time() - t0) / 60:.1f} min; memories {np.round(f.timescales(), 4)}; "
          f"a - c = {np.round(f.alpha[0] - f.alpha[1], 4)}", flush=True)
    out = Path(a.out) / f"{a.sym}_{a.definition}_K{a.K}_{a.month}.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump({"fit": f, "size": size}, fh)
    print("saved", out)


if __name__ == "__main__":
    main()
