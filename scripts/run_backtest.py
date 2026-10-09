"""Backtest the timing rules on one stock-month with a fit from another (or the same) month.

    python scripts/run_backtest.py --nbbo nbbo/2017Q2 --sym INTC --test-month 201705 \\
        --fit fits/INTC_E1m_K4_201704.pkl --info-lag 0.001 --exec-lat 0.002 --out results

The event definition is read from the fit file name (E0 or E1m). Writes per-day costs and a
summary CSV to --out (aggregates only; safe to commit).
"""

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stopping.backtest import backtest, load_day, summarize  # noqa: E402
from stopping.drift import SignedDrift  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_month import month_files  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nbbo", nargs="+", required=True)
    p.add_argument("--sym", required=True)
    p.add_argument("--test-month", required=True)
    p.add_argument("--fit", required=True)
    p.add_argument("--info-lag", type=float, default=None,
                   help="seconds; default 0.001 for E1m (event confirmation), 0 for E0")
    p.add_argument("--exec-lat", type=float, default=0.0)
    p.add_argument("--grid-step", type=float, default=30.0)
    p.add_argument("--rule", choices=["lookahead", "instant"], default="lookahead",
                   help="lookahead: expected displacement to the deadline >= 0; instant: current "
                        "signed intensity >= 0 (one-timescale comparison)")
    p.add_argument("--out", default="results")
    a = p.parse_args()

    definition = "E0" if "_E0_" in Path(a.fit).name else "E1m"
    info_lag = a.info_lag if a.info_lag is not None else (0.001 if definition == "E1m" else 0.0)
    with open(a.fit, "rb") as fh:
        obj = pickle.load(fh)
    drift = SignedDrift.from_fit(obj["fit"], obj["size"])
    days = [load_day(f, definition) for f in month_files(a.nbbo, a.sym, a.test_month)]
    per_day = backtest(days, drift, info_lag=info_lag, exec_lat=a.exec_lat,
                       grid_step=a.grid_step, log=lambda m: print(m, flush=True), rule=a.rule)
    summ = summarize(per_day)
    tag = (f"{a.sym}_{definition}_test{a.test_month}_fit{Path(a.fit).stem.split('_')[-1]}"
           f"_lag{info_lag * 1e3:g}ms_lat{a.exec_lat * 1e3:g}ms"
           + ("" if a.rule == "lookahead" else f"_{a.rule}"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    per_day.to_csv(out / f"{tag}_per_day.csv", index=False)
    summ.to_csv(out / f"{tag}.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"g = a - c = {drift.g.round(4)}, size {drift.size:.3f}")
    print(summ.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
