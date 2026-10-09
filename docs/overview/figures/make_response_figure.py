"""Figure: expected signed mid-price path after an up move, model against data (May 2017).

Model: the model's forecast of the mid change over H, given the signed accumulators just after
each clean (E1m) up move (its actual history), averaged over up moves; mean move size of the fit
month. Data: the realized change of the NBBO mid over the same H. Both averaged within each day,
then over days; bars are 2 day-clustered standard errors.

    HAWKES_TAQ=... python docs/overview/figures/make_response_figure.py NBBO_ROOT FIT_DIR
"""
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import stopping  # noqa: E402,F401
from fit_month import month_files  # noqa: E402
from hawkes_taq import events as ev  # noqa: E402
from hawkes_taq import hawkes as hk  # noqa: E402
from stopping.drift import Accumulators, SignedDrift  # noqa: E402

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a63", "#e6e5df"
HS = np.logspace(-3, 2, 21)


def forecast_matrix(dr, H_list):
    """Rows: for each H, the linear map d -> expected displacement over H."""
    lam = dr._lam
    rows = []
    for H in H_list:
        w = (np.exp(lam * H) - 1) / lam
        rows.append(np.real(dr.size * (dr._gV * w) @ dr._Vinv))
    return np.array(rows)          # (nH, K)


def empirical(nbbo_root, sym, drifts, month="201705"):
    """Realized mid change after clean up moves, and each model's forecast given actual history."""
    per_day, fc_day = [], {k: [] for k in drifts}
    Fs = {k: forecast_matrix(dr, HS) for k, dr in drifts.items()}
    for f in month_files([nbbo_root], sym, month):
        st = ev.build_states(pd.read_csv(f), hk.SESSION_END)
        e = ev.events_e1(st, min_state=0.001)
        t_up = e.loc[e["type"] == "up", "t"].to_numpy()
        st_t = st["t"].to_numpy()
        mid = (st["bid"].to_numpy() + st["ask"].to_numpy()) / 2.0      # ticks
        mid_at = lambda s: mid[np.clip(np.searchsorted(st_t, s, side="right") - 1, 0, None)]  # noqa
        t_up = t_up[t_up < hk.SESSION_END - HS[-1]]
        m0 = mid_at(t_up + 1e-7)
        per_day.append([np.mean(mid_at(t_up + H) - m0) for H in HS])
        t_all = e["t"].to_numpy()
        eps = np.where(e["type"].to_numpy() == "up", 1.0, -1.0)
        for k, dr in drifts.items():
            acc = Accumulators(t_all, eps, dr.beta)
            Dst = np.array([acc.state(t + 1e-7) for t in t_up])     # (n, K)
            fc_day[k].append((Fs[k] @ Dst.T).mean(axis=1))
    a = np.array(per_day)
    fc = {k: np.array(v).mean(0) for k, v in fc_day.items()}
    return a.mean(0), a.std(0, ddof=1) / np.sqrt(len(a)), fc


def drift(fitfile):
    obj = pickle.load(open(fitfile, "rb"))
    return SignedDrift.from_fit(obj["fit"], obj.get("size", 1.0))


def panel(ax, title, models, emp, se):
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    ax.errorbar(HS, emp, yerr=2 * se, fmt="o", ms=4.5, color=ORANGE, ecolor=ORANGE, elinewidth=1,
                capsize=0, mec="white", mew=0.8, zorder=3, label="Data: mean mid change (±2 SE)")
    for (lab, m), ls in zip(models.items(), ("-", "--")):
        ax.plot(HS, m, color=BLUE, lw=2, ls=ls, zorder=2, label=lab)
    ax.set_xscale("log")
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.set_xlabel("Horizon H after an up move (seconds)", color=INK, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def main(nbbo_root, fit_dir):
    fit_dir = Path(fit_dir)
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=False)
    dr = {"Model fitted on April (out of sample)": drift(fit_dir / "INTC_E1m_K4_201704.pkl"),
          "Model fitted on May (in sample)": drift(fit_dir / "INTC_E1m_K4_201705.pkl")}
    e, s, fc = empirical(nbbo_root, "INTC", dr)
    panel(axes[0], "INTC, May 2017", fc, e, s)
    dr = {"Model fitted on May (in sample)": drift(fit_dir / "TGT_E1m_K4_201705.pkl")}
    e, s, fc = empirical(nbbo_root, "TGT", dr)
    panel(axes[1], "TGT, May 2017", fc, e, s)
    lo = min(a.get_ylim()[0] for a in axes); hi = max(a.get_ylim()[1] for a in axes)
    for a in axes:
        a.set_ylim(lo, hi)
    axes[0].set_ylabel("Signed mid change (ticks)", color=INK, fontsize=9)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper right")
    axes[1].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    out = Path(__file__).with_name("fig_response.pdf")
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=160)
    print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
