# hawkes-stopping

When to execute: optimal timing of an order under multi-timescale Hawkes intensities of mid-price moves, on WRDS Millisecond TAQ data.

- [`docs/problem_statement.md`](docs/problem_statement.md): idea, model, questions and plan
- [`docs/prototype_201705.md`](docs/prototype_201705.md): first results (INTC, TGT, May 2017)
- [`stopping/drift.py`](stopping/drift.py): signed kernel g = a − c, signed accumulators, expected displacement in closed form
- [`stopping/backtest.py`](stopping/backtest.py): timing rules (now, wait, model) and their cost on the quoted ask, with information delay and execution latency
- [`scripts/fit_month.py`](scripts/fit_month.py), [`scripts/run_backtest.py`](scripts/run_backtest.py): fit one stock-month; backtest on another
- [`scripts/cluster/oos_panel.sbatch`](scripts/cluster/oos_panel.sbatch): out-of-sample panel, 8 stocks, 2017Q2

## Dependency

Event construction (E0, E1m) and fitting come from hawkes-taq, which is not packaged. Clone it next to this repository (`../hawkes-taq`) or set `HAWKES_TAQ=/path/to/hawkes-taq`; `stopping/__init__.py` adds it to the path.

## Workflow

Input: saved NBBO changes in the hawkes-taq format (`<root>/<SYM>/<YYYYMMDD>.csv.gz`, columns `t`, `bid`, `ask`).

```
python scripts/fit_month.py --nbbo nbbo/2017Q2 --sym INTC --month 201704 --definition E1m --out fits
python scripts/run_backtest.py --nbbo nbbo/2017Q2 --sym INTC --test-month 201705 \
    --fit fits/INTC_E1m_K4_201704.pkl --exec-lat 0.002 --out results
```

E1m fits use a 1 ms dead time and E1m rules a 1 ms information delay by default. NBBO files, events and fits are TAQ-derived and stay out of the repository (WRDS terms); summary CSVs in `results/` are aggregates.

## Tests

```
python -m pytest tests
```
