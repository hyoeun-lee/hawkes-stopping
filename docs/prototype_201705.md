# Prototype: INTC and TGT, May 2017

First test of whether timing a buy on Hawkes intensities has value (Oct 8, 2026). Data: saved NBBO changes, 2017Q2 (hawkes-taq format); events and fits from hawkes-taq (E1m, K = 4, 1 ms dead time, full start grid; E0, K = 4, no dead time).

## Setup

A buyer must buy one unit within H of a decision time t. Rules: **now** (buy at t), **wait** (buy at t + H), **rule** (buy at the first check time at which the model's expected mid displacement over the remaining horizon is ≥ 0, else at t + H; `stopping/backtest.py`). The rule is a one-step lookahead, not the optimal stopping rule, so its value is a lower bound.

- **Cost:** quoted NBBO ask at execution minus ask at t, in ticks. Negative = cheaper than buying at t.
- **Decision times:** every 30 s ("random"), just after each up move, just after each down move (moves of the rule's own event definition).
- **Latencies:** clean rules see an E1m event 1 ms after it (confirmation); execution latency ℓ as stated.
- **Uncertainty:** standard errors clustered by day (22 days).

## INTC, out of sample (fit April, test May; E1m, 1 ms confirmation, ℓ = 2 ms)

April fit: memories 16 ms, 0.55 s, 4.6 s, 53 s; signed kernel g = a − c = (+0.060, −0.006, −0.067, −0.035): continuation at the fastest scale, reversal at seconds to a minute.

| Decision time | H | now | wait | rule | rule waits (share of H) |
|---|---|---|---|---|---|
| random | 1 s | −0.000 | −0.002 (0.002) | **−0.005** (0.001) | 0.49 |
| random | 10 s | −0.000 | +0.004 (0.004) | **−0.012** (0.003) | 0.45 |
| random | 60 s | −0.000 | +0.001 (0.014) | **−0.040** (0.011) | 0.33 |
| after up | 10 s | −0.000 | −0.089 (0.021) | **−0.073** (0.019) | 0.63 |
| after up | 60 s | −0.000 | −0.068 (0.039) | **−0.155** (0.028) | 0.44 |
| after down | 10 s | −0.002 | +0.094 (0.019) | **+0.009** (0.003) | 0.05 |
| after down | 60 s | −0.002 | +0.208 (0.038) | **+0.007** (0.007) | 0.06 |

(From `scripts/run_backtest.py`; `results/INTC_E1m_test201705_fit201704_lag1ms_lat2ms.csv`.)

- At random times the rule saves 0.04 ticks per order with a 60 s deadline, 8% of the half-spread. After an up move it saves 0.155 ticks (31%); after a down move it buys at once and avoids the +0.21 ticks that waiting costs.
- **Execution latency does not matter:** with ℓ = 0 instead of 2 ms the clean rule's costs change by less than 0.001 ticks.
- **The confirmation delay matters a little, in both directions.** In sample (fit May, costs on the clean mid in the first prototype run), going from no delay to 1 ms changed the rule's cost from −0.033 to −0.038 at random times (60 s) and from −0.180 to −0.159 after up moves (60 s). The real-time objection (events confirmed only after 1 ms) changes the numbers by up to 0.02 ticks but not the conclusions; it should be reported, with the delay at 1 ms as the main specification.

## Raw-event rule (INTC, E0 fit on May, in sample)

The raw rule sees every NBBO change in real time, including moves into locks. A raw up move (often the bid lifting to the ask) predicts that the ask ticks up within about a millisecond, and the rule buys just before. That edge is a latency race:

| Execution latency ℓ | Raw rule, random times, 60 s | Raw rule, after raw up move, 60 s | Clean rule, random times, 60 s |
|---|---|---|---|
| 0 | −0.188 (0.007) | −0.004 (0.001) | −0.041 |
| 0.5 ms | −0.148 (0.008) | +0.101 (0.005) | — |
| 2 ms | −0.025 (0.008) | **+0.419** (0.005) | −0.042 |

At ℓ = 2 ms the raw rule buys into the move it tried to get ahead of: +0.42 ticks after a raw up move. Short-horizon predictability splits into a latency race in the brief states (gone by about 2 ms) and a structural part in the clean kernel (latency-free).

## TGT (E1m, fit May, in sample, ℓ = 2 ms)

May fit: memories 5 ms, 42 ms, 0.87 s, 10 s; g = (+0.036, −0.101, −0.000, +0.069). The sign changes twice: continuation at 5 ms, reversal at 40 ms, continuation at 10 s.

| Decision time | H | now | wait | rule |
|---|---|---|---|---|
| after up | 0.1 s | +0.002 | −0.051 (0.015) | **−0.066** (0.018) |
| after up | 10 s | +0.002 | +0.033 (0.023) | **−0.059** (0.022) |
| after up | 60 s | +0.002 | +0.090 (0.061) | **−0.025** (0.014) |
| random | 60 s | +0.000 | +0.026 (0.026) | **−0.033** (0.015) |

After an up move with a 10 s deadline the rule waits out the 40 ms reversal and buys before the slow continuation. Under a sign-changing kernel the optimal rule need not be monotone in the state (problem statement Q1). To be confirmed out of sample.

## One timescale against several (INTC, out of sample, ℓ = 2 ms)

The *instant* rule stops when the current signed intensity D = g'd is ≥ 0, ignoring how the components decay (the rule a one-timescale model gives). Same fit and setting as the INTC table above (`results/INTC_E1m_test201705_fit201704_lag1ms_lat2ms_instant.csv`).

| Decision time, H | One-step rule | Instant rule |
|---|---|---|
| random, 10 s | −0.012 (0.003) | −0.011 (0.004) |
| random, 60 s | −0.040 (0.011) | −0.026 (0.008) |
| after up, 10 s | −0.073 (0.019) | −0.000 (0.001) |
| after up, 60 s | −0.155 (0.028) | −0.000 (0.001) |
| after down, 10 s | +0.009 (0.003) | +0.027 (0.007) |
| after down, 60 s | +0.007 (0.007) | +0.073 (0.015) |

After an up move the fast continuation dominates D, so the instant rule buys at once and misses the slow reversal; after a down move it waits and pays for it.

## Correction (Oct 9)

TGT is not a counterexample to monotonicity of the rule in the state: the one-step rule is monotone in each signed accumulator for TGT as for INTC (overview, Proposition 4). TGT shows that the decision after an event is not monotone in the deadline.

## Caveats

- One month; one-unit orders; no impact; the consolidated ask during a lock may not be accessible.
- The rule is one-step lookahead; the optimal rule's value is at least as large.
- The prototype used the test month's mean move size; the package uses the fit month's. Decisions depend only on the sign of the expected displacement, so this does not change them.
- In the package, a check after an event is made once the event is confirmed (event + lag); the prototype checked at the event with lagged information. Differences are under 0.005 ticks.
