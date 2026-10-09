# Execution timing under multi-timescale Hawkes intensities of mid-price moves

*Research problem statement, v1 (October 2026)*

## 1. Idea

A Hawkes model of mid-price moves is an observable, real-time forecast of the direction of the next moves. A trader who must transact by a deadline faces an optimal stopping problem whose state is that forecast. This project characterizes the optimal rule, measures what timing is worth on consolidated US quote data, and separates two sources of short-horizon predictability: brief consolidated-quote states (a latency race) and the clustering of persistent price moves (latency-free).

It builds on two earlier projects:

- **Lee & Lee (2026), arXiv 2609.36631:** the spread-gated Hawkes-flocking model of best-quote moves, with a single-period limit-order rule whose execution probabilities are inputs. Its conclusion names a multi-period extension; this is one.
- **hawkes-taq:** multi-timescale (K = 4) symmetric Hawkes fits to clean mid-price moves (E1m: states shorter than 1 ms merged, 1 ms dead time) for 8 stocks over 14 months. It finds four timescales (about 10–16 ms, 70–200 ms, 1.5–3 s, 30–40 s), shows that the fitted kernel reproduces the signed post-event price path while the raw-event (E0) fit does not (results section 19), and that kernel mass within 1 s and 10 s is well identified while the total branching ratio is not (section 16).

## 2. Model

Two event types, up (+1) and down (−1), with the symmetric multi-exponential kernel of hawkes-taq:

$$
\lambda_{\pm}(t) = \mu(t) + \sum_k \beta_k \sum_{t_m<t} \big(a_k 1\{\varepsilon_m = \pm 1\} + c_k 1\{\varepsilon_m = \mp 1\}\big) e^{-\beta_k (t-t_m)} .
$$

The baseline is shared by both directions, so it cancels in the signed intensity

$$
D(t) = \lambda_+(t) - \lambda_-(t) = \sum_k g_k\, d_k(t), \qquad g_k = a_k - c_k, \qquad d_k(t) = \beta_k \sum_{t_m<t} \varepsilon_m e^{-\beta_k (t-t_m)} .
$$

- **State.** The signed accumulators $d = (d_1,\dots,d_K)$: a piecewise-deterministic Markov process (exponential decay between events, jump $\beta_k \varepsilon$ at an event). The unsigned accumulators, which carry the total intensity, matter for the variance of the price path but not for its drift.
- **Drift.** With $y(s) = E[d(s)\mid\mathcal F_t]$, $y' = My$, $M = -\mathrm{diag}(\beta) + \beta g^\top$, and the expected displacement over $(t, t+H]$ is $\text{size}\cdot g^\top \!\int_0^H e^{Ms}ds\; d(t)$ (`stopping/drift.py`, checked against simulated cascades in `tests/`).
- **Signed kernel.** $g_k$ is continuation ($>0$) or reversal ($<0$) at timescale $k$. The fits already estimate $a_k$ and $c_k$; reading them as a signed kernel is new. Its total $\sum_k g_k$ is the signed branching ratio.

## 3. Problems

**P1 (marketable order, main).** Buy one unit by $T$. Choose a stopping time $\tau \le T$ to minimize $E[A_\tau]$ (ask at execution), or with a running penalty $\phi$ for waiting. At a one-tick spread $A_\tau - A_t$ equals the mid change, so the problem is $\inf_\tau E\int_t^\tau \text{size}\cdot D(s)\,ds$ (+ penalty), a PDMP optimal stopping problem (Davis 1993; Costa & Davis 1988; Gugerli 1986) with value function given by an iterated single-jump operator.

**P2 (resting order).** A trader with a resting limit order chooses when to cancel and cross. Fill at the best requires queue depletion; the payoff makes the execution probabilities of Lee & Lee (2026) model outputs. Needs best sizes from `complete_nbbo` (one more WRDS pass).

**Latency.** Two delays enter every rule: information delay (an E1m event is confirmed only once its state has lasted h = 1 ms) and execution latency $\ell$. A rule decided on information at $s - \text{lag}$ executes at the ask at $s + \ell$.

## 4. Questions

- **Q1 (structure).** When is the optimal rule monotone in each $d_k$ (a threshold surface)? Conjecture: when the resolvent of the signed kernel has one sign on the deadline horizon; it fails when $g_k$ changes sign across scales. INTC (continuation at 15 ms, reversal beyond) is the monotone case; TGT ($g$ = +, −, 0, +) is a candidate counterexample.
- **Q2 (value).** What is timing worth, in ticks and as a share of the half-spread, by stock, deadline, decision context (random time, after an up move, after a down move) and latency? Out of sample.
- **Q3 (raw against clean).** Rules driven by raw (E0) intensities exploit brief states. How fast does that edge decay with execution latency, and what is left that only the clean kernel provides?
- **Q4 (decision-relevant identification).** The rule with deadline $T$ depends on the kernel mainly through its signed mass within $T$. Is the value of timing robust to the baseline specification (hawkes-taq section 16) where traders care, i.e. for $T$ up to about a minute?
- **Q5 (robustness to misfit).** The clean model is rejected by calibrated KS tests (hawkes-taq section 15). How much of the rule's value survives refitting with alternative specifications (full kernel, power-law truth)?

## 5. Evidence so far

Prototype, INTC and TGT, May 2017: [`prototype_201705.md`](prototype_201705.md). In short: out of sample, a one-step rule driven by the clean kernel saves 0.04 ticks per order at random decision times and 0.15 ticks after up moves (60 s deadline), unchanged by 0–2 ms execution latency and moved by at most 0.02 ticks by the 1 ms confirmation delay; the raw-event rule's larger zero-latency edge disappears by 2 ms.

## 6. Plan

1. Out-of-sample panel: 8 stocks, fit April → test May, fit May → test June, E1m and E0, latencies 0–5 ms (`scripts/cluster/oos_panel.sbatch`).
2. Optimal rule for P1 by dynamic programming over the signed accumulators (K = 4, PDMP), against the one-step rule; how much the one-step rule leaves.
3. Q1: monotonicity results and the TGT counterexample.
4. Extend to all 14 months; add the sell side (mirror image) and a running inventory penalty.
5. P2 after fetching best sizes.

## 7. Risks

- Small economic magnitudes (hundredths of a tick per order unconditionally). The contribution is structure (signed kernel, monotonicity, latency decomposition), not a trading strategy.
- One-unit orders, no impact; the consolidated ask during a lock may not be accessible.
- The model is rejected by calibrated fit tests; Q5 addresses whether that matters for decisions.

## References

- Costa, O. L. V. & Davis, M. H. A. (1988). Approximations for optimal stopping of a piecewise-deterministic process. *Mathematics of Control, Signals and Systems* 1, 123–146.
- Davis, M. H. A. (1993). *Markov Models and Optimization*. Chapman & Hall.
- Gugerli, U. S. (1986). Optimal stopping of a piecewise-deterministic Markov process. *Stochastics* 19, 221–236.
- Lee, H. & Lee, K. (2026). A spread-gated Hawkes-flocking model for best bid and ask dynamics, with an application to limit order placement. arXiv:2609.36631.
- Related, to check in a literature pass: Hawkes-driven optimal execution and market making (Alfonsi & Blanc; Cartea, Jaimungal & Ricci), latency in execution (Lehalle & Mounjid).
