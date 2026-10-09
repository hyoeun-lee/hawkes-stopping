"""Expected signed mid-price displacement under the symmetric up/down multi-kernel Hawkes model.

Model (as in hawkes_taq.hawkes, kernel="symmetric"): two event types, up (+1) and down (-1),

    lambda_up(t)   = mu(t) + sum_k beta_k sum_m (a_k 1{eps_m=+1} + c_k 1{eps_m=-1}) e^{-beta_k (t-t_m)}
    lambda_down(t) = mu(t) + sum_k beta_k sum_m (c_k 1{eps_m=+1} + a_k 1{eps_m=-1}) e^{-beta_k (t-t_m)}

The baseline is shared by both directions, so it cancels in the signed intensity

    D(t) = lambda_up - lambda_down = sum_k g_k d_k(t),   g_k = a_k - c_k,
    d_k(t) = beta_k sum_{t_m < t} eps_m e^{-beta_k (t - t_m)}   (signed accumulators).

Between events d_k decays at rate beta_k; at an event of sign eps, d_k jumps by beta_k eps.
Taking expectations, y_k(s) = E[d_k(s) | F_t] solves the linear ODE

    y' = M y,   M = -diag(beta) + beta g^T,

and the expected signed number of moves over (t, t+H] is g^T int_0^H e^{M s} ds d(t).
Multiplying by the mean move size (ticks) gives the expected displacement in ticks.

The refractory period (dead time) of the E1m fits is ignored here; its effect on the
realized kernel mass is below 1% for the core stocks (hawkes-taq, results section 10).
"""

import numpy as np


class SignedDrift:
    """Expected signed displacement implied by a fitted symmetric kernel.

    beta: decay rates (K,); a, c: self- and cross-excitation per component (K,);
    size: mean move size in ticks (about 0.95 for E1m, 0.5 for raw E0 events).
    """

    def __init__(self, beta, a, c, size=1.0):
        self.beta = np.asarray(beta, dtype=float)
        self.g = np.asarray(a, dtype=float) - np.asarray(c, dtype=float)
        self.size = float(size)
        self.K = len(self.beta)
        M = -np.diag(self.beta) + np.outer(self.beta, self.g)
        lam, V = np.linalg.eig(M)
        self._lam, self._Vinv = lam, np.linalg.inv(V)
        self._gV = self.g @ V

    @classmethod
    def from_fit(cls, fit, size=1.0):
        """From a hawkes_taq.hawkes.HawkesFit with kernel="symmetric"."""
        if fit.kernel != "symmetric":
            raise ValueError("SignedDrift needs a symmetric kernel")
        a, c = fit.alpha
        return cls(fit.beta, a, c, size)

    @property
    def signed_branching(self):
        """Sum of g_k: net continuation (+) or reversal (-) per move, all horizons."""
        return float(self.g.sum())

    def expected(self, d, H):
        """E[mid(t+H) - mid(t) | signed accumulators d at t], in ticks."""
        lam = self._lam
        small = np.abs(lam) < 1e-12
        w = np.where(small, H, (np.exp(lam * H) - 1) / np.where(small, 1.0, lam))
        return float(np.real(self.size * (self._gV * w) @ (self._Vinv @ np.asarray(d))))

    def after_event(self, H, sign=+1):
        """Expected displacement over H right after an isolated event of the given sign."""
        return self.expected(sign * self.beta, H)


class Accumulators:
    """Signed accumulators d_k(s) along one day's event stream, by an O(N) recursion.

    times: event times (sorted); eps: +1 / -1. state(s) uses events with t_m <= s.
    """

    def __init__(self, times, eps, beta):
        self.times = np.asarray(times, dtype=float)
        self.beta = np.asarray(beta, dtype=float)
        n, K = len(self.times), len(self.beta)
        self._after = np.zeros((n, K))
        d, last = np.zeros(K), 0.0
        for m in range(n):
            d = d * np.exp(-self.beta * (self.times[m] - last)) + self.beta * eps[m]
            self._after[m] = d
            last = self.times[m]

    def state(self, s):
        i = np.searchsorted(self.times, s, side="right")
        if i == 0:
            return np.zeros(len(self.beta))
        return self._after[i - 1] * np.exp(-self.beta * (s - self.times[i - 1]))
