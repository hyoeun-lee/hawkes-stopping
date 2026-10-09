import numpy as np
from scipy.linalg import expm

from stopping.drift import Accumulators, SignedDrift

BETA = np.array([50.0, 1.0])
A = np.array([0.1, 0.2])   # self
C = np.array([0.05, 0.3])  # cross


def test_closed_form_matches_matrix_exponential():
    dr = SignedDrift(BETA, A, C)
    K = len(BETA)
    M = -np.diag(BETA) + np.outer(BETA, A - C)
    big = np.zeros((2 * K, 2 * K))
    big[:K, :K] = M
    big[K:, :K] = np.eye(K)
    d = np.array([3.0, -0.4])
    for H in (0.01, 1.0, 30.0):
        ref = (A - C) @ expm(big * H)[K:, :K] @ d
        assert abs(dr.expected(d, H) - ref) < 1e-10


def _cascade_net(H, rng, n):
    """Monte Carlo: net up-minus-down moves within H triggered by one up move at 0."""
    vals = np.empty(n)
    for i in range(n):
        stack, net = [(0.0, 1)], 0
        while stack:
            t, e = stack.pop()
            for k in range(len(BETA)):
                for typ, alpha in ((e, A[k]), (-e, C[k])):
                    for _ in range(rng.poisson(alpha)):
                        tt = t + rng.exponential(1 / BETA[k])
                        if tt <= H:
                            net += typ
                            stack.append((tt, typ))
        vals[i] = net
    return vals.mean(), vals.std() / np.sqrt(n)


def test_expected_displacement_matches_simulated_cascades():
    dr = SignedDrift(BETA, A, C)
    rng = np.random.default_rng(1)
    for H in (0.1, 10.0):
        mc, se = _cascade_net(H, rng, 40_000)
        assert abs(dr.after_event(H) - mc) < 4 * se + 1e-3


def test_accumulators_recursion():
    t = np.array([0.5, 0.7, 2.0])
    eps = np.array([1.0, -1.0, 1.0])
    acc = Accumulators(t, eps, BETA)
    s = 2.3
    ref = (BETA[None, :] * eps[:, None] * np.exp(-BETA[None, :] * (s - t[:, None]))).sum(0)
    assert np.allclose(acc.state(s), ref)
    assert np.allclose(acc.state(0.1), 0.0)
