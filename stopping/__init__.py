"""Execution timing under multi-timescale Hawkes intensities of mid-price moves.

Depends on hawkes-taq (event construction and fitting), which has no packaging. If
`hawkes_taq` is not importable, its location is taken from the HAWKES_TAQ environment
variable, or a sibling clone ../hawkes-taq next to this repository.
"""

import os
import sys
from importlib.util import find_spec
from pathlib import Path

if find_spec("hawkes_taq") is None:
    _candidates = [os.environ.get("HAWKES_TAQ"), Path(__file__).resolve().parents[2] / "hawkes-taq"]
    for _c in _candidates:
        if _c and Path(_c, "hawkes_taq").is_dir():
            sys.path.insert(0, str(_c))
            break

from .drift import Accumulators, SignedDrift  # noqa: E402,F401
