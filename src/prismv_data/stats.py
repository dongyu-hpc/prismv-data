"""Bootstrap confidence intervals, vendored so the contract is analysable alone.

The main PRISM-V repo computes the same statistic in ``prismv.eval.metrics``. Keeping two
implementations is normally a smell; here it is the price of a dependency-free audit package,
and it is paid back by an equivalence test (``tests/test_stats.py``) pinned to identical
semantics: same seed, same values, same rounding. A future edit to either side shows up as a
divergence rather than as two repos quietly reporting different intervals.

Deliberately not a general stats library: this covers the one thing the contract needs,
percentile bootstrap over per-item 0/1 or rate values, with NaN dropped rather than counted
as zero -- because a missing item is not a wrong item, and folding the two together would
make pseudo-label accuracy read lower than the data supports.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def bootstrap_ci(values: Sequence[float], n: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> Dict[str, float]:
    """Percentile bootstrap CI over per-item scores.

    Audit samples are small by construction (a few dozen items per stratum), so a point
    estimate alone would let a 12-item precision figure get quoted as if it settled anything.
    Returning the interval is what makes "pseudo-label precision is 0.75" a checkable claim.
    """
    v = np.asarray([x for x in values if x == x], dtype=np.float64)   # drop NaN, not count as 0
    if len(v) == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(v), size=(n, len(v)))
    means = v[idx].mean(axis=1)
    return {"mean": round(float(v.mean()), 4),
            "lo": round(float(np.percentile(means, 100 * alpha / 2)), 4),
            "hi": round(float(np.percentile(means, 100 * (1 - alpha / 2))), 4),
            "n": int(len(v))}
