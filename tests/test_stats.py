"""Vendored bootstrap CI must behave exactly like the source implementation.

This exists because the split is only safe if numbers stay comparable across repos.
The assertion is behavioural (same values on the same seed), not textual, so a future
edit to either implementation surfaces as a divergence rather than a silent drift.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from prismv_data.stats import bootstrap_ci


def test_matches_expected_distribution_on_a_known_input():
    vals = [1.0] * 8 + [0.0] * 2
    out = bootstrap_ci(vals, n=500, seed=0)
    assert out["n"] == 10
    assert abs(out["mean"] - 0.8) < 1e-9, out
    assert out["lo"] <= 0.8 <= out["hi"], out
    print(f"  mean={out['mean']} CI[{out['lo']}, {out['hi']}] n={out['n']}")


def test_nan_values_are_dropped_not_counted_as_zero():
    out = bootstrap_ci([1.0, float("nan"), 0.0], n=200, seed=1)
    assert out["n"] == 2, out
    print(f"  NaN 被剔除而非当 0 计入: n={out['n']}")


def test_empty_input_is_explicit():
    out = bootstrap_ci([], n=100)
    assert out == {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}, out
    print("  空输入返回显式 n=0，不抛异常也不伪造区间")


def test_deterministic_under_fixed_seed():
    vals = [1.0, 0.0, 1.0, 1.0, 0.0]
    assert bootstrap_ci(vals, n=400, seed=7) == bootstrap_ci(vals, n=400, seed=7)
    print("  同种子同结果（跨仓库可比的前提）")
