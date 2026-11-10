"""The package must not import the model it is meant to audit.

Why this guard exists
---------------------
`truth.py` (relation-truth derivation) was deliberately kept OUT of this package: it reuses
the verifier's own predicate arithmetic, and that reuse only buys an interpretable
independent-vs-reconstructed disagreement while both versions share one implementation.
Vendoring or re-importing those predicates here would make the two sides diverge silently --
the resulting "geometry error" number would then mix geometry and formula differences and be
uninterpretable.

So the dependency direction is hard-coded in a test: this package may depend on numpy, never
on `prismv`. If someone later copy-pastes a predicate in to "just compute truth here", this
fails immediately.
"""
import os
import re
import sys

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC)


def test_no_prismv_import_inside_the_package():
    offenders = []
    for dirpath, dirnames, filenames in os.walk(SRC):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dirpath, fn)
            for i, line in enumerate(open(p, encoding="utf-8").read().splitlines(), 1):
                if re.match(r"\s*(from|import)\s+prismv\b", line):
                    offenders.append(f"{os.path.relpath(p, SRC)}:{i} {line.strip()}")
    assert not offenders, (
        "prismv-data must not import prismv -- see module docstring for why: " + str(offenders))
    print("  包内零 prismv 依赖（依赖方向被测试锁死）")


def test_package_imports_without_the_main_repo():
    import importlib

    for mod in ("prismv_data", "prismv_data.stats", "prismv_data.contract"):
        importlib.import_module(mod)
    import prismv_data

    assert "DatasetContract" in dir(prismv_data)
    print("  无主仓环境下可独立 import")


def test_ships_all_three_schemas():
    from prismv_data.contract import DatasetContract

    for name in ("scene_pack", "items", "label_audit"):
        s = DatasetContract._schema(f"{name}.schema.json")
        assert s["$schema"].startswith("https://json-schema.org"), s.get("$schema")
    print("  三个 schema 随包分发并可加载")
