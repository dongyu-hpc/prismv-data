"""Code hygiene gate for this package.

Mirrors the checks the main PRISM-V repo enforces on its own packages, so the two repos do
not drift apart in discipline. Each check caught a real defect during development, which is
why they are pinned rather than left to reviewer discretion: an unused import is cheap to add
and expensive to notice, and a loop whose body is `pass` reads like logic while doing nothing.
"""
import ast
import os
import re

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
MAX_COL = 100


def _files():
    out = []
    for dirpath, dirnames, filenames in os.walk(SRC):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        out += [os.path.join(dirpath, f) for f in filenames if f.endswith(".py")]
    return sorted(out)


def test_line_length_within_budget():
    over = []
    for p in _files():
        for i, ln in enumerate(open(p, encoding="utf-8").read().splitlines(), 1):
            if len(ln) > MAX_COL:
                over.append(os.path.relpath(p, SRC) + ":" + str(i))
    assert not over, "超过 " + str(MAX_COL) + " 列: " + ", ".join(over[:6])
    print("  全部行 <= " + str(MAX_COL) + " 列")


def test_no_dead_statements():
    bad = []
    for p in _files():
        lines = open(p, encoding="utf-8").read().splitlines()
        for i, ln in enumerate(lines, 1):
            s = ln.strip()
            if re.match(r"^for\s+\w+\s+in\s+\[\]\s*:", s):
                bad.append(p + ":" + str(i) + " " + s[:50])
            elif re.match(r"^for\s+.+:\s*$", s) and i < len(lines) \
                    and lines[i].strip() == "pass":
                bad.append(p + ":" + str(i) + " " + s[:50])
            elif re.match(r"^_ = \w+$", s):
                bad.append(p + ":" + str(i) + " " + s[:50])
    assert not bad, "死语句: " + "; ".join(bad[:5])
    print("  无死循环 / 占位赋值")


def test_no_unused_imports():
    offenders = []
    for p in _files():
        src = open(p, encoding="utf-8").read()
        tree = ast.parse(src)
        imported = {}
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    imported[a.asname or a.name.split(".")[0]] = n.lineno
            elif isinstance(n, ast.ImportFrom):
                if n.module == "__future__":
                    continue
                for a in n.names:
                    if a.name != "*":
                        imported[a.asname or a.name] = n.lineno
        used = {x.id for x in ast.walk(tree) if isinstance(x, ast.Name)}
        used |= {x.attr for x in ast.walk(tree) if isinstance(x, ast.Attribute)}
        lines = src.splitlines()
        for name, ln in imported.items():
            if name in used:
                continue
            if ln <= len(lines) and "noqa" in lines[ln - 1]:
                continue
            offenders.append(os.path.relpath(p, SRC) + ":" + str(ln) + " " + name)
    assert not offenders, "未使用 import: " + ", ".join(offenders[:6])
    print("  无未使用 import")


def test_every_module_has_a_docstring():
    missing = []
    for p in _files():
        doc = ast.get_docstring(ast.parse(open(p, encoding="utf-8").read()))
        if not doc:
            missing.append(os.path.relpath(p, SRC))
    assert not missing, "缺模块 docstring: " + ", ".join(missing)
    print("  模块 docstring 全覆盖")
