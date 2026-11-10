"""Dataset contract validation for PRISM-V route C.

Design rule: every check here blocks something that would otherwise produce a *plausible
number* rather than an error. Silent mis-measurement is the failure mode that survives review
and dies in reproduction, so several checks below raise instead of returning a metric.

Hard rules and the failure each one prevents
--------------------------------------------
R1 split leakage            two items of one scene in different splits -- the same room
                            appears in "held-out" evaluation, inflating every result
R2 pose independence        GT derived from the same estimator GeoVer verifies against makes
                            verifier agreement ~1.0 by construction
R3 auto provenance          pseudo-labels without a labeler hash are not reproducible
R4 per-point confidence     without it the conf_thres ablation is inert (found in review:
                            the knob silently did nothing on injected geometry)
R5 unscorable honesty       an item whose correct option names no grounded entity must say
                            so, instead of quietly dropping out of the denominator
R6 confidence provenance    gt_confidence must equal the precision MEASURED from label_audit;
                            hand-authored values are rejected
R7 stratum minimums         a precision estimate off 3 items is not a precision estimate;
                            it reports None, never 0.0
R8 margin/band coherence    a near-threshold relation labelled 'clear' biases audit sampling
R9 describe recall GT       relation recall needs gt_relations; precision does not, and the
                            two must not be conflated
R10 audit gate              verifier-vs-truth agreement is only computable against
                            independent truth; otherwise it refuses

Usage
-----
    python -m prismv_data.contract validate --dataset DIR
    python -m prismv_data.contract sample  --dataset DIR --per-stratum 40
    python -m prismv_data.contract precision --dataset DIR
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:  # dependency-optional: the offline test path must not require it
    import jsonschema as _js
except Exception:  # pragma: no cover
    _js = None

SCHEMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schemas")
MARGIN_NEAR = 0.05      # |margin| below this is a near-threshold call
MARGIN_MARGINAL = 0.15
MIN_PER_STRATUM = 30    # below this, a per-stratum precision is not reportable

TRUTH_VALUES = ("true", "false", "undetermined")


class ContractViolation(Exception):
    """Raised when a computation would be meaningless rather than merely imprecise."""


# --------------------------------------------------------------------- loading
def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


class DatasetContract:
    """Validates a dataset directory against the three-layer schema and the hard rules.

    Expected layout::

        DIR/scene_pack/*.json      layer 1, geometry (ground truth only)
        DIR/entities_pred/*.json   optional, model-predicted entities -- never merged into
                                   scene_pack, or grounding error becomes ground-truth error
        DIR/items.jsonl            layer 2, questions
        DIR/label_audit.jsonl      layer 3, human verification ledger
    """

    #: how far a declared gt_confidence may sit from the precision measured out of
    #: label_audit before it is treated as authored rather than derived
    CONFIDENCE_TOL = 0.10

    def __init__(self, dataset_dir: str, min_per_stratum: int = MIN_PER_STRATUM):
        self.dir = dataset_dir
        self.min_per_stratum = int(min_per_stratum)
        self.scenes: Dict[str, Dict] = {}
        self.items: List[Dict] = []
        self.audit: List[Dict] = []
        self._loaded = False

    # ------------------------------------------------------------------ load
    def _ensure(self) -> "DatasetContract":
        """Load once, on first access.

        ``validate`` and ``audit_gate`` both need the files on disk. An earlier revision got
        its load as a side effect of calling ``validate`` from inside the gate, which reads as
        redundant work -- it was the first thing deleted during cleanup, and deleting it turned
        the gate into a permanent refusal. Loading is explicit and idempotent from here on.
        """
        if not self._loaded:
            self.load()
        return self

    def load(self) -> "DatasetContract":
        pack_dir = os.path.join(self.dir, "scene_pack")
        if os.path.isdir(pack_dir):
            for fn in sorted(os.listdir(pack_dir)):
                if fn.endswith(".json"):
                    doc = _load_json(os.path.join(pack_dir, fn))
                    self.scenes[doc.get("scene_id", fn)] = doc
        self.items = _load_jsonl(os.path.join(self.dir, "items.jsonl"))
        self.audit = _load_jsonl(os.path.join(self.dir, "label_audit.jsonl"))
        self._loaded = True
        return self

    # ------------------------------------------------------------------ schema
    @staticmethod
    def _schema(name: str) -> Dict:
        return _load_json(os.path.join(SCHEMA_DIR, name))

    @classmethod
    def _check_schema(cls, doc: Dict, schema: Dict, where: str) -> List[str]:
        if _js is not None:
            validator = _js.Draft202012Validator(schema)
            return [f"{where}: "
                    f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}"
                    for e in sorted(validator.iter_errors(doc), key=lambda e: list(e.path))][:12]
        # structural fallback: required keys only, enough to keep the contract useful
        missing = [k for k in schema.get("required", []) if k not in doc]
        return [f"{where}: missing required '{k}' (structural fallback)" for k in missing]

    # ------------------------------------------------------------------ validate
    def validate(self) -> Dict[str, Any]:
        """Return a report; violations are hard failures, warnings are judgement calls."""
        self._ensure()
        violations: List[str] = []
        warnings: List[str] = []
        stats: Dict[str, Any] = {}

        if not self.scenes:
            violations.append("no scene_pack/*.json found -- nothing can be scored")
        if not self.items:
            violations.append("no items.jsonl found")

        # R1 schema conformance
        sp_schema = self._schema("scene_pack.schema.json")
        it_schema = self._schema("items.schema.json")
        la_schema = self._schema("label_audit.schema.json")
        for sid, doc in self.scenes.items():
            violations += self._check_schema(doc, sp_schema, f"scene_pack[{sid}]")
        for it in self.items:
            violations += self._check_schema(it, it_schema, f"item[{it.get('item_id')}]")
        for row in self.audit:
            violations += self._check_schema(row, la_schema, f"audit[{row.get('item_id')}]")

        # R2 split leakage: scene-disjoint by rule
        splits: Dict[str, set] = defaultdict(set)
        for it in self.items:
            if it.get("split") and it.get("scene_id"):
                splits[it["scene_id"]].add(it["split"])
        leaked = {s: sorted(v) for s, v in splits.items() if len(v) > 1}
        if leaked:
            violations.append(
                f"split leakage: {len(leaked)} scene(s) span multiple splits "
                f"{dict(list(leaked.items())[:3])}; evaluation on these is not held out")

        # R3 pose independence + circularity gate bookkeeping
        non_independent = []
        for sid, doc in self.scenes.items():
            prov = doc.get("gt_provenance", {})
            ps, indep = prov.get("pose_source"), prov.get("independent_of_verifier_input")
            if ps == "reconstruction" and indep is not False:
                violations.append(
                    f"scene[{sid}]: pose_source='reconstruction' but "
                    f"independent_of_verifier_input={indep!r}. GT computed from the same "
                    f"estimator family the verifier checks against inflates agreement by "
                    f"construction; it must be declared dependent.")
            if ps == "reconstruction":
                non_independent.append(sid)
            # R4 auto provenance
            if prov.get("source") == "auto" and not prov.get("pipeline_hash"):
                violations.append(
                    f"scene[{sid}]: auto-labelled truth without pipeline_hash is not "
                    f"reproducible -- the labeler version is part of the dataset")

        # R4b per-point confidence -> inert ablation detection
        inert = []
        for sid, doc in self.scenes.items():
            for obj in doc.get("objects", []):
                if obj.get("points") and not obj.get("point_conf"):
                    inert.append(f"{sid}/{obj.get('object_id')}")
        if inert:
            warnings.append(
                f"conf_thres ablation is INERT for {len(inert)} object(s) with points but no "
                f"point_conf (e.g. {inert[:2]}); do not plot a reconstruction-quality curve "
                f"on this data")

        # R5 unscorable honesty + R9 describe recall GT
        for it in self.items:
            iid = it.get("item_id")
            opts = it.get("options") or {}
            ans = it.get("answer")
            if it.get("task") == "localize":
                target = opts.get(ans, {}) if ans else {}
                if not target.get("object_id") and not it.get("unscorable_reason"):
                    violations.append(
                        f"item[{iid}]: correct option resolves to no grounded object but no "
                        f"unscorable_reason is given; silently dropping items corrupts the "
                        f"denominator")
            if it.get("task") == "describe" and not it.get("gt_relations"):
                warnings.append(
                    f"item[{iid}]: describe without gt_relations can score geometric "
                    f"precision/contradiction but NOT relation recall; report the two apart")
            # R8 margin/band coherence
            for letter, opt in opts.items():
                band, margin = opt.get("margin_band"), opt.get("margin")
                if band and margin is not None:
                    expected = ("near_threshold" if abs(margin) < MARGIN_NEAR else
                                "marginal" if abs(margin) < MARGIN_MARGINAL else "clear")
                    if band != expected:
                        warnings.append(
                            f"item[{iid}] option {letter}: margin={margin:+.4f} marked "
                            f"'{band}' but expected '{expected}'; mis-banded items bias "
                            f"stratified audit sampling")

        per_item_precision = _precision_map(self.audit, self.min_per_stratum)
        for it in self.items:
            declared = it.get("gt_confidence")
            if declared is None:
                continue
            got = per_item_precision.get(it.get("item_id"))
            if got is None and self.audit:
                warnings.append(
                    f"item[{it.get('item_id')}]: gt_confidence={declared} declared but no "
                    f"audit row covers its stratum, so the value is unverifiable")
            elif abs(declared - got) > self.CONFIDENCE_TOL:
                violations.append(
                    f"item[{it.get('item_id')}]: gt_confidence={declared} disagrees with "
                    f"precision measured from label_audit ({got:.3f}); confidence must be "
                    f"derived from the audit ledger, not authored")

        # coverage bookkeeping: honest denominators
        coverage: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for it in self.items:
            task = it.get("task", "?")
            coverage[task]["n"] += 1
            if it.get("answer"):
                coverage[task]["with_answer"] += 1
            if any(o.get("geo_truth_indep") for o in (it.get("options") or {}).values()):
                coverage[task]["with_independent_truth"] += 1
            if it.get("geo_unique") is True:
                coverage[task]["geo_unique"] += 1
            if it.get("unscorable_reason"):
                coverage[task]["unscorable"] += 1
            if it.get("audit_status") in ("checked", "adjudicated"):
                coverage[task]["audited"] += 1

        stats["scenes"] = len(self.scenes)
        stats["items"] = len(self.items)
        stats["audit_rows"] = len(self.audit)
        stats["coverage"] = {k: dict(v) for k, v in coverage.items()}
        stats["non_independent_scenes"] = len(non_independent)
        stats["pooled_precision"] = pooled_pseudo_label_precision(self.audit)

        return {"ok": not violations, "violations": violations, "warnings": warnings,
                "stats": stats, "jsonschema_active": _js is not None}


# --------------------------------------------------------------------- audit sampling
def margin_band(margin: Optional[float]) -> str:
    if margin is None:
        return "marginal"
    a = abs(margin)
    return "near_threshold" if a < MARGIN_NEAR else ("marginal" if a < MARGIN_MARGINAL else "clear")


def _stratum_key(stratum: Optional[Dict]) -> Tuple[str, str, str]:
    s = stratum or {}
    return (s.get("relation", "?"), s.get("frame_kind", "?"), s.get("margin_band", "?"))


def plan_audit_sample(items: Iterable[Dict], per_stratum: int = 40,
                      near_threshold_share: float = 0.35) -> List[Dict]:
    """Stratified human-check plan.

    Uniform random sampling is biased: pseudo-label accuracy is far lower near the decision
    threshold, so a random draw over-reports quality. This plan guarantees a floor of
    near-threshold items per relation, which is exactly the region where an 'explicit
    constraint' claim is most likely to be false.
    """
    buckets: Dict[Tuple[str, str, str], List[Dict]] = defaultdict(list)
    for it in items:
        if it.get("task") == "describe":
            rels = it.get("gt_relations") or []
            for r in rels:
                band = r.get("margin_band") or margin_band(r.get("margin"))
                key = (r.get("relation", "?"),
                       (it.get("ref_frame") or {}).get("kind", "unspecified"), band)
                buckets[key].append({"item_id": it.get("item_id"),
                                     "relation": r.get("relation"), "margin": r.get("margin"),
                                     "stratum": {"relation": r.get("relation"),
                                                 "frame_kind": key[1], "margin_band": band}})
            continue
        for letter, opt in (it.get("options") or {}).items():
            if not it.get("relation") or not opt.get("geo_truth_indep"):
                continue
            band = opt.get("margin_band") or margin_band(opt.get("margin"))
            key = (it["relation"], (it.get("ref_frame") or {}).get("kind", "unspecified"), band)
            buckets[key].append({"item_id": it.get("item_id"), "option": letter,
                                 "relation": it["relation"], "margin": opt.get("margin"),
                                 "stratum": {"relation": key[0], "frame_kind": key[1],
                                             "margin_band": band}})

    plan: List[Dict] = []
    near_budget = max(1, int(round(per_stratum * near_threshold_share)))
    warned: set = set()
    for key, rows in sorted(buckets.items()):
        near = [r for r in rows if r["stratum"]["margin_band"] == "near_threshold"]
        rest = [r for r in rows if r["stratum"]["margin_band"] != "near_threshold"]
        take_near = near[:min(len(near), near_budget)]
        take_rest = rest[:max(0, per_stratum - len(take_near))]
        chosen = take_near + take_rest
        for r in chosen:
            r = dict(r)
            r["reason"] = ("near_threshold_floor" if r in take_near else "stratum_fill")
            plan.append(r)
        if len(near) < near_budget and key[:2] not in warned:
            warned.add(key[:2])
            plan.append({"item_id": None, "stratum": {"relation": key[0], "frame_kind": key[1],
                                                      "margin_band": "near_threshold"},
                         "reason": f"INSUFFICIENT_NEAR_THRESHOLD: have {len(near)}, "
                                   f"need {near_budget} for an unbiased precision estimate"})
    return plan


# --------------------------------------------------------------------- precision
def pseudo_label_precision(audit: Iterable[Dict], min_n: int = MIN_PER_STRATUM) -> Dict[str, Dict]:
    """Measured pseudo-label accuracy per stratum, with bootstrap CI.

    Strata below ``min_n`` report ``None`` rather than 0.0 or a wide estimate: a precision
    computed from three items will still get quoted in a table, and 0.0 is at least honest
    about there being nothing there.
    """
    from prismv_data.stats import bootstrap_ci

    buckets: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for row in audit:
        if row.get("audit_status") not in ("checked", "adjudicated"):
            continue
        truth = row.get("adjudicated")
        if truth is None:
            hs = row.get("human_truths") or []
            if not hs:
                continue
            if "cannot tell" in hs:
                truth = "undetermined"
            else:
                truth = "true" if hs.count("true") > hs.count("false") else "false"
        pseudo = row.get("pseudo_truth")
        if pseudo is None:
            continue
        buckets[_stratum_key(row.get("stratum"))].append(1.0 if pseudo == truth else 0.0)

    out: Dict[str, Dict] = {}
    for key, vals in sorted(buckets.items()):
        ci = bootstrap_ci(vals)
        out["|".join(key)] = ({"n": len(vals), "precision": ci["mean"], "ci": [ci["lo"], ci["hi"]]}
                             if len(vals) >= min_n
                             else {"n": len(vals), "precision": None,
                                   "ci": [ci["lo"], ci["hi"]],
                                   "note": f"below min_n={min_n}, not reportable"})
    return out


def pooled_pseudo_label_precision(audit: Iterable[Dict]) -> Optional[Dict]:
    from prismv_data.stats import bootstrap_ci

    vals = []
    for row in audit:
        if row.get("audit_status") not in ("checked", "adjudicated"):
            continue
        pseudo, truth = row.get("pseudo_truth"), row.get("adjudicated")
        if pseudo is None or truth is None:
            continue
        vals.append(1.0 if pseudo == truth else 0.0)
    if not vals:
        return None
    ci = bootstrap_ci(vals)
    return {"n": len(vals), "precision": ci["mean"], "ci": [ci["lo"], ci["hi"]]}


def _precision_map(audit: List[Dict], min_n: int) -> Dict[str, float]:
    """item_id -> precision to use, from its own stratum (never hand-authored)."""
    table = pseudo_label_precision(audit, min_n)
    by_item: Dict[str, float] = {}
    for row in audit:
        key = "|".join(_stratum_key(row.get("stratum")))
        entry = table.get(key) or {}
        if entry.get("precision") is not None:
            by_item[row.get("item_id")] = float(entry["precision"])
    return by_item


# --------------------------------------------------------------------- the gate
def audit_gate(contract: DatasetContract) -> Dict[str, Any]:
    """Decide what evidence supports a verifier-vs-truth agreement claim.

    Raises :class:`ContractViolation` rather than returning a number when the only available
    truth was computed from the same estimator the verifier reads -- an agreement figure
    computed there is 1.0 by construction, and reporting it would be the single most damaging
    accidental claim this project could make.
    """
    contract._ensure()
    indep_scenes = {sid for sid, doc in contract.scenes.items()
                    if (doc.get("gt_provenance") or {}).get("independent_of_verifier_input")}
    audited = {r.get("item_id") for r in contract.audit
               if r.get("audit_status") in ("checked", "adjudicated")}
    pooled = pooled_pseudo_label_precision(contract.audit)

    if audited and pooled is not None:
        return {"scope": "audited_subset", "allowed": True,
                "n_audited": len(audited), "pseudo_label_precision": pooled,
                "note": "agreement is measurable against human-checked truth; report the "
                        "precision alongside it as the discount factor"}
    if indep_scenes and not audited:
        return {"scope": "independent_full", "allowed": True, "n_audited": 0,
                "pseudo_label_precision": None,
                "note": "truth comes from sensor/colmap/sim poses; agreement is valid but "
                        "state the pose source in the caption"}
    raise ContractViolation(
        "verifier-vs-geometric-truth agreement cannot be computed: no independent truth "
        "(pose_source=reconstruction) and no human-audited subset. This is the circularity "
        "case -- agreement would be ~1.0 by construction. Either pull poses from the source "
        "dataset or audit a stratified subset first.")


# --------------------------------------------------------------------- CLI
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="prismv_data.contract",
                                description="Validate a PRISM-V dataset against the contract")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--dataset", required=True)
        sp.add_argument("--min-per-stratum", type=int, default=MIN_PER_STRATUM)

    sp = sub.add_parser("validate"); common(sp)
    sp = sub.add_parser("sample"); common(sp); sp.add_argument("--per-stratum",
        type=int, default=40)
    sp.add_argument("--near-threshold-share", type=float, default=0.35)
    sp = sub.add_parser("precision"); common(sp)
    args = p.parse_args(argv)

    c = DatasetContract(args.dataset, args.min_per_stratum).load()

    if args.cmd == "validate":
        rep = c.validate()
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        if rep["violations"]:
            print(f"\n{len(rep['violations'])} violation(s) -- dataset not usable for claims",
                  file=sys.stderr)
            return 2
        for w in rep["warnings"]:
            print(f"  warn: {w}", file=sys.stderr)
        print("contract OK")
        return 0

    if args.cmd == "sample":
        plan = plan_audit_sample(c.items, args.per_stratum, args.near_threshold_share)
        for row in plan:
            print(json.dumps(row, ensure_ascii=False))
        print(f"\nplan size: {len(plan)}", file=sys.stderr)
        return 0

    if args.cmd == "precision":
        out = pseudo_label_precision(c.audit, args.min_per_stratum)
        out["__pooled__"] = pooled_pseudo_label_precision(c.audit)
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
