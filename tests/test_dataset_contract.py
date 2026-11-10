"""Contract tests: every rule here blocks a plausible-but-wrong number.

Each test drives a failure that would otherwise produce a plausible number instead of an
error, which is the only class of bug worth this much ceremony in an evaluation harness.

Run:  python tests/test_dataset_contract.py     (also works under pytest)
"""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from prismv_data.contract import (  # noqa: E402
    ContractViolation,
    DatasetContract,
    audit_gate,
    plan_audit_sample,
    pooled_pseudo_label_precision,
    pseudo_label_precision,
)

MIN_N = 5   # small so the fixture stays tiny; production default is 30


# --------------------------------------------------------------------- fixtures

def _frames():
    """view0 looks down +Z (+X right, +Y down); view1 is turned around, so left/right swap."""
    identity = np.eye(4).tolist()
    R = np.diag([-1.0, 1.0, -1.0])
    turned = np.eye(4)
    turned[:3, :3] = R
    turned[:3, 3] = -(R @ np.array([0.0, 0.0, 4.0]))
    eye3 = np.eye(3).tolist()
    return [
        {"index": 0, "image": "v0.jpg", "K": eye3, "extrinsic": identity,
         "convention": "world_to_cam", "depth_conf": {"mean": 0.9, "p05": 0.7, "model": "sensor"}},
        {"index": 1, "image": "v1.jpg", "K": eye3, "extrinsic": turned.tolist(),
         "convention": "world_to_cam", "depth_conf": {"mean": 0.9, "p05": 0.7, "model": "sensor"}},
    ]


def _objects():
    def obj(oid, cat, center, radius=0.12):
        return {"object_id": oid, "category": cat, "canonical_name": cat,
                "aliases": [cat], "center": list(center), "radius": radius,
                "views": [{"frame": 0, "pixels": 4000, "conf": 0.92},
                          {"frame": 1, "pixels": 3800, "conf": 0.88}]}
    return [
        obj("book", "book", [-0.40, 0.10, 2.00]),
        obj("cup", "cup", [0.30, 0.10, 2.00]),
        obj("plant", "plant", [-0.60, 0.10, 2.20]),
        obj("lamp", "lamp", [0.00, -0.50, 3.00], radius=0.25),
        obj("sofa", "sofa", [0.00, 0.20, 1.20], radius=0.45),
    ]


def _scene_pack(**over):
    pack = {
        "schema_version": "1.0", "scene_id": "sc1", "source": "fixture",
        "license": "cc-by-4.0", "unit": "meter",
        "world_frame": {"up_axis": "-Y", "up_source": "sensor", "handedness": "right",
                        "source_is_metric": True},
        "frames": _frames(), "objects": _objects(),
        "gt_provenance": {"source": "human", "pose_source": "colmap",
                          "independent_of_verifier_input": True, "human_verified": True,
                          "pipeline_hash": "fixture@1"},
    }
    pack.update(over)
    return pack


def _items():
    return [
        {"schema_version": "1.0", "item_id": "q1", "scene_id": "sc1", "task": "qa",
         "split": "eval", "relation": "left_of",
         "ref_frame": {"kind": "viewer", "view": 0}, "anchor_object_id": "book",
         "question": "From image 1, what is to the left of the book?",
         "options": {
             "A": {"text": "cup", "object_id": "cup", "geo_truth_indep": "false",
                   "margin": -0.34, "margin_band": "clear"},
             "B": {"text": "plant", "object_id": "plant", "geo_truth_indep": "true",
                   "margin": 0.064, "margin_band": "marginal"},
             "C": {"text": "lamp", "object_id": "lamp", "geo_truth_indep": "false",
                   "margin": -0.196, "margin_band": "clear"}},
         "answer": "B", "geo_unique": True,
         "attrs": {"n_views": 2, "occlusion_ratio": 0.1, "entity_spacing": 1.04}},
        {"schema_version": "1.0", "item_id": "l1", "scene_id": "sc1", "task": "localize",
         "split": "eval", "relation": "right_of",
         "ref_frame": {"kind": "viewer", "view": 0}, "anchor_object_id": "book",
         "question": "From image 1, which object is to the right of the book?",
         "options": {"A": {"text": "cup", "object_id": "cup"}, "B": {"text": "plant",
                                                                     "object_id": "plant"}},
         "answer": "A", "attrs": {"n_views": 2}},
        {"schema_version": "1.0", "item_id": "d1", "scene_id": "sc1", "task": "describe",
         "split": "eval",
         "question": "Describe the layout.",
         "gt_relations": [{"relation": "above", "subject_object_id": "lamp",
                           "object_object_id": "sofa", "ref_view": 0, "truth": "true",
                           "margin": 1.0}], "answer": ""},
    ]


def _audit(n_per_stratum: int = 8, precision: float = 1.0):
    """Audit ledger: first `precision` fraction of rows agree with the pseudo-label."""
    rows = []
    agree_n = int(round(n_per_stratum * precision))
    for i in range(n_per_stratum):
        agree = i < agree_n
        rows.append({"schema_version": "1.0", "item_id": f"q1",
                     "scene_id": "sc1",
                     "stratum": {"relation": "left_of", "frame_kind": "viewer",
                                 "margin_band": "clear" if i % 2 else "near_threshold"},
                     "pseudo_truth": "true", "margin": 0.4 if i % 2 else 0.02,
                     "human_truths": ["true" if agree else "false"],
                     "adjudicated": "true" if agree else "false",
                     "agree": agree, "audit_status": "checked"})
    return rows


def _dataset(tmp, pack=None, items=None, audit=None):
    root = tempfile.mkdtemp(prefix="prismv_ds_", dir=tmp)
    os.makedirs(os.path.join(root, "scene_pack"))
    pack = pack or _scene_pack()
    with open(os.path.join(root, "scene_pack", "sc1.json"), "w") as f:
        json.dump(pack, f)
    with open(os.path.join(root, "items.jsonl"), "w") as f:
        for it in (items if items is not None else _items()):
            f.write(json.dumps(it) + "\n")
    if audit is not None:
        with open(os.path.join(root, "label_audit.jsonl"), "w") as f:
            for row in audit:
                f.write(json.dumps(row) + "\n")
    return root


def test_valid_dataset_passes(tmp_path):
    root = _dataset(str(tmp_path), audit=_audit())
    rep = DatasetContract(root, min_per_stratum=MIN_N).validate()
    assert rep["ok"], rep["violations"]
    assert rep["stats"]["scenes"] == 1 and rep["stats"]["items"] == 3
    cov = rep["stats"]["coverage"]["qa"]
    assert cov["with_independent_truth"] == 1 and cov["geo_unique"] == 1, cov
    print(f"  合规数据集通过 (violations={len(rep['violations'])}, "
          f"jsonschema={rep['jsonschema_active']})")


def test_missing_required_field_is_a_violation(tmp_path):
    pack = _scene_pack()
    del pack["unit"]
    rep = DatasetContract(_dataset(str(tmp_path), pack=pack), MIN_N).validate()
    assert not rep["ok"] and any("unit" in v for v in rep["violations"]), rep["violations"]
    print("  缺必填字段 unit 被拦下")


def test_r1_split_leakage_blocked(tmp_path):
    items = _items()
    items[1]["split"] = "heldout"          # same scene, different split
    rep = DatasetContract(_dataset(str(tmp_path), items=items), MIN_N).validate()
    assert any("split leakage" in v for v in rep["violations"]), rep["violations"]
    print("  同场景跨 split 被拦下（几何上是同一个房间）")


def test_r2_circular_truth_declared(tmp_path):
    pack = _scene_pack(gt_provenance={"source": "auto", "pose_source": "reconstruction",
                                      "independent_of_verifier_input": True,
                                      "pipeline_hash": "x@1"})
    rep = DatasetContract(_dataset(str(tmp_path), pack=pack), MIN_N).validate()
    assert any("independent_of_verifier_input" in v for v in rep["violations"]), rep["violations"]
    print("  pose_source=reconstruction 却声称独立 被拦下")


def test_r3_auto_without_hash_blocked(tmp_path):
    pack = _scene_pack(gt_provenance={"source": "auto", "pose_source": "colmap",
                                      "independent_of_verifier_input": True})
    rep = DatasetContract(_dataset(str(tmp_path), pack=pack), MIN_N).validate()
    assert any("pipeline_hash" in v for v in rep["violations"]), rep["violations"]
    print("  自动伪标缺 pipeline_hash 被拦下（标签器版本属于数据集的一部分）")


def test_r4_inert_conf_ablation_warned(tmp_path):
    objs = _objects()
    objs[0]["points"] = (np.asarray([-0.4, 0.1, 2.0])
                         + np.random.RandomState(0).randn(60, 3) * 0.05).tolist()
    pack = _scene_pack(objects=objs)          # points but no point_conf
    rep = DatasetContract(_dataset(str(tmp_path), pack=pack), MIN_N).validate()
    assert any("INERT" in w and "point_conf" in w for w in rep["warnings"]), rep["warnings"]
    print("  有点无置信度 -> 明确警告重建质量消融为空转")


def test_r5_unscorable_must_be_declared(tmp_path):
    items = _items()
    items[1]["options"]["A"] = {"text": "a giraffe"}     # resolves to nothing
    rep = DatasetContract(_dataset(str(tmp_path), items=items), MIN_N).validate()
    assert any("unscorable_reason" in v for v in rep["violations"]), rep["violations"]
    print("  localize 正确项无法接地却不声明 -> 被拦下（分母不能悄悄缩水）")


def test_r6_handauthored_confidence_rejected(tmp_path):
    items = _items()
    items[0]["gt_confidence"] = 0.95
    # each stratum needs >= MIN_N rows to be reportable; the fixture splits rows across two
    # margin bands, so the count has to cover both or precision downgrades to "unverifiable"
    rep = DatasetContract(_dataset(str(tmp_path), items=items,
                                   audit=_audit(n_per_stratum=4 * MIN_N, precision=0.25)),
                          MIN_N).validate()
    assert any("gt_confidence" in v for v in rep["violations"]), rep["violations"]
    print("  手写 gt_confidence=0.95 与实测精度不符 -> 被拒")


def test_r8_margin_band_coherence(tmp_path):
    items = copy.deepcopy(_items())
    items[0]["options"]["B"]["margin"] = 0.001
    items[0]["options"]["B"]["margin_band"] = "clear"
    rep = DatasetContract(_dataset(str(tmp_path), items=items), MIN_N).validate()
    assert any("mis-banded" in w or "expected 'near_threshold'" in w
               for w in rep["warnings"]), rep["warnings"]
    print("  margin 与 margin_band 不自洽被标出（分层抽样的前提）")


def test_r9_describe_precision_without_recall(tmp_path):
    items = _items()
    items[2].pop("gt_relations")
    rep = DatasetContract(_dataset(str(tmp_path), items=items), MIN_N).validate()
    assert any("recall" in w for w in rep["warnings"]), rep["warnings"]
    assert rep["ok"], rep["violations"]      # precision is still legitimately computable
    print("  describe 缺关系真值：警告可算精确率但不可算召回，不冒充两者等价")


def test_r7_underpowered_stratum_reports_none_not_zero(tmp_path):
    table = pseudo_label_precision(_audit(n_per_stratum=3, precision=1.0), min_n=MIN_N)
    assert table, table
    for key, entry in table.items():
        assert entry["precision"] is None and entry["n"] < MIN_N, (key, entry)
    print(f"  样本不足的层 precision=None（而非 0.0 或宽区间）: {len(table)} 层")


def test_precision_measured_with_ci(tmp_path):
    table = pseudo_label_precision(_audit(n_per_stratum=12, precision=0.75), min_n=MIN_N)
    pooled = pooled_pseudo_label_precision(_audit(n_per_stratum=12, precision=0.75))
    assert pooled and pooled["n"] == 12 and abs(pooled["precision"] - 0.75) < 1e-6, pooled
    assert pooled["ci"][0] <= 0.75 <= pooled["ci"][1], pooled
    assert all(v["precision"] is not None for v in table.values()), table
    print(f"  伪标精度按层与合并均可测: pooled={pooled['precision']} "
          f"CI[{pooled['ci'][0]}, {pooled['ci'][1]}] n={pooled['n']}")


def test_audit_sample_keeps_near_threshold_floor(tmp_path):
    items = copy.deepcopy(_items())
    # one option deliberately placed on the decision boundary, which is the stratum where an
    # 'explicit constraint' claim is most likely to be wrong and therefore must be sampled
    items[0]["options"]["B"]["margin"] = 0.02
    items[0]["options"]["B"]["margin_band"] = "near_threshold"
    plan = plan_audit_sample(items, per_stratum=6, near_threshold_share=0.5)

    floors = [r for r in plan if r.get("reason") == "near_threshold_floor"]
    assert floors, plan
    assert all(r["stratum"]["margin_band"] == "near_threshold" for r in floors), floors

    # a relation with no near-threshold item must say so, rather than returning a plan that
    # looks complete and silently measures nothing at the boundary
    insufficient = [r for r in plan if isinstance(r.get("reason") or "", str)
                    and str(r["reason"]).startswith("INSUFFICIENT")]
    assert insufficient, plan
    assert len(insufficient) == len({(r["stratum"]["relation"], r["stratum"]["frame_kind"])
                                    for r in insufficient}), "缺口告警应按关系去重，否则会被重复计数"
    print(f"  近阈值下限抽 {len(floors)} 条；{len(insufficient)} 个关系缺近阈值样本被显式报出")


def test_gate_raises_on_circular_truth(tmp_path):
    pack = _scene_pack(gt_provenance={"source": "auto", "pose_source": "reconstruction",
                                      "independent_of_verifier_input": False,
                                      "pipeline_hash": "x@1"})
    c = DatasetContract(_dataset(str(tmp_path), pack=pack), MIN_N)
    try:
        audit_gate(c)
    except ContractViolation as e:
        assert "circularity" in str(e)
        print("  无独立真值且无人工子集时，一致率指标直接拒绝计算（而非给出 ~1.0）")
        return
    raise AssertionError("expected ContractViolation")


def test_gate_allows_independent_and_audited(tmp_path):
    c = DatasetContract(_dataset(str(tmp_path), audit=_audit(precision=0.875)), MIN_N)
    g = audit_gate(c)
    assert g["allowed"] and g["scope"] == "audited_subset", g
    assert g["pseudo_label_precision"]["precision"] == 0.875, g
    print(f"  有人工子集时允许计算，并把伪标精度一起带出: "
          f"{g['pseudo_label_precision']['precision']}")

    c2 = DatasetContract(_dataset(str(tmp_path)), MIN_N)
    g2 = audit_gate(c2)
    assert g2["scope"] == "independent_full", g2
    print("  colmap/sensor 位姿时全量可算，但要求标注位姿来源")


def main():
    tmp = tempfile.mkdtemp(prefix="prismv_ds_root_")
    print(f"running {len(ALL_TESTS)} data-contract tests\n")
    failed = []
    for fn in ALL_TESTS:
        args = ()
        if "tmp_path" in fn.__code__.co_varnames[: fn.__code__.co_argcount]:
            args = (tmp,)          # a str, matching the str(tmp_path) the pytest path uses
        try:
            fn(*args)
        except AssertionError as e:
            failed.append((fn.__name__, str(e)))
            print(f"  !! {fn.__name__} FAILED: {e}")
        except Exception as e:
            failed.append((fn.__name__, f"{type(e).__name__}: {e}"))
            print(f"  !! {fn.__name__} ERROR: {type(e).__name__}: {e}")
    print(f"\n{len(ALL_TESTS) - len(failed)}/{len(ALL_TESTS)} passed")
    return 1 if failed else 0

[v for k, v in sorted(globals().items()) if k.startswith("test_")]




ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main():
    tmp = tempfile.mkdtemp(prefix="prismv_ds_root_")
    print(f"running {len(ALL_TESTS)} data-contract tests\n")
    failed = []
    for fn in ALL_TESTS:
        args = ()
        if "tmp_path" in fn.__code__.co_varnames[: fn.__code__.co_argcount]:
            args = (tmp,)
        try:
            fn(*args)
        except AssertionError as e:
            failed.append((fn.__name__, str(e)))
            print(f"  !! {fn.__name__} FAILED: {e}")
        except Exception as e:
            failed.append((fn.__name__, f"{type(e).__name__}: {e}"))
            print(f"  !! {fn.__name__} ERROR: {type(e).__name__}: {e}")
    print(f"\n{len(ALL_TESTS) - len(failed)}/{len(ALL_TESTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
