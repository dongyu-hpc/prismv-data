# prismv-data

Dataset contract for [PRISM-V](https://example.invalid/prismv): schemas, hard validation
rules, stratified human-audit sampling, and measured pseudo-label precision.

The question this package answers is narrow and specific: **is this dataset allowed to support
the claim being made?** It is not a loader, not a preprocessing library, and holds no model code.

Auto-pseudo-labelled ground truth fails quietly rather than loudly. Every rule below exists to
block one concrete way of producing a plausible-but-wrong number.

## Install

```shell
pip install -e ".[dev]"          # needs numpy; jsonschema is used when present
```

Python >= 3.9. No dependency on PRISM-V — see [Import boundary](#import-boundary).

## Usage

```shell
prismv-data validate  --dataset DIR     # schema conformance + hard rules R1-R10
prismv-data sample    --dataset DIR     # stratified human-audit plan
prismv-data precision --dataset DIR     # measured pseudo-label accuracy + bootstrap CI

# equivalent module forms
python -m prismv_data validate --dataset DIR
python -m prismv_data.contract validate --dataset DIR
```

Expected layout:

```
DIR/scene_pack/*.json     layer 1 - geometry (ground truth only)
DIR/entities_pred/*.json  optional - model-predicted entities, never merged into scene_pack
DIR/items.jsonl           layer 2 - questions, the join key between the two truths
DIR/label_audit.jsonl     layer 3 - human verification ledger
```

## Two behaviours that look like bugs

**Under-powered strata report `precision=None`.** Never `0.0`, never a wide estimate that will
get quoted in a table anyway. A precision computed from three items is not a precision.

**`audit_gate()` raises instead of returning a number.** If the only available truth came from
the same estimator family the verifier reads (`pose_source: reconstruction`) and no human-checked
subset exists, verifier-agreement would be ~1.0 *by construction*. Refusing is cheaper than
publishing. The fix is data-side: pull poses from the source dataset, or audit a stratified
subset first.

## Dual ground truth

```
geo_truth_indep   sensor / COLMAP poses + GT object geometry   <- the only version that may back a claim
geo_truth_recon   reconstructed poses + reconstructed entities  <- what the verifier actually reads
```

Their disagreement rate is the verifier's **noise floor**: no implementation can track
independent truth better than the geometry it reads tracks it. Report it, or every disagreement
will be read as a verifier mistake.

## Import boundary

Relation-truth derivation lives in PRISM-V (`prismv/truth.py`), **not here**, and
`tests/test_import_boundary.py` fails the build if any `prismv` import appears in this package.

That is not packaging tidiness. Truth derivation reuses the verifier's own predicate arithmetic,
and that shared implementation is what makes the indep-vs-recon disagreement measure *geometry
error*. Copy the predicates into the audit package and the two sides diverge silently — the
resulting "geometry error" figure then mixes geometry error with formula differences and means
nothing. The boundary is a test because the tempting shortcut is a copy-paste.

## Layout

```
src/prismv_data/
  contract.py    rules R1-R10, audit gate, sampling planner, precision estimator, CLI
  stats.py       vendored bootstrap CI (equivalence pinned by tests/test_stats.py)
  schemas/       scene_pack / items / label_audit (JSON Schema 2020-12)
tests/           22 tests: contract, stats equivalence, import boundary, hygiene
docs/dataset-contract-v1.md   full spec: field-to-metric map, acceptance criteria, gap analysis
```

## Tests

```shell
python -m pytest -q
```

CI runs pytest, the import-boundary check and the hygiene gate on Python 3.9 / 3.11 / 3.12.

## License

Not chosen yet. **Decide before publishing** — the repo currently has no `LICENSE` file, which
means all rights reserved by default. If this dataset tooling is meant to accompany a public
PRISM-V release, pick the same licence as that release so downstream users get one consistent
answer.
