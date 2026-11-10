# Changelog

## 0.1.0

Extracted from PRISM-V as a standalone package.

- Three-layer dataset contract: `scene_pack` (geometry), `items` (questions joined to two
  truths), `label_audit` (human verification ledger).
- Hard rules R1-R10, including split leakage, pose-independence declaration, auto-provenance
  hash, per-point confidence, unscorable-item honesty, and rejection of hand-authored
  `gt_confidence`.
- `audit_gate()` refuses to compute verifier-agreement when no independent truth exists.
- Stratified audit-sampling planner with a near-threshold floor and explicit shortfall
  reporting.
- Per-stratum pseudo-label precision with bootstrap CI; under-powered strata report `None`.
- Vendored `bootstrap_ci`, with behavioural equivalence pinned by tests.
- Relation-truth derivation deliberately excluded; see `tests/test_import_boundary.py`.
