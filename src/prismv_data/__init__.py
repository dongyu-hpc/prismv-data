"""PRISM-V dataset contract: schemas, validation rules, and audit-precision estimation.

What this package decides
-------------------------
It answers one question: *is this dataset allowed to support the claim being made?* Route C
(auto pseudo-labels plus a human-verified subset) fails quietly rather than loudly, so every
rule here exists to block a specific way of producing a plausible-but-wrong number.

Two behaviours are intentional and will look like bugs:

1. Under-powered strata report ``precision=None``, never ``0.0`` and never a wide estimate
   that will nonetheless get quoted in a table.
2. :func:`audit_gate` **raises** instead of returning a verifier-agreement figure when the only
   available truth came from the same estimator family the verifier reads. In that case the
   figure would be ~1.0 by construction.

What is deliberately NOT here
-----------------------------
Relation-truth derivation (``prismv/truth.py`` in the main repo). It reuses the verifier's own
predicate arithmetic, and that shared implementation is what makes the
independent-vs-reconstructed disagreement measure geometry error rather than a mixture of
geometry error and formula divergence. Vendoring predicates in here would break that argument
silently, so ``tests/test_import_boundary.py`` forbids any ``prismv`` import in this package.
"""

from prismv_data.contract import (  # noqa: F401
    ContractViolation,
    DatasetContract,
    audit_gate,
    margin_band,
    plan_audit_sample,
    pooled_pseudo_label_precision,
    pseudo_label_precision,
)
from prismv_data.stats import bootstrap_ci  # noqa: F401

__version__ = "0.1.0"

__all__ = ["ContractViolation", "DatasetContract", "audit_gate", "margin_band",
           "plan_audit_sample", "pooled_pseudo_label_precision", "pseudo_label_precision",
           "bootstrap_ci", "__version__"]
