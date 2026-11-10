"""Allow ``python -m prismv_data`` as the CLI entry point.

Without this, ``python -m prismv_data.contract`` emits a RuntimeWarning because the package
``__init__`` imports the contract module before runpy gets to execute it as ``__main__``.
Harmless in effect, but a warning on the documented command makes people distrust the output --
and the console script ``prismv-data`` is the primary entry anyway.
"""

from prismv_data.contract import main

if __name__ == "__main__":
    raise SystemExit(main())
