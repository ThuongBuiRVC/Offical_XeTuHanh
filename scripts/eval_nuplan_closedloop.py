"""DEPRECATED — use scripts/run_navsim_pdm_score.py for closed-loop scoring.

The real benchmark is the official NAVSIM PDMS/EPDMS scorer, driven by
``scripts/run_navsim_pdm_score.py`` with ``src/eval/navsim_agent.py``. That path
shares feature extraction with training via ``src/data/navsim_features.py``, so
train and test stay in sync.

This file used to be a placeholder open-loop sketch on random data; it is kept
only as a redirect so old commands fail loudly with guidance.
"""
from __future__ import annotations

import sys


def main() -> None:
    sys.exit(
        "scripts/eval_nuplan_closedloop.py is deprecated.\n"
        "Run the official scorer instead:\n"
        "  .venv/bin/python scripts/run_navsim_pdm_score.py --help\n"
    )


if __name__ == "__main__":
    main()
