ACMnxt — Asset Condition Monitoring (Next-Gen)

Objective
- Rebuild ACM pipeline: modular, typed, documented, fast, and explainable.
- Outputs: dq.csv, features.parquet, scores.csv, events.csv, PNGs, HTML report.

Layout
```
acmnxt/
  core/       # ML, features, scoring
  io/         # loaders, writers, schemas
  vis/        # charts, plots, timelines
  report/     # md/html builders
  cli/        # entrypoints (train, score, report)
  scripts/    # PowerShell + bash wrappers
  conf/       # default + equipment configs
  tests/      # unit + golden image + e2e tests
  docs/       # diagrams, glossary, pipeline map
```

Quick Start (planned)
- `make setup` to create venv and install.
- `pytest -q` to run tests.
- `acmnxt train --csv <path> --equip FD_FAN --out-dir artifacts/FD_FAN`

