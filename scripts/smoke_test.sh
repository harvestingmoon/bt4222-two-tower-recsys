#!/usr/bin/env bash
# CPU check that everything runs end to end (~5 min after the one-off full-data preparation, ~1 min of which is
# parsing the raw files). Run this first on a new machine. The metrics it prints are meaningless (2,000 users, tiny models).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
OMP_NUM_THREADS=1 "$PY" -m pytest -q   # torch + lightgbm each bundle OpenMP: multi-threaded pytest segfaults on macOS
rm -rf outputs/sample_2000
PYTHON="$PY" bash scripts/run_all.sh configs/smoke.yaml
echo "SMOKE TEST PASSED"
