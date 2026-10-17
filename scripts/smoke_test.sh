#!/usr/bin/env bash
# CPU check that everything runs end to end (~5 min after the one-off full-data preparation, ~1 min of which is
# parsing the raw files). Run this first on a new machine. The metrics it prints are meaningless (2,000 users, tiny models).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
"$PY" -m pytest -q
rm -rf outputs/sample_2000
PYTHON="$PY" bash scripts/run_all.sh configs/smoke.yaml
"$PY" scripts/07_diagnostics.py --config configs/smoke.yaml --parts ablations
"$PY" scripts/03_train_tower.py --config configs/smoke.yaml --variant no_history
"$PY" scripts/04_score_tower.py --config configs/smoke.yaml --variant no_history
echo "SMOKE TEST PASSED"
