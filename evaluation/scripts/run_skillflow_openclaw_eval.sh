#!/bin/bash
set -euo pipefail

# Override with environment:
# CLIENT_TYPE=memos-cloud VERSION=my-label TASK_FAMILY_NAME=HWPX-Document-Automation NUM_RUNS=3 ./run_skillflow_openclaw_eval.sh
CLIENT_TYPE="${CLIENT_TYPE:-openclaw}"
VERSION="${VERSION:-default}"
TASK_FAMILY_NAME="${TASK_FAMILY_NAME:-HWPX-Document-Automation}"
NUM_RUNS="${NUM_RUNS:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$EVAL_ROOT"
export PYTHONPATH=/root/MemOS/evaluation

echo "Running skillflow_openclaw_eval.py..."
python3 scripts/SkillFlow/skillflow_openclaw_eval.py \
  --task_family_name "$TASK_FAMILY_NAME" \
  --num_runs "$NUM_RUNS" \
  --version "$VERSION" \
  --client_type "$CLIENT_TYPE"

echo "SkillFlow OpenClaw evaluation completed successfully!"
