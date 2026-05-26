#!/bin/bash
set -euo pipefail

# Override with environment:
# CLIENT_TYPE=memos-cloud VERSION=my-label TASK_FAMILY_NAME=HWPX-Document-Automation NUM_RUNS=3 ./run_skillflow_openclaw_eval.sh
# CLIENT_TYPE=memos-cloud VERSION=my-label TASK_FAMILY_NAMES="FamilyA FamilyB" NUM_RUNS=3 ./run_skillflow_openclaw_eval.sh
CLIENT_TYPE="${CLIENT_TYPE:-openclaw}"
VERSION="${VERSION:-default}"
TASK_FAMILY_NAMES="${TASK_FAMILY_NAMES:-${TASK_FAMILY_NAME:-HWPX-Document-Automation}}"
NUM_RUNS="${NUM_RUNS:-1}"
NUM_TRAIN_SET="${NUM_TRAIN_SET:-4}"
TRAIN_MAX_TURNS="${TRAIN_MAX_TURNS:-3}"
TEST_MAX_TURNS="${TEST_MAX_TURNS:-1}"
ONLY_TEST="${ONLY_TEST:-false}"
read -r -a TASK_FAMILY_NAME_ARGS <<< "$TASK_FAMILY_NAMES"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$EVAL_ROOT"
export PYTHONPATH=/root/MemOS/evaluation

echo "Running skillflow_openclaw_eval.py..."
python scripts/SkillFlow/skillflow_openclaw_eval.py \
  --task_family_name "${TASK_FAMILY_NAME_ARGS[@]}" \
  --num_runs "$NUM_RUNS" \
  --num_train_set "$NUM_TRAIN_SET" \
  --train_max_turns "$TRAIN_MAX_TURNS" \
  --test_max_turns "$TEST_MAX_TURNS" \
  --version "$VERSION" \
  --client_type "$CLIENT_TYPE" \
  $(if [[ "$ONLY_TEST" == "true" || "$ONLY_TEST" == "1" ]]; then echo "--only_test"; fi)

echo "SkillFlow OpenClaw evaluation completed successfully!"
