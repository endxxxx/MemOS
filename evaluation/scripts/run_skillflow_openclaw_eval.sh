#!/bin/bash
set -euo pipefail

# Override with environment:
# CLIENT_TYPE=memos-cloud VERSION=my-label NUM_RUNS=3 ./run_skillflow_openclaw_eval.sh
# CLIENT_TYPE=memos-cloud VERSION=my-label INCLUDE_DOMAIN="FamilyA FamilyB" NUM_RUNS=3 ./run_skillflow_openclaw_eval.sh
# CLIENT_TYPE=memos-cloud VERSION=my-label EXCLUDE_DOMAIN=SomeDomain NUM_RUNS=3 ./run_skillflow_openclaw_eval.sh
CLIENT_TYPE="${CLIENT_TYPE:-openclaw}"
VERSION="${VERSION:-default}"
INCLUDE_DOMAIN="${INCLUDE_DOMAIN:-}"
EXCLUDE_DOMAIN="${EXCLUDE_DOMAIN:-}"
NUM_RUNS="${NUM_RUNS:-1}"
NUM_TRAIN_SET="${NUM_TRAIN_SET:-4}"
TRAIN_MAX_TURNS="${TRAIN_MAX_TURNS:-3}"
TEST_MAX_TURNS="${TEST_MAX_TURNS:-1}"
ONLY_TEST="${ONLY_TEST:-false}"

if [[ -n "$INCLUDE_DOMAIN" && -n "$EXCLUDE_DOMAIN" ]]; then
  echo "Error: INCLUDE_DOMAIN and EXCLUDE_DOMAIN cannot be set at the same time." >&2
  exit 1
fi

DOMAIN_ARGS=()
if [[ -n "$INCLUDE_DOMAIN" ]]; then
  read -r -a INCLUDE_DOMAIN_ARGS <<< "$INCLUDE_DOMAIN"
  DOMAIN_ARGS+=(--include_domain "${INCLUDE_DOMAIN_ARGS[@]}")
elif [[ -n "$EXCLUDE_DOMAIN" ]]; then
  read -r -a EXCLUDE_DOMAIN_ARGS <<< "$EXCLUDE_DOMAIN"
  DOMAIN_ARGS+=(--exclude_domain "${EXCLUDE_DOMAIN_ARGS[@]}")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$EVAL_ROOT"
export PYTHONPATH=/root/MemOS/evaluation

echo "Running skillflow_openclaw_eval.py..."
python scripts/SkillFlow/skillflow_openclaw_eval.py \
  "${DOMAIN_ARGS[@]}" \
  --num_runs "$NUM_RUNS" \
  --num_train_set "$NUM_TRAIN_SET" \
  --train_max_turns "$TRAIN_MAX_TURNS" \
  --test_max_turns "$TEST_MAX_TURNS" \
  --version "$VERSION" \
  --client_type "$CLIENT_TYPE" \
  $(if [[ "$ONLY_TEST" == "true" || "$ONLY_TEST" == "1" ]]; then echo "--only_test"; fi)

echo "SkillFlow OpenClaw evaluation completed successfully!"
