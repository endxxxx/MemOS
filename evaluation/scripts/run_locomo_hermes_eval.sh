#!/bin/bash
set -euo pipefail

# Hermes Agent LoCoMo evaluation.
# Override: VERSION=my-label ./run_locomo_hermes_eval.sh
LIB="${LIB:-hermes-agent}"
VERSION="${VERSION:-default}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$EVAL_ROOT"
export PYTHONPATH="$EVAL_ROOT"

echo "Running locomo_hermes_eval.py for Hermes Agent..."
python scripts/locomo/locomo_hermes_eval.py --version "$VERSION"

echo "Running locomo_agent_metrics.py..."
python scripts/locomo/locomo_agent_metrics.py --client_type "$LIB" --version "$VERSION"
echo "All scripts completed successfully!"
