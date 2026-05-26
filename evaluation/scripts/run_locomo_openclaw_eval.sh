#!/bin/bash
set -euo pipefail

# Override with environment: LIB=memos-cloud VERSION=my-label ./run_locomo_openclaw_eval.sh
LIB="${LIB:-openclaw}"
VERSION="${VERSION:-default}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$EVAL_ROOT"
export PYTHONPATH="$EVAL_ROOT"

echo "Running locomo_openclaw_eval.py..."
python scripts/locomo/locomo_openclaw_eval.py --client_type "$LIB" --version "$VERSION"

echo "Running locomo_agent_metrics.py..."
python scripts/locomo/locomo_agent_metrics.py --client_type "$LIB" --version "$VERSION"
echo "All scripts completed successfully!"
