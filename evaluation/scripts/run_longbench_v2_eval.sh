#!/bin/bash
set -e

ROOT_DIR=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR"

# Common parameters
LIB="coze"
WORKERS=5
TOPK=30
VERSION_DIR="coze_longbench_v2_0224"
LIMIT=5
# CHAT_MODEL="gpt-4o-mini"
CHAT_MODEL="o4-mini"

ADD_MODE="fine"
SEARCH_MODE="fast"
ASYNC_MODE="sync"

# python evaluation/scripts/longbench_v2/longbench_v2_coze_search.py \
#   --version-dir "$VERSION_DIR"
#   # --limit "$LIMIT"

# # Add / Ingestion
# echo "Running longbench_v2_ingestion.py..."
# python -m evaluation.scripts.longbench_v2.longbench_v2_ingestion \
#   --lib "$LIB" \
#   --workers "$WORKERS" \
#   --version-dir "$VERSION_DIR" \
#   --mode "$ADD_MODE" \
#   --async-mode "$ASYNC_MODE" \
#   --limit "$LIMIT"


# #check
# echo "Running longbench_v2_check_files.py..."
# python -m evaluation.scripts.longbench_v2.longbench_v2_check_files \
#   --lib "$LIB" \
#   --version-dir "$VERSION_DIR" \

# # Search
# echo "Running longbench_v2_search.py..."
# python -m evaluation.scripts.longbench_v2.longbench_v2_search \
#  --lib "$LIB" \
#  --workers "$WORKERS" \
#  --version-dir "$VERSION_DIR" \
#  --top-k "$TOPK" \
#  --mode "$SEARCH_MODE" \
# #  --limit "$LIMIT"

# Eval
 echo "Running longbench_v2_eval.py..."
 python -m evaluation.scripts.longbench_v2.longbench_v2_eval \
  --lib "$LIB" \
  --version-dir "$VERSION_DIR" \
  --workers "$WORKERS" \
  --chat-model "$CHAT_MODEL"

#echo "All scripts completed successfully!"

# Add
# python evaluation/scripts/longbench_v2/longbench_v2_handler_add.py \
#     --version-dir "$VERSION_DIR" \
#     --lib memos-api \
#     --workers "$WORKERS"

# Search
# python evaluation/scripts/longbench_v2/longbench_v2_handler_search.py \
#     --version-dir "$VERSION_DIR" \
#     --lib memos-api \
#     --workers "$WORKERS"
