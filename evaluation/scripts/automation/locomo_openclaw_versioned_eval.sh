#!/usr/bin/env bash
# Daily automation: compare npm versions (openclaw + MemOS plugins) to a state file;
# on change, install, restart gateway, run LOCOMO eval for affected client types.
#
# Environment (optional):
#   MEMOS_EVAL_ROOT          — path to evaluation/ (default: repo evaluation dir)
#   MEMOS_LOCOMO_STATE_DIR   — state directory (default: /var/lib/memos-locomo-eval)
#   MEMOS_LOCOMO_STATE_FILE  — override state JSON path
#   MEMOS_LOCOMO_PROFILE     — both | cloud | local | openclaw (default: both).
#                              openclaw = only watch OPENCLAW_NPM_PACKAGE; on new version run
#                              LIB=openclaw (OpenClaw native memory-core path, no MemOS plugins).
#                              Use cloud/local on separate hosts; set distinct STATE_FILE per host.
#   OPENCLAW_NPM_PACKAGE     — default: openclaw
#   MEMOS_CLOUD_NPM_PACKAGE  — default: @memtensor/memos-cloud-openclaw-plugin
#   MEMOS_LOCAL_NPM_PACKAGE  — default: @memtensor/memos-local-openclaw-plugin
#   MEMOS_LOCOMO_VENV        — if set, source bin/activate before eval
#
# Cron example (run as the same user as openclaw gateway):
#   0 3 * * * MEMOS_EVAL_ROOT=/path/to/MemOS/evaluation MEMOS_LOCOMO_VENV=/path/to/venv /path/to/locomo_openclaw_versioned_eval.sh >>/var/log/memos-locomo-eval.log 2>&1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="${MEMOS_EVAL_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
STATE_DIR="${MEMOS_LOCOMO_STATE_DIR:-/var/lib/memos-locomo-eval}"
STATE_FILE="${MEMOS_LOCOMO_STATE_FILE:-$STATE_DIR/last-versions.json}"

OC_PKG="${OPENCLAW_NPM_PACKAGE:-openclaw}"
CLOUD_PKG="${MEMOS_CLOUD_NPM_PACKAGE:-@memtensor/memos-cloud-openclaw-plugin}"
LOCAL_PKG="${MEMOS_LOCAL_NPM_PACKAGE:-@memtensor/memos-local-openclaw-plugin}"

PROFILE_RAW="${MEMOS_LOCOMO_PROFILE:-both}"
PROFILE="$(printf '%s' "$PROFILE_RAW" | tr '[:upper:]' '[:lower:]')"
case "$PROFILE" in
  both | cloud | local | openclaw) ;;
  *)
    echo "error: MEMOS_LOCOMO_PROFILE must be both, cloud, local, or openclaw (got $PROFILE_RAW)" >&2
    exit 1
    ;;
esac

RUN_SCRIPT="$EVAL_ROOT/scripts/run_locomo_openclaw_eval.sh"

npm_latest() {
  npm view "$1" version 2>/dev/null | tr -d '\r\n' || true
}

read_state_versions() {
  local f="$1"
  python3 - "$f" <<'PY'
import json
import sys

path = sys.argv[1]
try:
    with open(path, encoding="utf-8") as fp:
        d = json.load(fp)
except (OSError, json.JSONDecodeError):
    d = {}
for k in ("openclaw", "cloud", "local"):
    print(d.get(k, "") or "")
PY
}

write_state_versions() {
  local f="$1" oc="$2" cloud="$3" local_v="$4" prof="$5"
  mkdir -p "$(dirname "$f")"
  python3 - "$f" "$oc" "$cloud" "$local_v" "$prof" <<'PY'
import json
import sys

path, oc, cloud, local_v, profile = sys.argv[1:6]
try:
    with open(path, encoding="utf-8") as fp:
        prev = json.load(fp)
except (OSError, json.JSONDecodeError):
    prev = {}
if profile == "both":
    out = {"openclaw": oc, "cloud": cloud, "local": local_v}
elif profile == "cloud":
    out = {"openclaw": oc, "cloud": cloud, "local": prev.get("local", "")}
elif profile == "openclaw":
    out = {"openclaw": oc, "cloud": prev.get("cloud", ""), "local": prev.get("local", "")}
elif profile == "local":
    out = {"openclaw": oc, "cloud": prev.get("cloud", ""), "local": local_v}
else:
    out = {"openclaw": oc, "cloud": prev.get("cloud", ""), "local": local_v}
with open(path, "w", encoding="utf-8") as fp:
    json.dump(out, fp, indent=2)
    fp.write("\n")
PY
}

if [[ ! -f "$RUN_SCRIPT" ]]; then
  echo "error: run script not found: $RUN_SCRIPT (set MEMOS_EVAL_ROOT?)" >&2
  exit 1
fi

if [[ -n "${MEMOS_LOCOMO_VENV:-}" ]]; then
  # shellcheck source=/dev/null
  source "$MEMOS_LOCOMO_VENV/bin/activate"
fi

oc_new="$(npm_latest "$OC_PKG")"
cloud_new=""
local_new=""
if [[ "$PROFILE" == "both" || "$PROFILE" == "cloud" ]]; then
  cloud_new="$(npm_latest "$CLOUD_PKG")"
fi
if [[ "$PROFILE" == "both" || "$PROFILE" == "local" ]]; then
  local_new="$(npm_latest "$LOCAL_PKG")"
fi
# openclaw profile: only OPENCLAW_NPM_PACKAGE; cloud_new/local_new stay empty

if [[ -z "$oc_new" ]]; then
  echo "error: failed to read openclaw npm version" >&2
  exit 1
fi
if [[ "$PROFILE" == "both" || "$PROFILE" == "cloud" ]] && [[ -z "$cloud_new" ]]; then
  echo "error: failed to read cloud plugin npm version" >&2
  exit 1
fi
if [[ "$PROFILE" == "both" || "$PROFILE" == "local" ]] && [[ -z "$local_new" ]]; then
  echo "error: failed to read local plugin npm version" >&2
  exit 1
fi

oc_old="" cloud_old="" local_old=""
if [[ -f "$STATE_FILE" ]]; then
  mapfile -t _st < <(read_state_versions "$STATE_FILE")
  oc_old="${_st[0]:-}"
  cloud_old="${_st[1]:-}"
  local_old="${_st[2]:-}"
fi

state_ok=0
if [[ "$PROFILE" == "both" ]]; then
  [[ -n "$oc_old" && -n "$cloud_old" && -n "$local_old" ]] && state_ok=1
elif [[ "$PROFILE" == "cloud" ]]; then
  [[ -n "$oc_old" && -n "$cloud_old" ]] && state_ok=1
elif [[ "$PROFILE" == "openclaw" ]]; then
  [[ -n "$oc_old" ]] && state_ok=1
else
  [[ -n "$oc_old" && -n "$local_old" ]] && state_ok=1
fi

oc_changed=0
cloud_changed=0
local_changed=0
if [[ $state_ok -eq 0 ]]; then
  oc_changed=1
  if [[ "$PROFILE" == "openclaw" ]]; then
    :
  else
    [[ "$PROFILE" == "both" || "$PROFILE" == "cloud" ]] && cloud_changed=1
    [[ "$PROFILE" == "both" || "$PROFILE" == "local" ]] && local_changed=1
  fi
  echo "no valid state at $STATE_FILE — treating tracked packages as changed"
else
  [[ "$oc_old" != "$oc_new" ]] && oc_changed=1
  if [[ "$PROFILE" == "both" || "$PROFILE" == "cloud" ]]; then
    [[ "$cloud_old" != "$cloud_new" ]] && cloud_changed=1
  fi
  if [[ "$PROFILE" == "both" || "$PROFILE" == "local" ]]; then
    [[ "$local_old" != "$local_new" ]] && local_changed=1
  fi
fi

if [[ $oc_changed -eq 0 && $cloud_changed -eq 0 && $local_changed -eq 0 ]]; then
  echo "npm versions unchanged; skip install and eval."
  exit 0
fi

echo "profile=$PROFILE | openclaw $oc_old -> $oc_new | cloud $cloud_old -> $cloud_new | local $local_old -> $local_new"

if [[ $oc_changed -eq 1 ]]; then
  echo "npm install -g ${OC_PKG}@${oc_new}"
  npm install -g "${OC_PKG}@${oc_new}"
  if [[ "$PROFILE" == "openclaw" ]]; then
    echo "openclaw profile: skip MemOS plugin reinstall (native memory-core only)"
  else
    echo "reinstall plugin(s) after openclaw upgrade"
    if [[ "$PROFILE" == "both" ]]; then
      openclaw plugins install "${CLOUD_PKG}@${cloud_new}"
      openclaw plugins install "${LOCAL_PKG}@${local_new}"
    elif [[ "$PROFILE" == "cloud" ]]; then
      openclaw plugins install "${CLOUD_PKG}@${cloud_new}"
    else
      openclaw plugins install "${LOCAL_PKG}@${local_new}"
    fi
  fi
else
  if [[ $cloud_changed -eq 1 && ( "$PROFILE" == "both" || "$PROFILE" == "cloud" ) ]]; then
    echo "openclaw plugins install ${CLOUD_PKG}@${cloud_new}"
    openclaw plugins install "${CLOUD_PKG}@${cloud_new}"
  fi
  if [[ $local_changed -eq 1 && ( "$PROFILE" == "both" || "$PROFILE" == "local" ) ]]; then
    echo "openclaw plugins install ${LOCAL_PKG}@${local_new}"
    openclaw plugins install "${LOCAL_PKG}@${local_new}"
  fi
fi

echo "openclaw gateway restart"
openclaw gateway restart
sleep 15

run_cloud=0
run_local=0
run_openclaw_native=0
if [[ "$PROFILE" == "openclaw" ]]; then
  if [[ $oc_changed -eq 1 ]]; then
    run_openclaw_native=1
  fi
fi
if [[ "$PROFILE" == "both" || "$PROFILE" == "cloud" ]]; then
  if [[ $oc_changed -eq 1 || $cloud_changed -eq 1 ]]; then
    run_cloud=1
  fi
fi
if [[ "$PROFILE" == "both" || "$PROFILE" == "local" ]]; then
  if [[ $oc_changed -eq 1 || $local_changed -eq 1 ]]; then
    run_local=1
  fi
fi

day="$(date +%Y%m%d)"
ver_slug="oc-${oc_new}"
if [[ "$PROFILE" == "both" || "$PROFILE" == "cloud" ]]; then
  ver_slug="${ver_slug}_cloud-${cloud_new}"
fi
if [[ "$PROFILE" == "both" || "$PROFILE" == "local" ]]; then
  ver_slug="${ver_slug}_local-${local_new}"
fi
if [[ "$PROFILE" == "openclaw" ]]; then
  ver_slug="${ver_slug}_memory-core"
fi
ver_slug="${ver_slug//\//_}"
export VERSION="${day}_${ver_slug}"

cd "$EVAL_ROOT"

if [[ $run_openclaw_native -eq 1 ]]; then
  echo "=== LOCOMO eval openclaw (native memory-core) VERSION=$VERSION ==="
  LIB=openclaw "$RUN_SCRIPT"
fi
if [[ $run_cloud -eq 1 ]]; then
  echo "=== LOCOMO eval memos-cloud VERSION=$VERSION ==="
  LIB=memos-cloud "$RUN_SCRIPT"
fi
if [[ $run_local -eq 1 ]]; then
  echo "=== LOCOMO eval memos-local VERSION=$VERSION ==="
  LIB=memos-local "$RUN_SCRIPT"
fi

write_state_versions "$STATE_FILE" "$oc_new" "$cloud_new" "$local_new" "$PROFILE"
echo "state updated: $STATE_FILE"
echo "done."
