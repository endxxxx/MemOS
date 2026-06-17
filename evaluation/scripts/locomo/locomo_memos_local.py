"""Per-user SQLite isolation helpers for memos-local-plugin LoCoMo evaluation."""
from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts.SkillFlow.skillflow_memos_local import (
    _backup_sqlite_db,
    _create_empty_sqlite_db,
    _remove_memos_db_sidecars,
    ensure_memos_local_bpp_timeout_ms,
    journal_since,
    restart_openclaw_gateway,
    resolve_memos_local_db_file,
    safe_session_fragment,
    wait_for_memos_pipeline_settle,
)

MEMOS_LOCAL_CLIENT_TYPES = frozenset({"memos-local", "memos-local-plugin"})
MEMOS_BPP_WARMUP_QUERY = "Reply with exactly OK."
MEMOS_BPP_WARMUP_SESSION_SUFFIX = "_bpp_warmup"


def is_memos_local_client(client_type: str) -> bool:
    return str(client_type).lower() in MEMOS_LOCAL_CLIENT_TYPES


def locomo_session_key(user_idx: int, version: str) -> str:
    """Legacy per-user session key (avoid for LoCoMo stage 1/2)."""
    return safe_session_fragment(f"locomo_u{user_idx}_{version}")


def locomo_add_session_key(user_idx: int, version: str, batch_num: int) -> str:
    return safe_session_fragment(f"locomo_u{user_idx}_{version}_add_{batch_num}")


def locomo_qa_session_key(user_idx: int, version: str, qa_idx: int) -> str:
    return safe_session_fragment(f"locomo_u{user_idx}_{version}_qa_{qa_idx}")


def locomo_bpp_warmup_session_key(user_idx: int, version: str) -> str:
    return safe_session_fragment(f"locomo_u{user_idx}_{version}_bpp_warmup")


def user_db_checkpoint_path(results_dir: str, user_idx: int) -> Path:
    return Path(results_dir) / ".user-dbs" / f"user_{user_idx}" / "memos.db"


def better_sqlite3_binding_path() -> Path:
    return (
        Path.home()
        / ".openclaw"
        / "npm"
        / "node_modules"
        / "better-sqlite3"
        / "build"
        / "Release"
        / "better_sqlite3.node"
    )


def ensure_better_sqlite3_bindings(*, force: bool = False) -> None:
    binding = better_sqlite3_binding_path()
    if binding.is_file() and not force:
        return

    npm_root = Path.home() / ".openclaw" / "npm"
    print("[memos-local] rebuilding better-sqlite3 native bindings")
    subprocess.run(
        ["npm", "rebuild", "better-sqlite3"],
        cwd=str(npm_root),
        check=True,
    )
    if not binding.is_file():
        raise RuntimeError(
            f"better-sqlite3 bindings still missing after rebuild: {binding}"
        )


def _journal_plugin_bootstrap_status(journal: str) -> str | None:
    if "memos-local: plugin ready" in journal:
        return "ready"
    if "memos-local: bootstrap failed" in journal:
        return "failed"
    if "plugin service failed" in journal and "memos-local-plugin" in journal:
        return "failed"
    return None


def _recent_gateway_journal(lines: int = 120) -> str:
    proc = subprocess.run(
        [
            "journalctl",
            "--user-unit=openclaw-gateway",
            "-n",
            str(lines),
            "--no-pager",
            "-q",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return proc.stdout or ""


def ensure_memos_plugin_ready(*, max_attempts: int = 2) -> None:
    """Ensure memos-local-plugin finished bootstrap (needs better-sqlite3)."""
    ensure_better_sqlite3_bindings()
    last_journal = ""

    for attempt in range(1, max_attempts + 1):
        poll_since = datetime.now(timezone.utc).isoformat()
        restart_openclaw_gateway(
            f"memos-local-plugin bootstrap verify (attempt {attempt})"
        )
        ensure_memos_local_bpp_timeout_ms()
        wait_for_memos_pipeline_settle(
            f"memos-local-plugin bootstrap verify (attempt {attempt})",
            phase="restart",
        )
        time.sleep(3)
        last_journal = journal_since(poll_since)
        status = _journal_plugin_bootstrap_status(last_journal)
        if status is None:
            last_journal = _recent_gateway_journal()
            status = _journal_plugin_bootstrap_status(last_journal)
        if status == "ready":
            print("[memos-local] plugin bootstrap verified (plugin ready)")
            return
        if status == "failed" and (
            "bindings file" in last_journal or "better_sqlite3" in last_journal
        ):
            print("[memos-local] plugin bootstrap failed due to sqlite bindings; rebuilding")
            ensure_better_sqlite3_bindings(force=True)
            continue

    snippet = "\n".join(
        line
        for line in last_journal.splitlines()
        if "memos-local" in line or "better_sqlite3" in line
    )[:2000]
    raise RuntimeError(
        "memos-local-plugin bootstrap did not reach plugin ready. "
        f"Recent gateway log snippet:\n{snippet or last_journal[-2000:]}"
    )


def memos_db_table_names(db_file: Path | None = None) -> list[str]:
    db_path = db_file or resolve_memos_local_db_file()
    if not db_path.is_file() or db_path.stat().st_size == 0:
        return []
    with sqlite3.connect(str(db_path), timeout=30) as conn:
        return [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ]


def settle_after_user_stage1(user_idx: int) -> dict:
    """Wait for memos background pipeline after all Stage-1 batches for one user."""
    return wait_for_memos_pipeline_settle(
        f"locomo user {user_idx} stage1 complete",
        phase="training",
    )


def _openclaw_config_get(path: str) -> str | None:
    import subprocess

    proc = subprocess.run(
        ["openclaw", "config", "get", path],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def configure_memos_local_plugin_openclaw() -> None:
    """Point OpenClaw memory slot at memos-local-plugin for evaluation."""
    memory_slot = _openclaw_config_get("plugins.slots.memory")
    memos_enabled = _openclaw_config_get("plugins.entries.memos-local-plugin.enabled")
    memory_core_enabled = _openclaw_config_get("plugins.entries.memory-core.enabled")
    config_ok = (
        memory_slot == "memos-local-plugin"
        and memos_enabled == "true"
        and memory_core_enabled == "false"
    )
    if not config_ok:
        cmds = [
            "openclaw config set plugins.slots.memory memos-local-plugin",
            "openclaw config set plugins.entries.memos-local-plugin.enabled true",
            "openclaw config set plugins.entries.memory-core.enabled false",
        ]
        for cmd in cmds:
            subprocess_run_checked(cmd)
    else:
        print("memos-local-plugin already configured, skipping plugin setup")

    ensure_memos_plugin_ready()


def subprocess_run_checked(cmd: str) -> None:
    import subprocess
    import time

    for attempt in range(3):
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            return
        combined = f"{proc.stdout}\n{proc.stderr}"
        if "ConfigMutationConflictError" in combined and attempt < 2:
            time.sleep(2)
            continue
        proc.check_returncode()


def install_user_db(source_db: Path, context: str) -> Path:
    if not source_db.is_file():
        raise FileNotFoundError(f"memos-local database not found: {source_db}")

    live_db_file = resolve_memos_local_db_file()
    live_db_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_restore_db_file = live_db_file.with_name(f".{live_db_file.name}.restore.tmp")
    if tmp_restore_db_file.exists():
        tmp_restore_db_file.unlink()

    shutil.copy2(source_db, tmp_restore_db_file)
    _remove_memos_db_sidecars(live_db_file)
    os.replace(tmp_restore_db_file, live_db_file)
    _remove_memos_db_sidecars(live_db_file)
    restart_openclaw_gateway(context)
    wait_for_memos_pipeline_settle(f"post-restart {context}", phase="restart")
    return live_db_file


def prepare_user_db(results_dir: str, user_idx: int) -> dict:
    checkpoint = user_db_checkpoint_path(results_dir, user_idx)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint.is_file():
        mode = "resume"
    else:
        _create_empty_sqlite_db(checkpoint)
        mode = "fresh"

    live_db = install_user_db(checkpoint, f"locomo user {user_idx} prepare")
    print(
        f"[memos-local-db] prepared user {user_idx} database "
        f"(mode={mode}, checkpoint={checkpoint}, live={live_db})"
    )
    return {
        "mode": mode,
        "user_idx": user_idx,
        "checkpoint": str(checkpoint),
        "live_db": str(live_db),
    }


def save_user_db_checkpoint(results_dir: str, user_idx: int) -> Path:
    live_db_file = resolve_memos_local_db_file()
    tables = memos_db_table_names(live_db_file)
    if not tables:
        raise RuntimeError(
            f"memos-local live database has no tables before checkpoint "
            f"(user={user_idx}, db={live_db_file}). "
            "Plugin bootstrap or post-add pipeline settle likely failed."
        )
    checkpoint = user_db_checkpoint_path(results_dir, user_idx)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    _backup_sqlite_db(live_db_file, checkpoint)
    print(
        f"[memos-local-db] checkpoint tables={len(tables)} "
        f"size={checkpoint.stat().st_size} bytes"
    )
    return checkpoint


def warmup_memos_bpp_hook(client, agent_id: str, session_key: str, *, context: str) -> dict:
    if os.getenv("MEMOS_BPP_WARMUP_ENABLED", "true").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        print(f"[memos-local] bpp warmup skipped ({context})")
        return {"skipped": True, "context": context}

    if session_key.endswith(MEMOS_BPP_WARMUP_SESSION_SUFFIX):
        warmup_session_key = safe_session_fragment(session_key)
    else:
        warmup_session_key = safe_session_fragment(
            f"{session_key}{MEMOS_BPP_WARMUP_SESSION_SUFFIX}"
        )
    print(f"[memos-local] warming up before_prompt_build hook ({context})")
    started_at = time.time()
    error = None
    try:
        client.search(
            query=MEMOS_BPP_WARMUP_QUERY,
            user_id=agent_id,
            top_k=0,
            session_key=warmup_session_key,
        )
    except Exception as exc:
        error = str(exc)
        print(f"[memos-local] bpp warmup request failed ({context}): {exc}")
    duration_s = time.time() - started_at
    print(f"[memos-local] bpp warmup finished in {duration_s:.1f}s ({context})")
    return {
        "skipped": False,
        "context": context,
        "session_key": warmup_session_key,
        "duration_seconds": round(duration_s, 2),
        "error": error,
    }
