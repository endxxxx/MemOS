"""MemOS local plugin support for SkillFlow OpenClaw evaluation.

Handles per-family SQLite isolation, training snapshots, gateway restarts,
and journal polling so background reflect/reward/L2/L3 work finishes before
the next OpenClaw request.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.SkillFlow.skillflow_openclaw_cli import openclaw_subprocess_env


SKILLFLOW_OUTPUT_ROOT = Path("/root/SkillFlow_Outputs")
MEMOS_LOCAL_SNAPSHOT_ROOT = SKILLFLOW_OUTPUT_ROOT / ".memos-local-db-snapshots"
MEMOS_LOCAL_DB_SIDECAR_SUFFIXES = ("-wal", "-shm")
MEMOS_LOCAL_AFTER_TRAINING_LABEL = "after_training"
OPENCLAW_GATEWAY_RESTART_WAIT_SECONDS = 10

# MemOS background pipeline logs use [core.*] channels and often omit "memos-local-plugin".
PIPELINE_ACTIVITY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"init\.dirty_closed_episodes\.rescore"),
    re.compile(r"init\.orphan_episodes"),
    re.compile(r"l3\.abstraction"),
    re.compile(r"\[core\.memory\.l2\]|\[core\.memory\.l3|memory\.l2"),
    re.compile(r"\[core\.capture\]|capture\.reflect|capture\.reward|capture\.lite"),
    re.compile(r"\[core\.pipeline"),
    re.compile(r"episode\.finalized"),
    re.compile(r"dirty_closed_reward"),
    re.compile(r"abstract\.llm_failed"),
)

PIPELINE_CONTEXT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"memos-local-plugin"),
    re.compile(r"memos-local:"),
    re.compile(r"\[core\.memory\.|\[core\.capture\]|\[core\.pipeline|\[core\.episode\]"),
    re.compile(r"\[llm\.json\].*l3\.abstraction"),
)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class PipelineSettleConfig:
    enabled: bool = True
    poll_seconds: float = 2.0
    training_quiet_seconds: float = 15.0
    training_max_seconds: float = 180.0
    training_min_seconds: float = 5.0
    restart_quiet_seconds: float = 8.0
    restart_max_seconds: float = 60.0
    restart_min_seconds: float = 3.0

    @classmethod
    def from_env(cls) -> PipelineSettleConfig:
        legacy_max = os.getenv("MEMOS_PIPELINE_SETTLE_MAX_SEC")
        legacy_quiet = os.getenv("MEMOS_PIPELINE_SETTLE_QUIET_SEC")
        return cls(
            enabled=_env_bool("MEMOS_PIPELINE_SETTLE_ENABLED", True),
            poll_seconds=_env_float("MEMOS_PIPELINE_SETTLE_POLL_SEC", 2.0),
            training_quiet_seconds=_env_float(
                "MEMOS_PIPELINE_SETTLE_TRAINING_QUIET_SEC",
                float(legacy_quiet) if legacy_quiet else 15.0,
            ),
            training_max_seconds=_env_float(
                "MEMOS_PIPELINE_SETTLE_TRAINING_MAX_SEC",
                float(legacy_max) if legacy_max else 180.0,
            ),
            training_min_seconds=_env_float("MEMOS_PIPELINE_SETTLE_TRAINING_MIN_SEC", 5.0),
            restart_quiet_seconds=_env_float(
                "MEMOS_PIPELINE_SETTLE_RESTART_QUIET_SEC",
                float(legacy_quiet) if legacy_quiet else 8.0,
            ),
            restart_max_seconds=_env_float(
                "MEMOS_PIPELINE_SETTLE_RESTART_MAX_SEC",
                float(legacy_max) if legacy_max else 60.0,
            ),
            restart_min_seconds=_env_float("MEMOS_PIPELINE_SETTLE_RESTART_MIN_SEC", 3.0),
        )

    def for_phase(self, phase: str) -> tuple[float, float, float]:
        if phase == "training":
            return (
                self.training_quiet_seconds,
                self.training_max_seconds,
                self.training_min_seconds,
            )
        return (
            self.restart_quiet_seconds,
            self.restart_max_seconds,
            self.restart_min_seconds,
        )


def should_manage_memos_local_db(client_type: str) -> bool:
    return str(client_type).lower() == "memos-local"


def safe_session_fragment(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


def resolve_memos_local_home() -> Path:
    env_home = os.getenv("MEMOS_HOME")
    if env_home and env_home.strip():
        return Path(env_home).expanduser().resolve()

    env_config = os.getenv("MEMOS_CONFIG_FILE")
    if env_config and env_config.strip():
        return Path(env_config).expanduser().resolve().parent

    return Path.home() / ".openclaw" / "memos-plugin"


def resolve_memos_local_db_file() -> Path:
    return resolve_memos_local_home() / "data" / "memos.db"


def journal_since(iso_start: str) -> str:
    proc = subprocess.run(
        [
            "journalctl",
            "--user-unit=openclaw-gateway",
            f"--since={iso_start}",
            "--no-pager",
            "-q",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return proc.stdout or ""


def journal_line_has_pipeline_activity(line: str) -> bool:
    if not any(pattern.search(line) for pattern in PIPELINE_CONTEXT_PATTERNS):
        return False
    return any(pattern.search(line) for pattern in PIPELINE_ACTIVITY_PATTERNS)


def journal_has_pipeline_activity(text: str) -> bool:
    return any(journal_line_has_pipeline_activity(line) for line in text.splitlines())


def wait_for_memos_pipeline_settle(
    reason: str,
    *,
    phase: str = "restart",
    config: PipelineSettleConfig | None = None,
) -> dict[str, Any]:
    """Poll gateway journal until MemOS background pipeline appears idle."""
    cfg = config or PipelineSettleConfig.from_env()
    quiet_seconds, max_seconds, min_seconds = cfg.for_phase(phase)
    started_at = datetime.now(timezone.utc)
    poll_since = started_at.isoformat()

    if not cfg.enabled:
        print(f"[memos-local-db] pipeline settle skipped ({reason})")
        return {
            "ok": True,
            "reason": reason,
            "phase": phase,
            "skipped": True,
            "waited_seconds": 0.0,
            "poll_since": poll_since,
        }

    print(
        "[memos-local-db] waiting for pipeline to settle "
        f"({reason}; phase={phase} quiet={quiet_seconds}s max={max_seconds}s)"
    )

    last_activity_at = started_at
    deadline = time.monotonic() + max_seconds
    saw_activity = False
    polls = 0

    while time.monotonic() < deadline:
        polls += 1
        journal = journal_since(poll_since)
        if journal_has_pipeline_activity(journal):
            saw_activity = True
            last_activity_at = datetime.now(timezone.utc)

        elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
        quiet_for = (datetime.now(timezone.utc) - last_activity_at).total_seconds()
        min_wait_met = elapsed >= min_seconds
        quiet_met = quiet_for >= quiet_seconds

        if min_wait_met and quiet_met and (saw_activity or elapsed >= quiet_seconds):
            print(
                "[memos-local-db] pipeline settled "
                f"({reason}; phase={phase} waited={elapsed:.1f}s "
                f"polls={polls} saw_activity={saw_activity})"
            )
            return {
                "ok": True,
                "reason": reason,
                "phase": phase,
                "skipped": False,
                "waited_seconds": round(elapsed, 2),
                "polls": polls,
                "saw_activity": saw_activity,
                "poll_since": poll_since,
            }

        time.sleep(cfg.poll_seconds)

    elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
    print(
        "[memos-local-db] pipeline settle timed out "
        f"({reason}; phase={phase} waited={elapsed:.1f}s "
        f"polls={polls} saw_activity={saw_activity})"
    )
    return {
        "ok": False,
        "reason": reason,
        "phase": phase,
        "skipped": False,
        "timed_out": True,
        "waited_seconds": round(elapsed, 2),
        "polls": polls,
        "saw_activity": saw_activity,
        "poll_since": poll_since,
    }


def restart_openclaw_gateway(reason: str) -> None:
    print(f"[memos-local-db] restarting OpenClaw gateway after {reason}")
    subprocess.run(
        ["openclaw", "gateway", "restart"],
        check=True,
        env=openclaw_subprocess_env(),
    )
    time.sleep(OPENCLAW_GATEWAY_RESTART_WAIT_SECONDS)


class MemosLocalEvalSupport:
    """Per-family memos-local DB lifecycle for one SkillFlow task family."""

    def __init__(
        self,
        *,
        client_type: str,
        version: str,
        task_family_name: str,
        settle_config: PipelineSettleConfig | None = None,
    ) -> None:
        self.client_type = client_type
        self.version = version
        self.task_family_name = task_family_name
        self.settle_config = settle_config or PipelineSettleConfig.from_env()
        self._prep: dict[str, Any] | None = None
        self._snapshot: dict[str, Any] | None = None

    @property
    def enabled(self) -> bool:
        return should_manage_memos_local_db(self.client_type)

    @property
    def prep(self) -> dict[str, Any] | None:
        return self._prep

    @property
    def snapshot(self) -> dict[str, Any] | None:
        return self._snapshot

    def family_storage_name(self) -> str:
        return safe_session_fragment(f"{self.client_type}_{self.version}_{self.task_family_name}")

    def family_dir(self) -> Path:
        return MEMOS_LOCAL_SNAPSHOT_ROOT / self.family_storage_name()

    def family_db_file(self) -> Path:
        return self.family_dir() / "memos.db"

    def training_snapshot_dir(self) -> Path:
        return self.family_dir() / MEMOS_LOCAL_AFTER_TRAINING_LABEL

    def legacy_snapshot_db(self, source_label: str) -> Path:
        snapshot_name = safe_session_fragment(
            f"{self.client_type}_{self.version}_{self.task_family_name}_{source_label}"
        )
        return MEMOS_LOCAL_SNAPSHOT_ROOT / snapshot_name / "memos.db"

    def setup(self, *, only_test: bool) -> None:
        if not self.enabled:
            return
        if only_test:
            self._snapshot = self._bootstrap_for_only_test()
        else:
            self._prep = self._prepare_fresh_family_db()

    def finalize_training(self) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("memos-local support is disabled for this client")
        # Primary settle point: drain reflect/reward/L2/L3 while gateway stays up,
        # then snapshot a clean DB for lighter only_test bootstrap later.
        settle = self.wait_pipeline_settle(
            "post-training before snapshot",
            phase="training",
        )
        self._snapshot = self._create_snapshot(
            source_label=MEMOS_LOCAL_AFTER_TRAINING_LABEL,
            settle_before_copy=False,
        )
        self._snapshot["pipeline_settle"] = settle
        self._snapshot["snapshot_after_pipeline_settle"] = True
        return self._snapshot

    def wait_pipeline_settle(self, reason: str, *, phase: str = "restart") -> dict[str, Any]:
        return wait_for_memos_pipeline_settle(
            reason,
            phase=phase,
            config=self.settle_config,
        )

    def restore_after_test(self, context: str) -> dict[str, Any]:
        if not self.enabled or self._snapshot is None:
            raise RuntimeError("memos-local snapshot is not available")
        return self._restore_snapshot(self._snapshot, context)

    def _prepare_fresh_family_db(self) -> dict[str, Any]:
        family_dir = self.family_dir()
        if family_dir.exists():
            shutil.rmtree(family_dir)
        family_dir.mkdir(parents=True, exist_ok=True)

        family_db_file = self.family_db_file()
        _create_empty_sqlite_db(family_db_file)
        live_db_file = self._install_db_file(
            family_db_file,
            f"fresh family db {self.task_family_name}",
        )
        prepared_at = datetime.now(timezone.utc).isoformat()
        print(
            "[memos-local-db] prepared isolated family database at "
            f"{family_db_file} (live={live_db_file})"
        )
        return {
            "enabled": True,
            "mode": "fresh",
            "client_type": self.client_type,
            "version": self.version,
            "task_family_name": self.task_family_name,
            "family_dir": str(family_dir),
            "family_db_file": str(family_db_file),
            "db_file": str(live_db_file),
            "prepared_at": prepared_at,
        }

    def _bootstrap_for_only_test(self) -> dict[str, Any]:
        snapshot = self._load_snapshot(MEMOS_LOCAL_AFTER_TRAINING_LABEL)
        self._install_db_file(
            Path(str(snapshot["snapshot_db_file"])),
            f"only_test bootstrap {self.task_family_name}",
        )
        print(
            "[memos-local-db] only_test loaded training snapshot from "
            f"{snapshot['snapshot_db_file']}"
        )
        return snapshot

    def _create_snapshot(
        self,
        source_label: str,
        *,
        settle_before_copy: bool = True,
    ) -> dict[str, Any]:
        if settle_before_copy:
            self.wait_pipeline_settle(
                f"before snapshot ({source_label})",
                phase="training",
            )

        live_db_file = resolve_memos_local_db_file()
        if not live_db_file.is_file():
            raise FileNotFoundError(f"memos-local database not found: {live_db_file}")

        family_dir = self.family_dir()
        family_dir.mkdir(parents=True, exist_ok=True)
        family_db_file = self.family_db_file()
        snapshot_dir = family_dir / source_label
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_db_file = snapshot_dir / "memos.db"

        _backup_sqlite_db(live_db_file, snapshot_db_file)
        _backup_sqlite_db(live_db_file, family_db_file)

        metadata = self._build_snapshot_metadata(
            source_label=source_label,
            family_dir=family_dir,
            family_db_file=family_db_file,
            snapshot_dir=snapshot_dir,
            snapshot_db_file=snapshot_db_file,
            live_db_file=live_db_file,
        )
        print(
            "[memos-local-db] saved family snapshot "
            f"({source_label}) to {snapshot_db_file}"
        )
        return metadata

    def _load_snapshot(self, source_label: str = MEMOS_LOCAL_AFTER_TRAINING_LABEL) -> dict[str, Any]:
        snapshot_dir = self.family_dir() / source_label
        metadata_path = snapshot_dir / "metadata.json"
        if metadata_path.is_file():
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            if not isinstance(metadata, dict):
                raise ValueError(f"Invalid snapshot metadata: {metadata_path}")
            return metadata

        snapshot_db_file = snapshot_dir / "memos.db"
        if not snapshot_db_file.is_file():
            legacy_snapshot_db_file = self.legacy_snapshot_db(source_label)
            if legacy_snapshot_db_file.is_file():
                snapshot_db_file = legacy_snapshot_db_file
                snapshot_dir = legacy_snapshot_db_file.parent
            else:
                raise FileNotFoundError(
                    "memos-local training snapshot not found for "
                    f"{self.task_family_name} ({self.client_type}/{self.version}). "
                    f"Expected {snapshot_dir / 'memos.db'} or legacy "
                    f"{legacy_snapshot_db_file}. Run training for this family first."
                )

        live_db_file = resolve_memos_local_db_file()
        return self._build_snapshot_metadata(
            source_label=source_label,
            family_dir=self.family_dir(),
            family_db_file=self.family_db_file(),
            snapshot_dir=snapshot_dir,
            snapshot_db_file=snapshot_db_file,
            live_db_file=live_db_file,
        )

    def _build_snapshot_metadata(
        self,
        *,
        source_label: str,
        family_dir: Path,
        family_db_file: Path,
        snapshot_dir: Path,
        snapshot_db_file: Path,
        live_db_file: Path,
    ) -> dict[str, Any]:
        metadata = {
            "enabled": True,
            "source_label": source_label,
            "client_type": self.client_type,
            "version": self.version,
            "task_family_name": self.task_family_name,
            "home": str(resolve_memos_local_home()),
            "family_dir": str(family_dir),
            "family_db_file": str(family_db_file),
            "db_file": str(live_db_file),
            "snapshot_dir": str(snapshot_dir),
            "snapshot_db_file": str(snapshot_db_file),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        metadata_path = snapshot_dir / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)
        metadata["metadata_file"] = str(metadata_path)
        return metadata

    def _restore_snapshot(self, snapshot: dict[str, Any], context: str) -> dict[str, Any]:
        snapshot_db_file = Path(str(snapshot["snapshot_db_file"]))
        live_db_file = self._install_db_file(snapshot_db_file, context)
        return {
            "ok": True,
            "context": context,
            "db_file": str(live_db_file),
            "snapshot_db_file": str(snapshot_db_file),
            "restored_at": datetime.now(timezone.utc).isoformat(),
        }

    def _install_db_file(self, source_db: Path, context: str) -> Path:
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
        # Light settle after restart; heavy work should already be done at training time.
        self.wait_pipeline_settle(f"post-restart {context}", phase="restart")
        return live_db_file


def _create_empty_sqlite_db(db_file: Path) -> None:
    db_file.parent.mkdir(parents=True, exist_ok=True)
    if db_file.exists():
        db_file.unlink()
    with sqlite3.connect(str(db_file), timeout=30) as conn:
        conn.execute("SELECT 1")


def _backup_sqlite_db(source_db: Path, target_db: Path) -> None:
    if not source_db.is_file():
        raise FileNotFoundError(f"memos-local database not found: {source_db}")

    target_db.parent.mkdir(parents=True, exist_ok=True)
    tmp_target_db = target_db.with_name(f".{target_db.name}.backup.tmp")
    if tmp_target_db.exists():
        tmp_target_db.unlink()

    source_uri = f"{source_db.resolve().as_uri()}?mode=ro"
    with (
        sqlite3.connect(source_uri, uri=True, timeout=30) as source,
        sqlite3.connect(str(tmp_target_db), timeout=30) as target,
    ):
        source.backup(target)
    shutil.move(str(tmp_target_db), str(target_db))
    for suffix in MEMOS_LOCAL_DB_SIDECAR_SUFFIXES:
        with suppress(FileNotFoundError):
            Path(str(target_db) + suffix).unlink()


def _remove_memos_db_sidecars(db_file: Path) -> None:
    for suffix in MEMOS_LOCAL_DB_SIDECAR_SUFFIXES:
        with suppress(FileNotFoundError):
            Path(str(db_file) + suffix).unlink()
