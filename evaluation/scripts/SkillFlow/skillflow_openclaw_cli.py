"""Shared OpenClaw CLI helpers for SkillFlow evaluation scripts."""
from __future__ import annotations

import os


def openclaw_subprocess_env() -> dict[str, str]:
    """Prefer system Node 24+ for OpenClaw CLI (Cursor shells often default to v20)."""
    env = os.environ.copy()
    preferred = ("/usr/bin", "/usr/local/bin")
    current = env.get("PATH", "")
    parts = [p for p in current.split(":") if p]
    env["PATH"] = ":".join([*preferred, *parts])
    return env
