"""OpenClaw memory plugin configuration helpers for SkillFlow evaluation."""
from __future__ import annotations

import subprocess
import time
from typing import Any

from scripts.SkillFlow.skillflow_openclaw_cli import openclaw_subprocess_env

PLUGIN_NAME_BY_CLIENT: dict[str, str] = {
    "mem0": "openclaw-mem0",
    "memos-cloud": "memos-cloud-openclaw-plugin",
    "memos-local": "memos-local-plugin",
    "mem9": "mem9",
    "openviking": "openviking",
    "supermemory": "openclaw-supermemory",
    "memorylake": "memorylake-openclaw",
    "honcho": "openclaw-honcho",
    "byterover": "byterover",
    "tencentdb": "memory-tencentdb",
}


def update_plugin_and_restart(client_type: str, **kwargs: Any) -> None:
    normalized_client_type = str(client_type).lower()
    if normalized_client_type == "openclaw":
        print("evaluating openclaw client, no need for plugin config update")
        return

    base = "openclaw config set plugins.entries"
    plugin_name = PLUGIN_NAME_BY_CLIENT[normalized_client_type]

    cmds = []
    for key, value in kwargs.items():
        if value is not None:
            config_key = key.replace("_", "-")
            if plugin_name == "memory-tencentdb":
                config_key = f"{config_key}.enabled"

            if isinstance(value, bool):
                cmd = f"{base}.{plugin_name}.config.{config_key} {str(value).lower()}"
            else:
                cmd = f"{base}.{plugin_name}.config.{config_key} '{value}'"
            cmds.append(cmd)

    env = openclaw_subprocess_env()
    for cmd in cmds:
        subprocess.run(cmd, shell=True, check=True, env=env)

    subprocess.run("openclaw gateway restart", shell=True, check=True, env=env)

    if cmds:
        print(f"Updated plugin config: {', '.join(cmds)}")

    time.sleep(10)


def configure_plugin_before_training(client_type: str) -> None:
    normalized_client_type = str(client_type).lower()
    if normalized_client_type == "tencentdb":
        update_plugin_and_restart(client_type, capture=True, extraction=True, recall=True)
    if normalized_client_type == "memos-cloud":
        update_plugin_and_restart(client_type, recallEnabled=True, addEnabled=True)
    if normalized_client_type in ["memorylake", "mem0", "supermemory", "memorylake", "openviking"]:
        update_plugin_and_restart(client_type, autoCapture=True, autoRecall=True)


def configure_plugin_before_testing(client_type: str) -> None:
    normalized_client_type = str(client_type).lower()
    if normalized_client_type == "tencentdb":
        update_plugin_and_restart(client_type, capture=False, extraction=False, recall=True)
    if normalized_client_type == "memos-cloud":
        update_plugin_and_restart(client_type, recallEnabled=True, addEnabled=False)
    if normalized_client_type in ["memorylake", "mem0", "supermemory", "memorylake", "openviking"]:
        update_plugin_and_restart(client_type, autoCapture=False, autoRecall=True)


def configure_plugin_for_agent(client_type: str, agent_id: str) -> None:
    normalized_client_type = str(client_type).lower()
    if normalized_client_type == "honcho":
        update_plugin_and_restart(client_type, workspaceId=agent_id)
    if normalized_client_type in ["memos-cloud", "mem0", "openviking"]:
        update_plugin_and_restart(client_type, userId=agent_id)
    if normalized_client_type == "supermemory":
        update_plugin_and_restart(client_type, containerTag=agent_id)
