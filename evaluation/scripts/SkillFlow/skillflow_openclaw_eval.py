from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import time

from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


EVAL_ROOT = Path(__file__).resolve().parents[2]
if str(EVAL_ROOT) not in sys.path:
    sys.path.append(str(EVAL_ROOT))

from scripts.utils.client import OpenclawClient  # noqa: E402


RANKING_FILE = "ALL_TASK_DIFFICULTY_RANKING.json"
SKILLFLOW_DATA_ROOT = Path("/root/SkillFlow_Data")
SKILLFLOW_OUTPUT_ROOT = Path("/root/SkillFlow_Outputs")
OPENCLAW_AGENTS_ROOT = Path("/root/.openclaw/agents")
TASK_FAMILY_PERMISSION_MODE = 0o755


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OpenClaw on a SkillFlow task family.")
    parser.add_argument(
        "--task_family_name",
        type=str,
        required=True,
        help="Name of the task family directory under /root/SkillFlow_Data.",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="Number of full evaluation iterations to run.",
    )
    parser.add_argument(
        "--num_train_set",
        type=int,
        default=4,
        help="Number of easiest ranked tasks to use as the training set.",
    )
    parser.add_argument(
        "--train_max_turns",
        type=int,
        default=3,
        help="Maximum user turns per training task, including the initial task prompt.",
    )
    parser.add_argument(
        "--test_max_turns",
        type=int,
        default=1,
        help="Maximum user turns per test task, including the initial task prompt.",
    )
    parser.add_argument(
        "--only_test",
        action="store_true",
        help="Skip training rounds and run only the test-set tasks.",
    )
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="Version label used in the output JSON filename.",
    )
    parser.add_argument(
        "--client_type",
        type=str,
        choices=[
            "openclaw",
            "memos-cloud",
            "memos-local",
            "honcho",
            "byterover",
            "supermemory",
            "memorylake",
            "mem9",
            "openviking",
            "mem0",
        ],
        default="openclaw",
        help="OpenClaw client/plugin type to evaluate.",
    )
    return parser.parse_args()


def resolve_task_family_path(task_family_name: str) -> Path:
    task_family_path = SKILLFLOW_DATA_ROOT / task_family_name
    if not task_family_path.is_dir():
        raise FileNotFoundError(
            f"Task family '{task_family_name}' not found under {SKILLFLOW_DATA_ROOT}"
        )
    return task_family_path


def chmod_task_family_path(task_family_path: Path) -> None:
    task_family_path.chmod(TASK_FAMILY_PERMISSION_MODE)
    for path in task_family_path.rglob("*"):
        path.chmod(TASK_FAMILY_PERMISSION_MODE)


def load_task_order(task_family_path: Path) -> list[Path]:
    ranking_path = task_family_path / RANKING_FILE
    if not task_family_path.is_dir():
        raise FileNotFoundError(f"Task family directory not found: {task_family_path}")
    if not ranking_path.is_file():
        raise FileNotFoundError(f"Difficulty ranking file not found: {ranking_path}")

    with ranking_path.open("r", encoding="utf-8") as f:
        task_names = json.load(f)

    if not isinstance(task_names, list) or not all(isinstance(name, str) for name in task_names):
        raise ValueError(f"{ranking_path} must contain a JSON array of task directory names")

    tasks: list[Path] = []
    for task_name in task_names:
        task_path = task_family_path / task_name
        instruction_path = task_path / "instruction.md"
        test_script = task_path / "tests" / "test.sh"

        if not task_path.is_dir():
            raise FileNotFoundError(f"Ranked task directory not found: {task_path}")
        if not instruction_path.is_file():
            raise FileNotFoundError(f"Task instruction file not found: {instruction_path}")
        if not test_script.is_file():
            raise FileNotFoundError(f"Task test script not found: {test_script}")

        tasks.append(task_path)

    if not tasks:
        raise ValueError(f"No tasks listed in {ranking_path}")

    return tasks


def require_openclaw_env() -> tuple[str, str]:
    api_key = os.getenv("OPENCLAW_API_KEY")
    base_url = os.getenv("OPENCLAW_BASE_URL")
    missing = [
        name
        for name, value in (
            ("OPENCLAW_API_KEY", api_key),
            ("OPENCLAW_BASE_URL", base_url),
        )
        if not value
    ]
    if missing:
        raise OSError(f"Missing required environment variables: {', '.join(missing)}")
    return api_key, base_url


def update_plugin_and_restart(client_type: str, **kwargs: Any) -> None:
    plugin_mapping = {
        "mem0": "openclaw-mem0",
        "memos-cloud": "memos-cloud-openclaw-plugin",
        "memos-local": "memos-local-openclaw-plugin",
        "mem9": "mem9",
        "openviking": "openviking",
        "supermemory": "openclaw-supermemory",
        "memorylake": "memorylake-openclaw",
        "honcho": "openclaw-honcho",
        "byterover": "byterover",
    }

    normalized_client_type = str(client_type).lower()
    if normalized_client_type == "openclaw":
        print("evaluating openclaw client, no need for plugin config update")
        return
    if normalized_client_type != "memos-cloud":
        print(f"evaluating {client_type} client, no per-run plugin config update needed")
        return

    base = "openclaw config set plugins.entries"
    plugin_name = plugin_mapping[normalized_client_type]

    cmds = []
    for key, value in kwargs.items():
        if value is not None:
            config_key = key.replace("_", "-")
            if isinstance(value, bool):
                cmd = f"{base}.{plugin_name}.config.{config_key} {str(value).lower()}"
            else:
                cmd = f"{base}.{plugin_name}.config.{config_key} '{value}'"
            cmds.append(cmd)

    for c in cmds:
        subprocess.run(c, shell=True, check=True)

    subprocess.run("openclaw gateway restart", shell=True, check=True)

    if cmds:
        print(f"Updated plugin config: {', '.join(cmds)}")

    time.sleep(10)


def new_agent_id(client_type: str, version: str, task_family_name: str, run_index: int) -> str:
    return f"sf_{client_type}_{version}_{task_family_name.lower()[:10]}_run_{run_index}"


def read_instruction(task_path: Path) -> str:
    return (task_path / "instruction.md").read_text(encoding="utf-8")


def task_output_dir(task_path: Path) -> Path:
    return SKILLFLOW_OUTPUT_ROOT / task_path.parent.name / task_path.name


def family_output_dir(task_family_path: Path) -> Path:
    return SKILLFLOW_OUTPUT_ROOT / task_family_path.name


def prepare_family_output_dir(task_family_path: Path) -> None:
    output_dir = family_output_dir(task_family_path)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def verifier_dirs(task_path: Path) -> list[Path]:
    return [Path("/logs/verifier"), task_path / "tests" / ".verifier"]


def reward_lookup_dirs(task_path: Path) -> list[Path]:
    primary = Path("/logs/verifier")
    fallback = task_path / "tests" / ".verifier"
    if primary.is_dir() and os.access(primary, os.W_OK):
        return [primary, fallback]
    return [fallback, primary]


def clear_verifier_artifacts(task_path: Path) -> None:
    for verifier_dir in verifier_dirs(task_path):
        for filename in ("reward.txt", "ctrf.json"):
            artifact = verifier_dir / filename
            if artifact.exists():
                with suppress(OSError):
                    artifact.unlink()


def read_reward(task_path: Path) -> tuple[float, Path | None]:
    for verifier_dir in reward_lookup_dirs(task_path):
        reward_path = verifier_dir / "reward.txt"
        if reward_path.is_file():
            raw_reward = reward_path.read_text(encoding="utf-8").strip()
            return float(raw_reward), reward_path
    raise FileNotFoundError(
        "Verifier reward.txt was not produced in /logs/verifier or the task-local .verifier"
    )


def string_path_from_node(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Path"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    ):
        value = node.args[0].value
        return value if isinstance(value, str) else None
    return None


def collect_task_outputs(task_path: Path) -> set[Path]:
    test_outputs = task_path / "tests" / "test_outputs.py"
    if not test_outputs.is_file():
        return set()

    tree = ast.parse(test_outputs.read_text(encoding="utf-8"), filename=str(test_outputs))
    outputs: set[Path] = set()
    output_names = {"PLAN_FILE", "SUMMARY_FILE"}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue

        target_names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if any(name.startswith("OUTPUT") or name in output_names for name in target_names):
            path_value = string_path_from_node(node.value)
            if path_value and path_value.startswith("/root/"):
                outputs.add(Path(path_value))

        if "CONFIG" in target_names and isinstance(node.value, ast.Call) and node.value.args:
            config_raw = string_path_from_node(node.value.args[0])
            if not config_raw:
                continue
            try:
                config = json.loads(config_raw)
            except json.JSONDecodeError:
                continue
            for key, value in config.items():
                if (
                    key.startswith("output_")
                    and isinstance(value, str)
                    and value.startswith("/root/")
                ):
                    outputs.add(Path(value))

    return outputs


def remove_task_outputs(task_path: Path) -> None:
    """Best-effort cleanup of prior deliverables that the verifier checks."""
    for output_path in collect_task_outputs(task_path):
        if output_path.exists() and (output_path.is_file() or output_path.is_symlink()):
            with suppress(OSError):
                output_path.unlink()


def list_task_output_files(task_path: Path) -> list[str]:
    output_dir = task_output_dir(task_path)
    if not output_dir.is_dir():
        return []
    return sorted(
        str(path) for path in output_dir.rglob("*") if path.is_file() or path.is_symlink()
    )


def numeric_usage_value(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def assistant_tool_call_count(message: dict[str, Any]) -> int:
    content = message.get("content")
    if not isinstance(content, list):
        return 0
    return sum(1 for item in content if isinstance(item, dict) and item.get("type") == "toolCall")


def assistant_total_tokens(message: dict[str, Any]) -> int | None:
    usage = message.get("usage")
    if not isinstance(usage, dict):
        return None

    total_tokens = usage.get("totalTokens")
    if isinstance(total_tokens, int | float) and not isinstance(total_tokens, bool):
        return int(total_tokens)

    fallback_total = sum(
        numeric_usage_value(usage.get(field))
        for field in ("input", "output", "cacheRead", "cacheWrite")
    )
    return fallback_total if fallback_total else None


def summarize_agent_sessions(agent_id: str) -> dict[str, Any]:
    agent_dir = OPENCLAW_AGENTS_ROOT / agent_id
    sessions_dir = agent_dir / "sessions"
    stats = {
        "agent_dir": str(agent_dir),
        "sessions_dir": str(sessions_dir),
        "sessions_dir_exists": sessions_dir.is_dir(),
        "session_files": 0,
        "jsonl_lines": 0,
        "assistant_messages": 0,
        "assistant_messages_with_usage": 0,
        "assistant_messages_without_usage": 0,
        "tool_call_count": 0,
        "total_tokens": 0,
        "invalid_json_lines": 0,
    }

    if not sessions_dir.is_dir():
        return stats

    for jsonl_file in sorted(sessions_dir.glob("*.jsonl")):
        stats["session_files"] += 1
        with jsonl_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                stats["jsonl_lines"] += 1
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    stats["invalid_json_lines"] += 1
                    continue

                if not isinstance(record, dict):
                    continue
                message = record.get("message")
                if not isinstance(message, dict) or message.get("role") != "assistant":
                    continue

                stats["assistant_messages"] += 1
                stats["tool_call_count"] += assistant_tool_call_count(message)
                tokens = assistant_total_tokens(message)
                if tokens is None:
                    stats["assistant_messages_without_usage"] += 1
                    continue
                stats["assistant_messages_with_usage"] += 1
                stats["total_tokens"] += tokens

    return stats


SESSION_STATS_COUNTERS = (
    "session_files",
    "jsonl_lines",
    "assistant_messages",
    "assistant_messages_with_usage",
    "assistant_messages_without_usage",
    "tool_call_count",
    "total_tokens",
    "invalid_json_lines",
)


def session_stats_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    return {
        key: numeric_usage_value(after.get(key)) - numeric_usage_value(before.get(key))
        for key in SESSION_STATS_COUNTERS
    }


def usage_metrics_from_delta(delta: dict[str, Any]) -> dict[str, int]:
    return {
        "tool_call_count": numeric_usage_value(delta.get("tool_call_count")),
        "total_tokens": numeric_usage_value(delta.get("total_tokens")),
    }


def empty_usage_metrics() -> dict[str, int]:
    return {"tool_call_count": 0, "total_tokens": 0}


def add_usage_metrics(target: dict[str, int], delta: dict[str, Any]) -> None:
    target["tool_call_count"] += numeric_usage_value(delta.get("tool_call_count"))
    target["total_tokens"] += numeric_usage_value(delta.get("total_tokens"))


def safe_session_fragment(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


def build_initial_task_prompt(task_path: Path) -> str:
    return f"阅读{task_path.resolve()}下的instruction.md并完成其中的任务"


def build_retry_prompt(verifier_stdout: str) -> str:
    stdout = verifier_stdout.strip() or "验证器 stdout 为空。"
    return f"任务执行结果验证不通过，请参考下面的信息继续修改\n\n{stdout}"


def run_task_test(task_path: Path) -> dict[str, Any]:
    clear_verifier_artifacts(task_path)
    test_script = task_path / "tests" / "test.sh"

    started_at = time.time()
    completed = subprocess.run(
        ["bash", str(test_script)],
        cwd=str(task_path),
        capture_output=True,
        text=True,
        check=False,
    )
    duration_ms = (time.time() - started_at) * 1000

    reward_path: Path | None = None
    error: str | None = None
    try:
        reward, reward_path = read_reward(task_path)
    except Exception as exc:
        reward = 0.0
        error = str(exc)

    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "duration_ms": duration_ms,
        "reward": reward,
        "success": reward >= 1.0,
        "reward_path": str(reward_path) if reward_path else None,
        "error": error,
    }


def run_task(
    client: OpenclawClient,
    task_path: Path,
    agent_id: str,
    session_key: str,
    max_turns: int,
    split: str,
    train_round: int | None = None,
) -> dict[str, Any]:
    task_output_dir(task_path).mkdir(parents=True, exist_ok=True)
    remove_task_outputs(task_path)
    openclaw_query = build_initial_task_prompt(task_path)
    current_query = openclaw_query
    attempts: list[dict[str, Any]] = []
    aggregate_usage = empty_usage_metrics()

    for turn_index in range(1, max_turns + 1):
        before_stats = summarize_agent_sessions(agent_id)
        started_at = time.time()
        search_response = client.search(
            query=current_query,
            user_id=agent_id,
            top_k=0,
            session_key=session_key,
        )
        openclaw_duration_ms = (time.time() - started_at) * 1000
        after_stats = summarize_agent_sessions(agent_id)
        stats_delta = session_stats_delta(before_stats, after_stats)
        add_usage_metrics(aggregate_usage, stats_delta)

        test_result = run_task_test(task_path)
        attempts.append(
            {
                "turn_index": turn_index,
                "prompt": current_query,
                "openclaw_duration_ms": openclaw_duration_ms,
                "openclaw_response": search_response,
                "session_stats_delta": stats_delta,
                "usage": usage_metrics_from_delta(stats_delta),
                "test": test_result,
                "success": bool(test_result["success"]),
            }
        )

        if test_result["success"] or turn_index >= max_turns:
            break
        current_query = build_retry_prompt(str(test_result.get("stdout", "")))

    final_test_result = attempts[-1]["test"] if attempts else None
    success = bool(final_test_result and final_test_result["success"])

    return {
        "task_name": task_path.name,
        "task_path": str(task_path),
        "instruction_path": str(task_path / "instruction.md"),
        "split": split,
        "train_round": train_round,
        "session_key": session_key,
        "max_turns": max_turns,
        "turns_used": len(attempts),
        "user_agent_interactions": len(attempts),
        "openclaw_query": openclaw_query,
        "usage": aggregate_usage,
        "attempts": attempts,
        "test": final_test_result,
        "success": success,
    }


def failed_task_result(
    task_path: Path,
    split: str,
    session_key: str,
    max_turns: int,
    error: str,
    train_round: int | None = None,
) -> dict[str, Any]:
    return {
        "task_name": task_path.name,
        "task_path": str(task_path),
        "instruction_path": str(task_path / "instruction.md"),
        "split": split,
        "train_round": train_round,
        "session_key": session_key,
        "max_turns": max_turns,
        "turns_used": 0,
        "user_agent_interactions": 0,
        "usage": empty_usage_metrics(),
        "attempts": [],
        "test": None,
        "success": False,
        "error": error,
    }


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_test_task_with_output_check(
    client: OpenclawClient,
    task_path: Path,
    agent_id: str,
    base_session_key: str,
    max_turns: int,
) -> dict[str, Any]:
    task_result = run_task(
        client=client,
        task_path=task_path,
        agent_id=agent_id,
        session_key=base_session_key,
        max_turns=max_turns,
        split="test",
    )
    output_files = list_task_output_files(task_path)
    task_result["output_check"] = {
        "output_dir": str(task_output_dir(task_path)),
        "files": output_files,
        "file_count": len(output_files),
        "rerun_due_to_empty_output": False,
    }
    initial_execution = dict(task_result)
    task_result["executions"] = [initial_execution]

    if output_files:
        return task_result

    rerun_session_key = f"{base_session_key}_rerun_1"
    try:
        rerun_result = run_task(
            client=client,
            task_path=task_path,
            agent_id=agent_id,
            session_key=rerun_session_key,
            max_turns=max_turns,
            split="test",
        )
    except Exception as exc:
        rerun_result = failed_task_result(
            task_path=task_path,
            split="test",
            session_key=rerun_session_key,
            max_turns=max_turns,
            error=str(exc),
        )
    rerun_output_files = list_task_output_files(task_path)
    combined_usage = empty_usage_metrics()
    add_usage_metrics(combined_usage, task_result["usage"])
    add_usage_metrics(combined_usage, rerun_result["usage"])

    rerun_result["output_check"] = {
        "output_dir": str(task_output_dir(task_path)),
        "files": rerun_output_files,
        "file_count": len(rerun_output_files),
        "rerun_due_to_empty_output": True,
    }
    rerun_execution = dict(rerun_result)
    rerun_result["executions"] = [initial_execution, rerun_execution]
    rerun_result["usage"] = combined_usage
    rerun_result["turns_used"] = numeric_usage_value(
        task_result.get("turns_used")
    ) + numeric_usage_value(rerun_result.get("turns_used"))
    rerun_result["user_agent_interactions"] = numeric_usage_value(
        task_result.get("user_agent_interactions")
    ) + numeric_usage_value(rerun_result.get("user_agent_interactions"))
    rerun_result["output_check"]["initial_files"] = output_files
    rerun_result["output_check"]["final_files"] = rerun_output_files
    rerun_result["output_check"]["initial_session_key"] = base_session_key
    rerun_result["output_check"]["rerun_session_key"] = rerun_session_key
    return rerun_result


def run_evaluation(
    task_family_name: str,
    num_runs: int,
    version: str,
    client_type: str,
    num_train_set: int,
    train_max_turns: int,
    test_max_turns: int,
    only_test: bool,
) -> dict[str, Any]:
    if num_runs <= 0:
        raise ValueError("--num_runs must be a positive integer")
    if train_max_turns <= 0:
        raise ValueError("--train_max_turns must be a positive integer")
    if test_max_turns <= 0:
        raise ValueError("--test_max_turns must be a positive integer")

    load_dotenv()
    api_key, base_url = require_openclaw_env()
    task_family_path = resolve_task_family_path(task_family_name)
    tasks = load_task_order(task_family_path)
    if num_train_set < 0 or num_train_set >= len(tasks):
        raise ValueError(
            "--num_train_set must be greater than or equal to 0 and less than the number of tasks"
        )

    train_tasks = tasks[:num_train_set]
    test_tasks = tasks[num_train_set:]
    runs = []

    for run_index in range(1, num_runs + 1):
        chmod_task_family_path(task_family_path)
        prepare_family_output_dir(task_family_path)
        agent_id = new_agent_id(client_type, version, task_family_name, run_index)
        if str(client_type).lower() == "honcho":
            update_plugin_and_restart(client_type, workspaceId=agent_id)
        if str(client_type).lower() in ["memos-cloud", "mem0"]:
            update_plugin_and_restart(client_type, userId=agent_id)
        if str(client_type).lower() == "supermemory":
            update_plugin_and_restart(client_type, containerTag=agent_id)
        client = OpenclawClient(apikey=api_key, baseurl=base_url, agent_id=agent_id)

        print(f"\n=== SkillFlow run {run_index}/{num_runs} | agent_id={agent_id} ===")
        train_session_key = safe_session_fragment(f"{agent_id}_train")
        train_rounds: dict[str, list[dict[str, Any]]] = {"1": [], "2": []}
        train_round_metrics = {
            "1": empty_usage_metrics(),
            "2": empty_usage_metrics(),
        }
        test_results = []
        test_successes = 0
        test_usage = empty_usage_metrics()
        test_interactions = 0

        if not only_test:
            for train_round in (1, 2):
                for task_index, task_path in enumerate(train_tasks, start=1):
                    print(
                        f"[run {run_index}] train round {train_round}/2 "
                        f"task {task_index}/{len(train_tasks)}: {task_path.name}"
                    )
                    try:
                        task_result = run_task(
                            client=client,
                            task_path=task_path,
                            agent_id=agent_id,
                            session_key=train_session_key,
                            max_turns=train_max_turns,
                            split="train",
                            train_round=train_round,
                        )
                    except Exception as exc:
                        task_result = failed_task_result(
                            task_path=task_path,
                            split="train",
                            session_key=train_session_key,
                            max_turns=train_max_turns,
                            error=str(exc),
                            train_round=train_round,
                        )
                    add_usage_metrics(train_round_metrics[str(train_round)], task_result["usage"])
                    train_rounds[str(train_round)].append(task_result)

        for task_index, task_path in enumerate(test_tasks, start=1):
            session_key = safe_session_fragment(f"{agent_id}_test_{task_index}")
            print(f"[run {run_index}] test task {task_index}/{len(test_tasks)}: {task_path.name}")
            try:
                task_result = run_test_task_with_output_check(
                    client=client,
                    task_path=task_path,
                    agent_id=agent_id,
                    base_session_key=session_key,
                    max_turns=test_max_turns,
                )
            except Exception as exc:
                task_result = failed_task_result(
                    task_path=task_path,
                    split="test",
                    session_key=session_key,
                    max_turns=test_max_turns,
                    error=str(exc),
                )

            if task_result["success"]:
                test_successes += 1
            add_usage_metrics(test_usage, task_result["usage"])
            test_interactions += numeric_usage_value(task_result.get("user_agent_interactions"))
            test_results.append(task_result)

        test_attempts = len(test_tasks)
        test_completion_rate = test_successes / test_attempts if test_attempts else 0.0
        run_metrics = {
            "test_completion_rate": test_completion_rate,
            "train_round_1": train_round_metrics["1"],
            "train_round_2": train_round_metrics["2"],
            "test": {
                "successes": test_successes,
                "attempts": test_attempts,
                "completion_rate": test_completion_rate,
                "total_tool_call_count": test_usage["tool_call_count"],
                "total_tokens": test_usage["total_tokens"],
                "total_user_agent_interactions": test_interactions,
                "avg_tool_call_count_per_task": (
                    test_usage["tool_call_count"] / test_attempts if test_attempts else 0.0
                ),
                "avg_tokens_per_task": (
                    test_usage["total_tokens"] / test_attempts if test_attempts else 0.0
                ),
                "avg_interactions_per_task": (
                    test_interactions / test_attempts if test_attempts else 0.0
                ),
            },
        }
        agent_session_stats = summarize_agent_sessions(agent_id)
        runs.append(
            {
                "run_index": run_index,
                "agent_id": agent_id,
                "train_session_key": train_session_key,
                "metrics": run_metrics,
                "agent_session_stats": agent_session_stats,
                "train_rounds": train_rounds,
                "test_tasks": test_results,
            }
        )
        print(
            f"[run {run_index}] test_completion_rate={test_completion_rate:.4f} "
            f"train_r1_tool_calls={train_round_metrics['1']['tool_call_count']} "
            f"train_r2_tool_calls={train_round_metrics['2']['tool_call_count']} "
            f"test_avg_tool_calls={run_metrics['test']['avg_tool_call_count_per_task']:.2f}"
        )

    total_test_tasks = len(test_tasks) * len(runs)
    aggregate_test_tool_calls = sum(
        numeric_usage_value(run["metrics"]["test"]["total_tool_call_count"]) for run in runs
    )
    aggregate_test_tokens = sum(
        numeric_usage_value(run["metrics"]["test"]["total_tokens"]) for run in runs
    )
    aggregate_test_interactions = sum(
        numeric_usage_value(run["metrics"]["test"]["total_user_agent_interactions"]) for run in runs
    )
    metrics = {
        "average_test_completion_rate": average(
            [float(run["metrics"]["test_completion_rate"]) for run in runs]
        ),
        "train_round_1": {
            "average_tool_call_count_per_run": average(
                [float(run["metrics"]["train_round_1"]["tool_call_count"]) for run in runs]
            ),
            "average_tokens_per_run": average(
                [float(run["metrics"]["train_round_1"]["total_tokens"]) for run in runs]
            ),
        },
        "train_round_2": {
            "average_tool_call_count_per_run": average(
                [float(run["metrics"]["train_round_2"]["tool_call_count"]) for run in runs]
            ),
            "average_tokens_per_run": average(
                [float(run["metrics"]["train_round_2"]["total_tokens"]) for run in runs]
            ),
        },
        "test": {
            "avg_tool_call_count_per_task": (
                aggregate_test_tool_calls / total_test_tasks if total_test_tasks else 0.0
            ),
            "avg_tokens_per_task": (
                aggregate_test_tokens / total_test_tasks if total_test_tasks else 0.0
            ),
            "avg_interactions_per_task": (
                aggregate_test_interactions / total_test_tasks if total_test_tasks else 0.0
            ),
        },
    }

    return {
        "metadata": {
            "benchmark": "SkillFlow",
            "client": client_type,
            "version": version,
            "task_family_path": str(task_family_path),
            "task_family": task_family_path.name,
            "output_root": str(SKILLFLOW_OUTPUT_ROOT),
            "task_family_output_dir": str(family_output_dir(task_family_path)),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "task_order": [task.name for task in tasks],
        "train_task_order": [task.name for task in train_tasks],
        "test_task_order": [task.name for task in test_tasks],
        "num_runs": num_runs,
        "num_train_set": num_train_set,
        "train_max_turns": train_max_turns,
        "test_max_turns": test_max_turns,
        "only_test": only_test,
        "num_tasks": len(tasks),
        "num_train_tasks": len(train_tasks),
        "num_test_tasks": len(test_tasks),
        "metrics": metrics,
        "runs": runs,
    }


def save_results(
    results: dict[str, Any], version: str, client_type: str, task_family_name: str
) -> Path:
    output_dir = Path(__file__).resolve().parent.parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"skillflow-{client_type}-{version}-{task_family_name.lower()}.json"

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    shutil.move(str(tmp_path), str(output_path))

    return output_path


def main() -> None:
    args = parse_args()

    results = run_evaluation(
        task_family_name=args.task_family_name,
        num_runs=args.num_runs,
        version=args.version,
        client_type=args.client_type,
        num_train_set=args.num_train_set,
        train_max_turns=args.train_max_turns,
        test_max_turns=args.test_max_turns,
        only_test=args.only_test,
    )
    output_path = save_results(results, args.version, args.client_type, args.task_family_name)

    print("\n=== SkillFlow OpenClaw Evaluation Complete ===")
    print(f"Average test completion rate: {results['metrics']['average_test_completion_rate']:.4f}")
    print(
        "Train round 1 average tool calls/tokens per run: "
        f"{results['metrics']['train_round_1']['average_tool_call_count_per_run']:.2f}/"
        f"{results['metrics']['train_round_1']['average_tokens_per_run']:.2f}"
    )
    print(
        "Train round 2 average tool calls/tokens per run: "
        f"{results['metrics']['train_round_2']['average_tool_call_count_per_run']:.2f}/"
        f"{results['metrics']['train_round_2']['average_tokens_per_run']:.2f}"
    )
    print(
        "Test averages per task tool calls/tokens/interactions: "
        f"{results['metrics']['test']['avg_tool_call_count_per_task']:.2f}/"
        f"{results['metrics']['test']['avg_tokens_per_task']:.2f}/"
        f"{results['metrics']['test']['avg_interactions_per_task']:.2f}"
    )
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
