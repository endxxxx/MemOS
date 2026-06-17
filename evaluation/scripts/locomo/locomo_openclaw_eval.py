import argparse
import asyncio
import json
import os
import subprocess
import sys
import time

from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm


load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.locomo.locomo_eval import (  # noqa: E402
    calculate_nlp_metrics,
    convert_numpy_types,
    locomo_grader,
)
from scripts.locomo.locomo_memos_local import (  # noqa: E402
    configure_memos_local_plugin_openclaw,
    is_memos_local_client,
    locomo_add_session_key,
    locomo_bpp_warmup_session_key,
    locomo_qa_session_key,
    prepare_user_db,
    save_user_db_checkpoint,
    settle_after_user_stage1,
    warmup_memos_bpp_hook,
)
from scripts.utils.client import OpenclawClient  # noqa: E402


OPENCLAW_AGENTS_ROOT = Path(os.path.expanduser(os.getenv("OPENCLAW_STATE_DIR", "~/.openclaw"))) / "agents"
MEMORY_ADD_PROMPT_MARKER = "You need to remember the following messages:"


def update_plugin_and_restart(client_type, **kwargs):
    plugin_mapping = {
        "mem0": "openclaw-mem0",
        "memos-cloud-cli": "memos-cloud-openclaw-plugin",
        "memos-cloud": "memos-cloud-openclaw-plugin",
        "memos-local": "memos-local-plugin",
        "memos-local-plugin": "memos-local-plugin",
        "mem9": "mem9",
        "openviking": "openviking",
        "supermemory": "openclaw-supermemory",
        "memorylake": "memorylake-openclaw",
        "honcho": "openclaw-honcho",
        "byterover": "byterover",
        "hindsight": "hindsight-openclaw",
        "tencentdb": "memory-tencentdb",
    }

    if str(client_type).lower() == "openclaw":
        print("evaluating openclaw client, no need for plugin config update")
        return
    if str(client_type).lower() not in plugin_mapping:
        print(f"Warning: unknown client type: {client_type}, leaving all config as is")
        return

    base = "openclaw config set plugins.entries"

    plugin_name = plugin_mapping.get(str(client_type).lower(), str(client_type).lower())

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

    for c in cmds:
        subprocess.run(c, shell=True, check=True)

    subprocess.run("openclaw gateway restart", shell=True, check=True)

    if cmds:
        print(f"Updated plugin config: {', '.join(cmds)}")

    time.sleep(10)


def update_cli_user_id(user_id):
    cmd = f"memos config set defaults.user_id '{user_id}'"
    subprocess.run(cmd, shell=True, check=True)


def update_model_apikey_and_base_url(model_apikey=None, model_base_url=None):
    if model_apikey is None and model_base_url is None:
        print("No model apikey or base url provided, leaving all config as is")
        return
    if model_apikey is not None:
        cmd = f"openclaw config set models.providers.memtensor.apiKey '{model_apikey}'"
        subprocess.run(cmd, shell=True, check=True)
    if model_base_url is not None:
        cmd = f"openclaw config set models.providers.memtensor.baseUrl '{model_base_url}'"
        subprocess.run(cmd, shell=True, check=True)
    subprocess.run("openclaw gateway restart", shell=True, check=True)


def parse_datetime(date_time_str):
    try:
        cleaned_str = date_time_str.replace("on", "").strip()
        dt = datetime.strptime(cleaned_str, "%I:%M %p %d %B, %Y")
        return dt.timestamp()
    except ValueError:
        return time.time()


def process_conversation(conversation):
    messages = []

    for key, value in conversation.items():
        if key.startswith("session_") and not key.endswith("_date_time"):
            date_time_key = f"{key}_date_time"
            timestamp = time.time()

            if date_time_key in conversation:
                timestamp = parse_datetime(conversation[date_time_key])

            for msg in value:
                if "text" in msg:
                    time_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
                    content_with_time = f"[{time_str}] {msg['text']}"
                    messages.append(
                        {
                            "role": msg["speaker"].lower(),
                            "content": content_with_time,
                            "timestamp": timestamp,
                        }
                    )

    return messages


async def process_qa_pair(client, qa, user_id, loop, session_key=None):
    question = qa.get("question")
    golden_answer = qa.get("answer")
    category = qa.get("category")

    if not question or not golden_answer:
        return None

    search_start = time.time()
    try:
        search_result = await loop.run_in_executor(
            None,
            lambda: client.search(
                question, user_id, top_k=5, session_key=session_key
            ),
        )
        search_duration = time.time() - search_start

        if isinstance(search_result, dict) and "choices" in search_result:
            response = search_result["choices"][0]["message"]["content"]
        else:
            response = str(search_result)

        return {
            "question": question,
            "answer": response,
            "golden_answer": golden_answer,
            "category": category,
            "search_duration_ms": search_duration * 1000,
        }

    except Exception as e:
        print(f"Error searching for question '{question}': {e}")
        return None


def _save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _load_json(path, default=None):
    if default is None:
        default = {}
    if not os.path.exists(path):
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Warning: could not load checkpoint {path}: {e}, starting fresh")
        return default


def _load_success_records(path):
    if not os.path.exists(path):
        return set()
    try:
        with open(path) as f:
            return {line.strip() for line in f if line.strip()}
    except (OSError, json.JSONDecodeError) as e:
        print(f"Warning: could not load success records {path}: {e}")
        return set()


def _append_success_record(path, record):
    with open(path, "a") as f:
        f.write(f"{record}\n")


def _save_stage_timing(path, stage_name, start_time, end_time):
    # transform start_time and end_time to datetime
    start_time_str = str(datetime.fromtimestamp(start_time))
    end_time_str = str(datetime.fromtimestamp(end_time))
    timing_data = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                timing_data = json.load(f)
        except (OSError, json.JSONDecodeError):
            timing_data = {}

    timing_data[stage_name] = {
        "start_time": start_time_str,
        "end_time": end_time_str,
        "duration_ms": (end_time - start_time) * 1000,
    }

    with open(path, "w") as f:
        json.dump(timing_data, f, indent=2)


def _numeric_usage_value(value):
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _assistant_total_tokens(usage):
    if not isinstance(usage, dict):
        return 0

    total_tokens = usage.get("totalTokens")
    if isinstance(total_tokens, int | float) and not isinstance(total_tokens, bool):
        return int(total_tokens)

    return sum(_numeric_usage_value(usage.get(field)) for field in ("input", "output", "cacheRead", "cacheWrite"))


def _usage_cost_total(usage):
    cost = usage.get("cost") if isinstance(usage, dict) else None
    if isinstance(cost, dict) and isinstance(cost.get("total"), int | float):
        return float(cost["total"])
    return 0.0


def _user_message_text(message):
    content = message.get("content", "")
    if isinstance(content, list):
        return "".join(
            item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"
        )
    return str(content)


def _empty_phase_usage():
    return {
        "sessions": 0,
        "assistant_messages": 0,
        "total_tokens": 0,
        "input": 0,
        "output": 0,
        "cache_read": 0,
        "cache_write": 0,
        "cost_usd": 0.0,
    }


def _merge_phase_usage(target, source):
    for key in (
        "sessions",
        "assistant_messages",
        "total_tokens",
        "input",
        "output",
        "cache_read",
        "cache_write",
    ):
        target[key] += source[key]
    target["cost_usd"] += source["cost_usd"]


def _parse_stage_local_dt(value):
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S.%f")


def _load_stage_windows(stage_timing_path):
    if not os.path.exists(stage_timing_path):
        return None

    try:
        with open(stage_timing_path) as f:
            timing_data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Warning: could not load stage timing for token stats: {e}")
        return None

    required = ("memory_addition", "qa_processing")
    if not all(stage in timing_data for stage in required):
        return None

    return {
        "memory_addition": (
            _parse_stage_local_dt(timing_data["memory_addition"]["start_time"]),
            _parse_stage_local_dt(timing_data["memory_addition"]["end_time"]),
        ),
        "qa_processing": (
            _parse_stage_local_dt(timing_data["qa_processing"]["start_time"]),
            _parse_stage_local_dt(timing_data["qa_processing"]["end_time"]),
        ),
    }


def _session_start_local(records):
    for record in records:
        timestamp = record.get("timestamp")
        if not timestamp:
            continue
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone().replace(tzinfo=None)
    return None


def _classify_session_phase(records, stage_windows):
    session_start = _session_start_local(records)
    if session_start is None:
        return None

    add_start, add_end = stage_windows["memory_addition"]
    qa_start, qa_end = stage_windows["qa_processing"]
    eval_start = add_start - timedelta(minutes=5)
    eval_end = qa_end + timedelta(minutes=5)

    if session_start < eval_start or session_start > eval_end:
        return None

    is_memory_addition = False
    for record in records:
        if record.get("type") != "message":
            continue
        message = record.get("message", {})
        if message.get("role") != "user":
            continue
        if MEMORY_ADD_PROMPT_MARKER in _user_message_text(message):
            is_memory_addition = True
            break

    if is_memory_addition:
        return "memory_addition"
    if session_start >= qa_start - timedelta(minutes=1):
        return "qa_retrieval"
    if add_start <= session_start <= add_end:
        return "memory_addition_other"
    return None


def _summarize_session_usage(records):
    usage_stats = _empty_phase_usage()
    for record in records:
        if record.get("type") != "message":
            continue
        message = record.get("message", {})
        if message.get("role") != "assistant":
            continue

        usage = message.get("usage")
        if not usage:
            continue

        usage_stats["assistant_messages"] += 1
        usage_stats["total_tokens"] += _assistant_total_tokens(usage)
        usage_stats["input"] += _numeric_usage_value(usage.get("input"))
        usage_stats["output"] += _numeric_usage_value(usage.get("output"))
        usage_stats["cache_read"] += _numeric_usage_value(usage.get("cacheRead"))
        usage_stats["cache_write"] += _numeric_usage_value(usage.get("cacheWrite"))
        usage_stats["cost_usd"] += _usage_cost_total(usage)

    if usage_stats["assistant_messages"] == 0:
        return None

    usage_stats["sessions"] = 1
    return usage_stats


def _finalize_phase_usage(phase_usage):
    sessions = phase_usage["sessions"]
    phase_usage["avg_tokens_per_session"] = (
        phase_usage["total_tokens"] / sessions if sessions else 0.0
    )
    phase_usage["cost_usd"] = round(phase_usage["cost_usd"], 6)
    return phase_usage


def collect_openclaw_token_usage(client_type, version, num_users, stage_timing_path):
    stage_windows = _load_stage_windows(stage_timing_path)
    if stage_windows is None:
        print("Skipping OpenClaw token stats: stage timing unavailable")
        return None

    report = {
        "client_type": client_type,
        "version": version,
        "memory_addition": _empty_phase_usage(),
        "qa_retrieval": _empty_phase_usage(),
        "excluded_sessions": 0,
        "memory_addition_other_sessions": 0,
        "per_user": {
            "memory_addition": {},
            "qa_retrieval": {},
        },
        "stage_timing_source": stage_timing_path,
    }

    for user_idx in range(num_users):
        agent_id = f"locomo_exp_user_{user_idx}_{client_type}_{version}"
        sessions_dir = OPENCLAW_AGENTS_ROOT / agent_id / "sessions"
        if not sessions_dir.is_dir():
            continue

        for jsonl_path in sorted(sessions_dir.glob("*.jsonl")):
            if jsonl_path.name.endswith(".trajectory.jsonl"):
                continue

            records = []
            try:
                with jsonl_path.open(encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        records.append(json.loads(line))
            except (OSError, json.JSONDecodeError) as e:
                print(f"Warning: could not read session log {jsonl_path}: {e}")
                continue

            phase = _classify_session_phase(records, stage_windows)
            if phase is None:
                report["excluded_sessions"] += 1
                continue
            if phase == "memory_addition_other":
                report["memory_addition_other_sessions"] += 1
                continue

            session_usage = _summarize_session_usage(records)
            if session_usage is None:
                report["excluded_sessions"] += 1
                continue

            _merge_phase_usage(report[phase], session_usage)
            report["per_user"][phase][agent_id] = (
                report["per_user"][phase].get(agent_id, 0) + session_usage["total_tokens"]
            )

    report["memory_addition"] = _finalize_phase_usage(report["memory_addition"])
    report["qa_retrieval"] = _finalize_phase_usage(report["qa_retrieval"])
    report["total"] = {
        "sessions": report["memory_addition"]["sessions"] + report["qa_retrieval"]["sessions"],
        "assistant_messages": (
            report["memory_addition"]["assistant_messages"] + report["qa_retrieval"]["assistant_messages"]
        ),
        "total_tokens": report["memory_addition"]["total_tokens"] + report["qa_retrieval"]["total_tokens"],
        "input": report["memory_addition"]["input"] + report["qa_retrieval"]["input"],
        "output": report["memory_addition"]["output"] + report["qa_retrieval"]["output"],
        "cache_read": report["memory_addition"]["cache_read"] + report["qa_retrieval"]["cache_read"],
        "cache_write": report["memory_addition"]["cache_write"] + report["qa_retrieval"]["cache_write"],
        "cost_usd": round(report["memory_addition"]["cost_usd"] + report["qa_retrieval"]["cost_usd"], 6),
    }
    return report


def _print_token_usage_summary(report):
    print("\n=== OpenClaw Token Usage ===")
    for phase, label in (
        ("memory_addition", "Memory Addition"),
        ("qa_retrieval", "QA Retrieval"),
    ):
        stats = report[phase]
        print(f"\n{label}:")
        print(f"  sessions: {stats['sessions']}")
        print(f"  assistant model calls: {stats['assistant_messages']}")
        print(f"  total_tokens: {stats['total_tokens']:,}")
        print(f"  input: {stats['input']:,}")
        print(f"  output: {stats['output']:,}")
        print(f"  cache_read: {stats['cache_read']:,}")
        print(f"  cache_write: {stats['cache_write']:,}")
        print(f"  cost_usd: ${stats['cost_usd']:.4f}")
        print(f"  avg_tokens_per_session: {stats['avg_tokens_per_session']:,.0f}")

    total = report["total"]
    print("\nTotal:")
    print(f"  sessions: {total['sessions']}")
    print(f"  assistant model calls: {total['assistant_messages']}")
    print(f"  total_tokens: {total['total_tokens']:,}")
    print(f"  cost_usd: ${total['cost_usd']:.4f}")

    if report["excluded_sessions"] or report["memory_addition_other_sessions"]:
        print(
            f"\nExcluded sessions: {report['excluded_sessions']}, "
            f"unclassified add-window sessions: {report['memory_addition_other_sessions']}"
        )


def save_openclaw_token_usage(report, results_dir):
    output_path = os.path.join(results_dir, "openclaw_token_usage.json")
    _save_json(output_path, report)
    print(f"OpenClaw token usage saved to: {output_path}")
    return output_path


async def evaluate_client(
    client_type, client, data, oai_client, num_runs=3, batch_size=10, resume=True, version="default"
):
    results_dir = f"results/locomo/{client_type}-{version}"
    os.makedirs(results_dir, exist_ok=True)

    response_path = f"{results_dir}/locomo_responses.json"
    judged_path = f"{results_dir}/locomo_judged.json"
    stage_timing_path = f"{results_dir}/stage_timing.json"
    add_records_path = f"{results_dir}/add_records.txt"
    qa_records_path = f"{results_dir}/qa_records.txt"

    if resume:
        responses = _load_json(response_path)
        all_grades = _load_json(judged_path)
    else:
        responses = {}
        all_grades = {}

    loop = asyncio.get_event_loop()

    if is_memos_local_client(client_type):
        configure_memos_local_plugin_openclaw()

    print("\n=== Stage 1: Memory Addition ===")
    update_model_apikey_and_base_url(
        model_apikey=os.getenv("MEMORY_ADDITION_API_KEY"),
        model_base_url=os.getenv("MEMORY_ADDITION_BASE_URL"),
    )
    add_start_time = time.time()

    if resume:
        completed_add_records = _load_success_records(add_records_path)
        print(f"Resume: found {len(completed_add_records)} completed add records")
    else:
        completed_add_records = set()
        if os.path.exists(add_records_path):
            os.remove(add_records_path)

    for user_idx, user_data in enumerate(data):
        user_id = f"locomo_exp_user_{user_idx}_{client_type}_{version}"
        client.set_agent_id(user_id)

        if is_memos_local_client(client_type):
            prepare_user_db(results_dir, user_idx)

        if str(client_type).lower() in ["memos-cloud-cli", "memos-cli"]:
            update_cli_user_id(user_id)
        if str(client_type).lower() == "honcho":
            update_plugin_and_restart(client_type, workspaceId=user_id)
        if str(client_type).lower() in ["memos-cloud", "mem0", "memos-cloud-cli"]:
            update_plugin_and_restart(client_type, userId=user_id)
        if str(client_type).lower() == "supermemory":
            update_plugin_and_restart(client_type, containerTag=user_id)
        if str(client_type).lower() == "hindsight":
            update_plugin_and_restart(client_type, autoRetain=True, autoRecall=False)
        if str(client_type).lower() in ["memos-cloud", "memos-cloud-cli"]:
            update_plugin_and_restart(client_type, addEnabled=True, recallEnabled=False)
        elif str(client_type).lower() in ["mem0", "openviking", "memorylake", "supermemory"]:
            update_plugin_and_restart(client_type, autoCapture=True, autoRecall=False)
        elif str(client_type).lower() == "tencentdb":
            update_plugin_and_restart(client_type, capture=True, extraction=True, recall=False)

        conversation = user_data.get("conversation", {})
        messages = process_conversation(conversation)

        print(f"Adding memory for user {user_id}...current time: {datetime.now()}")

        user_messages = messages
        total_messages = len(user_messages)

        if total_messages == 0:
            if resume and f"{user_idx}_0" not in completed_add_records:
                _append_success_record(add_records_path, f"{user_idx}_0")
            continue

        num_batches = (total_messages + batch_size - 1) // batch_size
        all_batches_completed = all(
            f"{user_idx}_{i}" in completed_add_records for i in range(num_batches)
        )

        if resume and all_batches_completed:
            print(f"Skipping memory addition for user {user_id} (all batches completed)")
            continue

        for batch_idx in tqdm(range(0, total_messages, batch_size)):
            batch_start_time = time.time()
            batch_messages = user_messages[batch_idx : batch_idx + batch_size]
            batch_num = batch_idx // batch_size

            record_key = f"{user_idx}_{batch_num}"

            if resume and record_key in completed_add_records:
                print(f"  Skipping batch {batch_num} for user {user_id} (already completed)")
                continue

            try:
                timestamp = batch_messages[0]["timestamp"] if batch_messages else time.time()
                batch_session_key = (
                    locomo_add_session_key(user_idx, version, batch_num)
                    if is_memos_local_client(client_type)
                    else None
                )
                client.add(
                    batch_messages,
                    user_id,
                    timestamp,
                    len(batch_messages),
                    session_key=batch_session_key,
                )
                batch_duration = time.time() - batch_start_time
                print(
                    f"  Added batch {batch_num} ({len(batch_messages)} messages) in {batch_duration:.2f}s"
                )
                if is_memos_local_client(client_type):
                    checkpoint = save_user_db_checkpoint(results_dir, user_idx)
                    print(f"  Saved memos-local DB checkpoint: {checkpoint}")
            except Exception as e:
                print(f"  Error adding batch {batch_num} for user {user_id}: {e}")
                continue

            _append_success_record(add_records_path, record_key)

        if is_memos_local_client(client_type):
            completed_after_user = _load_success_records(add_records_path)
            user_stage1_complete = all(
                f"{user_idx}_{i}" in completed_after_user for i in range(num_batches)
            )
            if user_stage1_complete:
                print(f"Settling memos pipeline after Stage 1 for user {user_id}...")
                settle = settle_after_user_stage1(user_idx)
                if not settle.get("ok"):
                    print(
                        f"  Warning: memos pipeline settle timed out after "
                        f"Stage 1 for user {user_idx}"
                    )
                checkpoint = save_user_db_checkpoint(results_dir, user_idx)
                print(f"  Final memos-local DB checkpoint for user {user_idx}: {checkpoint}")

    add_end_time = time.time()
    _save_stage_timing(stage_timing_path, "memory_addition", add_start_time, add_end_time)
    print(f"Memory addition stage completed in {(add_end_time - add_start_time):.2f}s")

    print("\n=== Stage 2: QA Processing ===")
    update_model_apikey_and_base_url(
        model_apikey=os.getenv("QA_PROCESSING_API_KEY"),
        model_base_url=os.getenv("QA_PROCESSING_BASE_URL"),
    )
    qa_start_time = time.time()

    if resume:
        completed_qa_records = _load_success_records(qa_records_path)
        print(f"Resume: found {len(completed_qa_records)} completed QA records")
    else:
        completed_qa_records = set()
        if os.path.exists(qa_records_path):
            os.remove(qa_records_path)

    for user_idx, user_data in enumerate(data):
        user_id = f"locomo_exp_user_{user_idx}_{client_type}_{version}"
        client.set_agent_id(user_id)

        if is_memos_local_client(client_type):
            prepare_user_db(results_dir, user_idx)
            warmup_memos_bpp_hook(
                client,
                user_id,
                locomo_bpp_warmup_session_key(user_idx, version),
                context=f"locomo QA user {user_idx}",
            )

        if str(client_type).lower() in ["memos-cloud-cli", "memos-cli"]:
            update_cli_user_id(user_id)
        if str(client_type).lower() == "honcho":
            update_plugin_and_restart(client_type, workspaceId=user_id)
        if str(client_type).lower() in ["memos-cloud", "mem0", "memos-cloud-cli", "openviking"]:
            update_plugin_and_restart(client_type, userId=user_id)
        if str(client_type).lower() == "supermemory":
            update_plugin_and_restart(client_type, containerTag=user_id)
        if str(client_type).lower() == "hindsight":
            update_plugin_and_restart(client_type, autoRetain=False, autoRecall=True)
        if str(client_type).lower() in ["memos-cloud", "memos-cloud-cli"]:
            update_plugin_and_restart(client_type, addEnabled=False, recallEnabled=True)
        elif str(client_type).lower() in ["mem0", "openviking", "memorylake", "supermemory"]:
            update_plugin_and_restart(client_type, autoCapture=False, autoRecall=True)
        elif str(client_type).lower() == "tencentdb":
            update_plugin_and_restart(client_type, capture=False, extraction=False, recall=True)

        qa_pairs = user_data.get("qa", [])
        print(f"Processing QA for user {user_id}...current time: {datetime.now()}")

        semaphore = asyncio.Semaphore(4)

        async def process_with_semaphore(
            qa,
            qa_idx,
            semaphore=semaphore,
            user_id=user_id,
            loop=loop,
        ):
            async with semaphore:
                qa_session_key = (
                    locomo_qa_session_key(user_idx, version, qa_idx)
                    if is_memos_local_client(client_type)
                    else None
                )
                return await process_qa_pair(
                    client, qa, user_id, loop, session_key=qa_session_key
                )

        for qa_idx, qa in tqdm(
            enumerate(qa_pairs), total=len(qa_pairs), desc=f"Processing QA for user {user_id}"
        ):
            qa_record_key = f"{user_idx}_{qa_idx}"

            if resume and qa_record_key in completed_qa_records:
                print(f"  Skipping QA {qa_idx} for user {user_id} (already completed)")
                continue

            result = await process_with_semaphore(qa, qa_idx)

            if result is not None:
                if user_id not in responses:
                    responses[user_id] = []
                responses[user_id].append(result)
                _save_json(response_path, responses)
                _append_success_record(qa_records_path, qa_record_key)

    qa_end_time = time.time()
    _save_stage_timing(stage_timing_path, "qa_processing", qa_start_time, qa_end_time)
    print(f"QA processing stage completed in {(qa_end_time - qa_start_time):.2f}s")

    print("\n=== 阶段 3: 评估响应 ===")

    if resume:
        graded_user_ids = set(all_grades.keys())
        if graded_user_ids:
            print(f"Resume: found {len(graded_user_ids)} graded users")
    else:
        graded_user_ids = set()

    for user_id, user_responses in responses.items():
        if user_id in graded_user_ids:
            print(f"Skipping grading for {user_id} (already graded)")
            continue

        graded_responses = []
        semaphore = asyncio.Semaphore(3)

        async def grade_with_semaphore(response, semaphore=semaphore):
            async with semaphore:
                question = response.get("question")
                answer = response.get("answer")
                ground_truth = response.get("golden_answer")
                category = response.get("category")
                search_duration_ms = response.get("search_duration_ms", 0.0)

                if not ground_truth:
                    return None

                grading_tasks = [
                    locomo_grader(oai_client, question, ground_truth, answer)
                    for _ in range(num_runs)
                ]
                judgments = await asyncio.gather(*grading_tasks)
                judgments_dict = {f"judgment_{i + 1}": j for i, j in enumerate(judgments)}

                nlp_metrics = calculate_nlp_metrics(
                    ground_truth, answer, "", ["lexical", "semantic"]
                )

                return {
                    "question": question,
                    "answer": answer,
                    "golden_answer": ground_truth,
                    "category": category,
                    "llm_judgments": judgments_dict,
                    "nlp_metrics": nlp_metrics,
                    "search_duration_ms": search_duration_ms,
                    "total_duration_ms": search_duration_ms,
                }

        grade_tasks = [grade_with_semaphore(response) for response in user_responses]

        for future in tqdm(
            asyncio.as_completed(grade_tasks),
            total=len(grade_tasks),
            desc=f"Grading responses for {user_id}",
            unit="response",
        ):
            graded = await future
            if graded is not None:
                graded_responses.append(graded)

        all_grades[user_id] = graded_responses

        to_save = convert_numpy_types(all_grades)
        _save_json(judged_path, to_save)

    all_grades = convert_numpy_types(all_grades)
    with open(judged_path, "w") as f:
        json.dump(all_grades, f, indent=2)

    token_usage = collect_openclaw_token_usage(client_type, version, len(data), stage_timing_path)
    if token_usage is not None:
        save_openclaw_token_usage(token_usage, results_dir)
        _print_token_usage_summary(token_usage)

    return all_grades


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--client_type",
        type=str,
        choices=[
            "openclaw",
            "memos-cloud-cli",
            "memos-cli",
            "memos-cloud",
            "memos-local",
            "memos-local-plugin",
            "openviking",
            "mem9",
            "mem0",
            "supermemory",
            "memorylake",
            "honcho",
            "byterover",
            "hindsight",
            "tencentdb",
        ],
        default="openclaw",
        help="The type of client to evaluate",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=3,
        help="Number of times to run the LLM grader for each question",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="default",
        help="Version of the evaluation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="The number of messages to add to memory at once",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Disable checkpoint resume; start evaluation from scratch",
    )
    args = parser.parse_args()

    load_dotenv()

    oai_client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL")
    )

    data_path = "/root/MemOS/evaluation/data/locomo/locomo10.json"
    with open(data_path) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} users from LoCoMo dataset")

    client = OpenclawClient(
        apikey=os.getenv("OPENCLAW_API_KEY"),
        baseurl=os.getenv("OPENCLAW_BASE_URL"),
    )

    await evaluate_client(
        args.client_type,
        client,
        data,
        oai_client,
        args.num_runs,
        args.batch_size,
        resume=not args.no_resume,
        version=args.version,
    )

    print("\n=== Evaluation Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
