import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

from datetime import datetime
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
from scripts.locomo.locomo_openclaw_eval import (  # noqa: E402
    _append_success_record,
    _load_json,
    _load_success_records,
    _save_json,
    _save_stage_timing,
    process_conversation,
    process_qa_pair,
)
from scripts.utils.client import HermesClient  # noqa: E402


CLIENT_TYPE = "hermes-agent"

_HERMES_PROFILES_ROOT = Path.home() / ".hermes" / "profiles"
_active_hermes_gateway_profile = None
_HERMES_CMD_TIMEOUT = 180
_HERMES_GATEWAY_READY_TIMEOUT = 120
_HERMES_DEFAULT_BASE_URL = os.getenv("HERMES_BASE_URL", "http://127.0.0.1:8642")


def _sanitize_hermes_profile_token(value):
    token = re.sub(r"[^a-z0-9_-]", "-", str(value).lower()).strip("-")
    return token or "default"


def hermes_profile_name(user_idx, version):
    """Return a valid Hermes profile id for one LoCoMo conversation group."""
    safe_version = _sanitize_hermes_profile_token(version)
    return f"locomo-u{user_idx}-{safe_version}"


def _hermes_env(profile_name):
    env = os.environ.copy()
    env["HERMES_HOME"] = str(_HERMES_PROFILES_ROOT / profile_name)
    return env


def _run_hermes_cmd(args, env=None, input_text=None, timeout=_HERMES_CMD_TIMEOUT, check=False):
    """Run a Hermes CLI command with a hard timeout to avoid indefinite hangs."""
    result = subprocess.run(
        args,
        env=env or os.environ.copy(),
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if check and result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        detail = stderr or stdout or f"exit code {result.returncode}"
        raise RuntimeError(f"Hermes command failed ({' '.join(args)}): {detail}")
    return result


def _list_eval_hermes_profile_names():
    if not _HERMES_PROFILES_ROOT.is_dir():
        return []
    return sorted(
        profile_dir.name
        for profile_dir in _HERMES_PROFILES_ROOT.iterdir()
        if profile_dir.is_dir() and profile_dir.name.startswith("locomo-u")
    )


def _ensure_hermes_port_free(port=8642):
    """Release the shared Hermes API port if a stale process still holds it."""
    result = subprocess.run(
        ["ss", "-tlnp"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if f":{port}" not in (result.stdout or ""):
        return
    subprocess.run(
        ["fuser", "-k", f"{port}/tcp"],
        capture_output=True,
        check=False,
        timeout=15,
    )
    time.sleep(2)


def _wait_for_hermes_gateway_ready(base_url=None, timeout=_HERMES_GATEWAY_READY_TIMEOUT):
    health_url = f"{(base_url or _HERMES_DEFAULT_BASE_URL).rstrip('/')}/health"
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=5) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
        time.sleep(2)
    raise RuntimeError(f"Hermes gateway not ready at {health_url} after {timeout}s: {last_error}")


def ensure_hermes_eval_profile(profile_name):
    profile_dir = _HERMES_PROFILES_ROOT / profile_name
    if profile_dir.is_dir():
        return profile_dir

    profile_dir.parent.mkdir(parents=True, exist_ok=True)
    _run_hermes_cmd(
        ["hermes", "profile", "create", profile_name, "--clone"],
        check=True,
    )
    if not profile_dir.is_dir():
        raise RuntimeError(f"Hermes profile directory was not created: {profile_dir}")
    return profile_dir


def _hermes_gateway_unit_path(profile_name):
    return Path.home() / ".config/systemd/user" / f"hermes-gateway-{profile_name}.service"


def _hermes_gateway_installed(profile_name):
    if _hermes_gateway_unit_path(profile_name).exists():
        return True

    env = _hermes_env(profile_name)
    result = _run_hermes_cmd(["hermes", "gateway", "status"], env=env, check=False)
    output = (result.stdout or "") + (result.stderr or "")
    lowered = output.lower()
    if "service is not installed" in lowered or "gateway service is not installed" in lowered:
        return False
    return f"hermes-gateway-{profile_name}.service" in output


def stop_hermes_gateway_for_profile(profile_name):
    _run_hermes_cmd(
        ["hermes", "gateway", "stop"],
        env=_hermes_env(profile_name),
        check=False,
    )


def stop_all_hermes_eval_gateways():
    """Stop the default gateway and every LoCoMo eval profile gateway."""
    global _active_hermes_gateway_profile

    _run_hermes_cmd(["hermes", "gateway", "stop"], check=False)
    for profile_name in _list_eval_hermes_profile_names():
        stop_hermes_gateway_for_profile(profile_name)
    _active_hermes_gateway_profile = None
    _ensure_hermes_port_free()


def _apply_hermes_model_config(env, model_apikey=None, model_base_url=None):
    if model_apikey is not None:
        _run_hermes_cmd(
            ["hermes", "config", "set", "model.api_key", model_apikey],
            env=env,
            check=True,
        )
    if model_base_url is not None:
        _run_hermes_cmd(
            ["hermes", "config", "set", "model.base_url", model_base_url],
            env=env,
            check=True,
        )


def start_hermes_gateway_for_profile(profile_name):
    _run_hermes_cmd(
        ["hermes", "gateway", "start"],
        env=_hermes_env(profile_name),
        check=True,
    )


def prepare_hermes_gateway_for_eval(profile_name, model_apikey=None, model_base_url=None):
    """Switch to one isolated profile gateway with a single stop/start cycle."""
    global _active_hermes_gateway_profile

    ensure_hermes_eval_profile(profile_name)
    stop_all_hermes_eval_gateways()

    env = _hermes_env(profile_name)
    if not _hermes_gateway_installed(profile_name):
        _run_hermes_cmd(
            ["hermes", "gateway", "install"],
            env=env,
            input_text="n\ny\n",
            check=True,
        )

    _apply_hermes_model_config(env, model_apikey, model_base_url)
    start_hermes_gateway_for_profile(profile_name)
    _wait_for_hermes_gateway_ready()
    _active_hermes_gateway_profile = profile_name


def save_hermes_profile_manifest(results_dir, mapping):
    manifest_path = Path(results_dir) / "hermes_profiles.json"
    with manifest_path.open("w") as f:
        json.dump(mapping, f, indent=2)


async def evaluate_hermes_client(
    client,
    data,
    oai_client,
    num_runs=3,
    batch_size=10,
    resume=True,
    version="default",
):
    client_type = CLIENT_TYPE
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
    hermes_profile_manifest = _load_json(f"{results_dir}/hermes_profiles.json", default={})

    print("\n=== Stage 1: Memory Addition ===")
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
        profile_name = hermes_profile_name(user_idx, version)

        prepare_hermes_gateway_for_eval(
            profile_name,
            model_apikey=os.getenv("MEMORY_ADDITION_API_KEY"),
            model_base_url=os.getenv("MEMORY_ADDITION_BASE_URL"),
        )
        hermes_profile_manifest[user_id] = {
            "user_idx": user_idx,
            "profile_name": profile_name,
            "hermes_home": str(_HERMES_PROFILES_ROOT / profile_name),
            "state_db": str(_HERMES_PROFILES_ROOT / profile_name / "state.db"),
            "memory_dir": str(_HERMES_PROFILES_ROOT / profile_name / "memories"),
        }
        save_hermes_profile_manifest(results_dir, hermes_profile_manifest)

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
                client.add(batch_messages, user_id, timestamp, len(batch_messages))
                batch_duration = time.time() - batch_start_time
                print(
                    f"  Added batch {batch_num} ({len(batch_messages)} messages) in {batch_duration:.2f}s"
                )
            except Exception as e:
                print(f"  Error adding batch {batch_num} for user {user_id}: {e}")
                continue

            _append_success_record(add_records_path, record_key)

    add_end_time = time.time()
    _save_stage_timing(stage_timing_path, "memory_addition", add_start_time, add_end_time)
    print(f"Memory addition stage completed in {(add_end_time - add_start_time):.2f}s")

    print("\n=== Stage 2: QA Processing ===")
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
        profile_name = hermes_profile_name(user_idx, version)

        prepare_hermes_gateway_for_eval(
            profile_name,
            model_apikey=os.getenv("QA_PROCESSING_API_KEY"),
            model_base_url=os.getenv("QA_PROCESSING_BASE_URL"),
        )

        qa_pairs = user_data.get("qa", [])
        print(f"Processing QA for user {user_id}...current time: {datetime.now()}")

        semaphore = asyncio.Semaphore(4)

        async def process_with_semaphore(qa, semaphore=semaphore, user_id=user_id, loop=loop):
            async with semaphore:
                return await process_qa_pair(client, qa, user_id, loop)

        for qa_idx, qa in enumerate(qa_pairs):
            qa_record_key = f"{user_idx}_{qa_idx}"

            if resume and qa_record_key in completed_qa_records:
                print(f"  Skipping QA {qa_idx} for user {user_id} (already completed)")
                continue

            result = await process_with_semaphore(qa)

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

    return all_grades


async def main():
    parser = argparse.ArgumentParser(description="LoCoMo evaluation for Hermes Agent")
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

    client = HermesClient(
        apikey=os.getenv("HERMES_API_KEY"),
        baseurl=os.getenv("HERMES_BASE_URL", "http://127.0.0.1:8642"),
    )

    await evaluate_hermes_client(
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
