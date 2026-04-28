import argparse
import asyncio
import json
import os
import subprocess
import sys
import time

from datetime import datetime

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
from scripts.utils.client import OpenclawClient  # noqa: E402


def update_plugin_and_restart(client_type, **kwargs):
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


def update_model_apikey_and_base_url(model_apikey=None, model_base_url=None):
    if model_apikey is None and model_base_url is None:
        print("No model apikey or base url provided, leaving all config as is")
        return
    if model_apikey is not None:
        cmd = f"openclaw config set models.providers.openai-codex.apiKey '{model_apikey}'"
        subprocess.run(cmd, shell=True, check=True)
    if model_base_url is not None:
        cmd = f"openclaw config set models.providers.openai-codex.baseUrl '{model_base_url}'"
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


async def process_qa_pair(client, qa, user_id, loop):
    question = qa.get("question")
    golden_answer = qa.get("answer")
    category = qa.get("category")

    if not question or not golden_answer:
        return None

    search_start = time.time()
    try:
        search_result = await loop.run_in_executor(
            None, lambda: client.search(question, user_id, top_k=5)
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

        if str(client_type).lower() == "honcho":
            update_plugin_and_restart(client_type, workspaceId=user_id)
        if str(client_type).lower() in ["memos-cloud", "mem0"]:
            update_plugin_and_restart(client_type, userId=user_id)
        if str(client_type).lower() == "supermemory":
            update_plugin_and_restart(client_type, containerTag=user_id)
        if str(client_type).lower() == "memos-cloud":
            update_plugin_and_restart(client_type, addEnabled=True, recallEnabled=False)
        elif str(client_type).lower() in ["mem0", "openviking", "memorylake", "supermemory"]:
            update_plugin_and_restart(client_type, autoCapture=True, autoRecall=False)

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

        if str(client_type).lower() == "honcho":
            update_plugin_and_restart(client_type, workspaceId=user_id)
        if str(client_type).lower() in ["memos-cloud", "mem0"]:
            update_plugin_and_restart(client_type, userId=user_id)
        if str(client_type).lower() == "supermemory":
            update_plugin_and_restart(client_type, containerTag=user_id)
        if str(client_type).lower() == "memos-cloud":
            update_plugin_and_restart(client_type, addEnabled=False, recallEnabled=True)
        elif str(client_type).lower() in ["mem0", "openviking", "memorylake", "supermemory"]:
            update_plugin_and_restart(client_type, autoCapture=False, autoRecall=True)

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--client_type",
        type=str,
        choices=[
            "openclaw",
            "memos-cloud",
            "memos-local",
            "openviking",
            "mem9",
            "mem0",
            "supermemory",
            "memorylake",
            "honcho",
            "byterover",
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
