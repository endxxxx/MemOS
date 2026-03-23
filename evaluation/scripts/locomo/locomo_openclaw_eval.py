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

from evaluation.scripts.locomo.locomo_eval import (
    calculate_nlp_metrics,
    convert_numpy_types,
    locomo_grader,
)
from evaluation.scripts.utils.client import OpenclawClient


# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

load_dotenv()


def update_plugin_and_restart(client_type, set_user_id=None, recall_enabled=None, add_enabled=None):
    """
    Update plugin configuration and restart the gateway so settings take effect.
    """
    if str(client_type).lower() not in ["memos", "openviking"]:
        raise Exception(f"unknown client type: {client_type}")
    base = "openclaw config set plugins.entries"
    cmds = []
    if set_user_id is not None:
        assert (
            str(client_type).lower() == "memos"
        )  # only memos-cloud-openclaw-plugin has this configuration
        cmds.append(f"{base}.memos-cloud-openclaw-plugin.config.userId '{set_user_id}'")
    if recall_enabled is not None:
        if client_type == "memos":
            cmds.append(
                f"{base}.memos-cloud-openclaw-plugin.config.recallEnabled {str(recall_enabled).lower()}"
            )
        else:
            cmds.append(f"{base}.openviking.config.autoRecall {str(recall_enabled).lower()}")
    if add_enabled is not None:
        if client_type == "memos":
            cmds.append(
                f"{base}.memos-cloud-openclaw-plugin.config.addEnabled {str(add_enabled).lower()}"
            )
        else:
            cmds.append(f"{base}.openviking.config.autoCapture {str(add_enabled).lower()}")
    # after updating config, restart the gateway
    cmds.append("openclaw gateway restart")
    for c in cmds:
        subprocess.run(c, shell=True, check=True)
    if cmds:
        print(f"Updated plugin config: {', '.join(cmds)}")
    time.sleep(10)


def parse_datetime(date_time_str):
    """
    Parse datetime string from the conversation data
    Example: "1:56 pm on 8 May, 2023"
    """
    try:
        # Remove "on" and strip whitespace
        cleaned_str = date_time_str.replace("on", "").strip()
        # Parse the datetime
        dt = datetime.strptime(cleaned_str, "%I:%M %p %d %B, %Y")
        return dt.timestamp()
    except ValueError:
        # If parsing fails, return current timestamp as fallback
        return time.time()


def process_conversation(conversation):
    """
    Process conversation data into messages format for client.add()
    """
    messages = []

    # Process each session
    for key, value in conversation.items():
        if key.startswith("session_") and not key.endswith("_date_time"):
            # Get the corresponding date time
            date_time_key = f"{key}_date_time"
            timestamp = time.time()

            if date_time_key in conversation:
                timestamp = parse_datetime(conversation[date_time_key])

            # Process each message in the session
            for msg in value:
                if "text" in msg:
                    # Format timestamp to include in content
                    time_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
                    # Add timestamp to content
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
    """
    Process a single QA pair
    """
    question = qa.get("question")
    golden_answer = qa.get("answer")
    category = qa.get("category")

    if not question or not golden_answer:
        return None

    # Run synchronous search in executor
    search_start = time.time()
    try:
        search_result = await loop.run_in_executor(
            None, lambda: client.search(question, user_id, top_k=5)
        )
        search_duration = time.time() - search_start

        # Extract response from search result
        if isinstance(search_result, dict) and "choices" in search_result:
            response = search_result["choices"][0]["message"]["content"]
        else:
            response = str(search_result)

        # Store response
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
    """Save data to JSON file (for checkpoint)."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _load_json(path, default=None):
    """Load JSON from file if exists, else return default."""
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


async def evaluate_client(
    client_type, client, data, oai_client, num_runs=3, batch_size=10, resume=True, version="default"
):
    """
    Evaluate a client on the LoCoMo dataset.
    When resume=True (default), loads existing response/judged checkpoints and skips completed users.
    """

    results_dir = f"results/locomo/{client_type}-{version}"
    os.makedirs(results_dir, exist_ok=True)

    response_path = f"{results_dir}/locomo_responses.json"
    judged_path = f"{results_dir}/locomo_judged.json"

    # Resume: load existing responses, only process users not yet completed
    responses = _load_json(response_path) if resume else {}
    completed_user_ids = set(responses.keys())
    if completed_user_ids:
        print(
            f"Resume: found checkpoint with {len(completed_user_ids)} users, skipping them in response phase"
        )

    # Process each user's data
    for user_idx, user_data in enumerate(data):
        user_id = f"locomo_exp_user_{user_idx}"
        if user_id in completed_user_ids:
            print(f"Skipping user {user_id} (already in checkpoint)")
            continue

        # Sync memos_openclaw_plugin userId and restart gateway
        if client_type == "memos":
            update_plugin_and_restart(client_type, set_user_id=user_id)

        user_responses = []

        print(f"Processing user {user_id}...")

        # Step 1: Add conversation to memory (enable add, disable recall)
        update_plugin_and_restart(client_type, add_enabled=True, recall_enabled=False)
        conversation = user_data.get("conversation", {})
        messages = process_conversation(conversation)

        time.sleep(60)

        print(
            f"Adding {len(messages)} messages to memory for {user_id}...current time: {datetime.now()}"
        )
        start_time = time.time()

        try:
            # Extract timestamp from messages (use the first one if available)
            timestamp = messages[0]["timestamp"] if messages else time.time()

            # Add messages to memory
            client.add(messages, user_id, timestamp, batch_size)
            add_duration = time.time() - start_time
            print(f"Added messages in {add_duration:.2f} seconds")
        except Exception as e:
            print(f"Error adding messages: {e}")
            continue

        time.sleep(60)  # Add delay to avoid rate limiting

        # Step 2: Process each QA pair in parallel (disable add, enable recall)
        update_plugin_and_restart(client_type, add_enabled=False, recall_enabled=True)
        qa_pairs = user_data.get("qa", [])
        print(f"Processing {len(qa_pairs)} QA pairs for {user_id}...current time: {datetime.now()}")

        # Get event loop
        loop = asyncio.get_event_loop()

        # Create semaphore to limit concurrent workers to 4
        semaphore = asyncio.Semaphore(4)

        async def process_with_semaphore(qa, semaphore=semaphore, user_id=user_id, loop=loop):
            async with semaphore:
                return await process_qa_pair(client, qa, user_id, loop)

        # Process all QA pairs in parallel with progress bar
        tasks = [process_with_semaphore(qa) for qa in qa_pairs]

        # Use tqdm to show progress
        completed_responses = []
        for future in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc=f"Processing QA pairs for {user_id}",
            unit="qa",
        ):
            result = await future
            if result is not None:
                completed_responses.append(result)

        user_responses = completed_responses
        responses[user_id] = user_responses

        # Checkpoint: save after each user's QA completes
        _save_json(response_path, responses)
        print(f"Checkpoint: saved responses (user {user_id}) to {response_path}")

    # If checkpoint was not used or all runs finished, write again here for consistency
    _save_json(response_path, responses)
    print(f"Saved responses to {response_path}")

    # Step 3: Evaluate responses
    print("\n=== Evaluating responses ===")

    # Resume: load existing grading results, only grade users not yet scored
    all_grades = _load_json(judged_path) if resume else {}
    graded_user_ids = set(all_grades.keys())
    if graded_user_ids:
        print(
            f"Resume: found judged checkpoint with {len(graded_user_ids)} users, skipping them in grading phase"
        )

    for user_id, user_responses in responses.items():
        if user_id in graded_user_ids:
            print(f"Skipping grading for {user_id} (already in judged checkpoint)")
            continue
        graded_responses = []

        semaphore = asyncio.Semaphore(30)  # Concurrency limit: 30

        async def grade_with_semaphore(response, semaphore=semaphore):
            async with semaphore:
                question = response.get("question")
                answer = response.get("answer")
                ground_truth = response.get("golden_answer")
                category = response.get("category")
                search_duration_ms = response.get("search_duration_ms", 0.0)

                if not ground_truth:
                    return None

                # Parallel grading tasks
                grading_tasks = [
                    locomo_grader(oai_client, question, ground_truth, answer)
                    for _ in range(num_runs)
                ]
                judgments = await asyncio.gather(*grading_tasks)
                judgments_dict = {f"judgment_{i + 1}": j for i, j in enumerate(judgments)}

                # Compute NLP metrics
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

        # Create all grading tasks
        grade_tasks = [grade_with_semaphore(response) for response in user_responses]

        # Show progress with tqdm
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

        # Checkpoint: save after each user's grading completes (convert numpy types before save, same as final format)
        to_save = convert_numpy_types(all_grades)
        _save_json(judged_path, to_save)
        print(f"Checkpoint: saved judged (user {user_id}) to {judged_path}")

    # Save judged results
    all_grades = convert_numpy_types(all_grades)
    with open(judged_path, "w") as f:
        json.dump(all_grades, f, indent=2)
    print(f"Saved detailed evaluation results to {judged_path}")

    return all_grades


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--client_type",
        type=str,
        choices=["openclaw", "memos", "openviking"],
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
        help="Disable checkpoint resume; start evaluation from scratch (ignore existing response/judged files)",
    )
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    oai_client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL")
    )

    # Load LoCoMo dataset
    data_path = "evaluation/data/locomo/locomo10.json"
    with open(data_path) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} users from LoCoMo dataset")

    # Initialize client
    client = OpenclawClient(
        apikey="xxxx",  # gateway.auth.token in .openclaw/openclaw.json
        baseurl="http://localhost:18789",
    )

    # Run evaluation (from scratch if --no_resume is set, else resume from checkpoint)
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
