import argparse
import asyncio
import json
import os
import sys
import time

from datetime import datetime

from dotenv import load_dotenv
from tqdm import tqdm

from evaluation.scripts.locomo.locomo_eval import (
    calculate_nlp_metrics,
    convert_numpy_types,
    locomo_grader,
)
from evaluation.scripts.utils.client import OpenclawClient, OpenclawMemOSClient


# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

load_dotenv()


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


async def evaluate_client(client_name, client, data, oai_client, num_runs=3):
    """
    Evaluate a client on the LoCoMo dataset
    """
    print(f"\n=== Evaluating {client_name} ===")

    results_dir = f"results/locomo/{client_name}"
    os.makedirs(results_dir, exist_ok=True)

    response_path = f"{results_dir}/{client_name}_locomo_responses.json"
    judged_path = f"{results_dir}/{client_name}_locomo_judged.json"

    responses = {}

    # Process each user's data
    for user_idx, user_data in enumerate(data):
        user_id = f"locomo_exp_user_{user_idx}"
        user_responses = []

        print(f"Processing user {user_id}...")

        # Step 1: Add conversation to memory
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
            client.add(messages, user_id, timestamp)
            add_duration = time.time() - start_time
            print(f"Added messages in {add_duration:.2f} seconds")
        except Exception as e:
            print(f"Error adding messages: {e}")
            continue

        time.sleep(60)  # Add delay to avoid rate limiting

        # Step 2: Process each QA pair in parallel
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

    # Save responses
    with open(response_path, "w") as f:
        json.dump(responses, f, indent=2)
    print(f"Saved responses to {response_path}")

    # Step 3: Evaluate responses
    print("\n=== Evaluating responses ===")

    all_grades = {}

    for user_id, user_responses in responses.items():
        graded_responses = []

        semaphore = asyncio.Semaphore(30)  # 并发限制为30

        async def grade_with_semaphore(response, semaphore=semaphore):
            async with semaphore:
                question = response.get("question")
                answer = response.get("answer")
                ground_truth = response.get("golden_answer")
                category = response.get("category")
                search_duration_ms = response.get("search_duration_ms", 0.0)

                if not ground_truth:
                    return None

                # 并行评分任务
                grading_tasks = [
                    locomo_grader(oai_client, question, ground_truth, answer)
                    for _ in range(num_runs)
                ]
                judgments = await asyncio.gather(*grading_tasks)
                judgments_dict = {f"judgment_{i + 1}": j for i, j in enumerate(judgments)}

                # 计算 NLP 指标
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

        # 创建所有评分任务
        grade_tasks = [grade_with_semaphore(response) for response in user_responses]

        # 使用 tqdm 显示进度
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

    # Save judged results
    all_grades = convert_numpy_types(all_grades)
    with open(judged_path, "w") as f:
        json.dump(all_grades, f, indent=2)
    print(f"Saved detailed evaluation results to {judged_path}")

    return all_grades


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--client",
        type=str,
        choices=["openclaw", "openclaw_memos"],
        default="openclaw_memos",
        help="Which client to evaluate",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=3,
        help="Number of times to run the LLM grader for each question",
    )
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Initialize OpenAI client for grading
    from openai import AsyncOpenAI

    oai_client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL")
    )

    # Load LoCoMo dataset
    data_path = "evaluation/data/locomo/locomo10.json"
    with open(data_path) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} users from LoCoMo dataset")

    # Initialize client
    if args.client == "openclaw":
        client = OpenclawClient(
            apikey="bb8804823a5140e82324a5417545e42a85b023c95abe77b5",
            baseurl="http://47.116.116.227:18789",
        )
        client_name = "openclaw"
    else:
        client = OpenclawMemOSClient(
            apikey="bb8804823a5140e82324a5417545e42a85b023c95abe77b5",
            baseurl="http://47.116.195.3:18789",
        )
        client_name = "openclaw_memos"

    # Run evaluation
    await evaluate_client(client_name, client, data, oai_client, args.num_runs)

    print("\n=== Evaluation Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
