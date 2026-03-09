import argparse
import asyncio
import json
import os
import sys
import time

from datetime import datetime

from dotenv import load_dotenv
from tqdm import tqdm

from scripts.longmemeval.lme_eval import calculate_nlp_metrics, convert_numpy_types, lme_grader
from scripts.utils.client import OpenclawClient, OpenclawMemOSClient


# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

load_dotenv()


def parse_datetime(date_time_str):
    """
    Parse datetime string from the dataset
    Example: "2023/05/20 (Sat) 02:21"
    """
    try:
        # Remove weekday in parentheses and keep the space between date and time
        # Format: "2023/05/20 (Sat) 02:21" -> "2023/05/20 02:21"
        if " (" in date_time_str and ") " in date_time_str:
            # Split by ' (' to get date part, then split by ') ' to get time part
            date_part = date_time_str.split(" (")[0]
            time_part = date_time_str.split(") ")[1]
            cleaned_str = f"{date_part} {time_part}"
        else:
            cleaned_str = date_time_str
        # Parse the datetime
        dt = datetime.strptime(cleaned_str, "%Y/%m/%d %H:%M")
        return dt.timestamp()
    except ValueError as e:
        print(f"Error parsing datetime '{date_time_str}': {e}")
        # If parsing fails, return current timestamp as fallback
        return time.time()


def process_conversation(conversation_sessions, haystack_dates):
    """
    Process conversation data into messages format for client.add()
    """
    messages = []

    # Process each session with corresponding date
    for _, (session, date_str) in enumerate(
        zip(conversation_sessions, haystack_dates, strict=False)
    ):
        timestamp = parse_datetime(date_str)

        # Process each message in the session
        for msg in session:
            time_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            messages.append(
                {
                    "role": msg["role"],
                    "content": msg["content"][:8000],
                    "timestamp": time_str,
                }
            )
    return messages


async def process_qa_pair(client, qa, user_id, loop):
    """
    Process a single QA pair
    """
    question = qa.get("question")
    golden_answer = qa.get("answer")
    question_date = qa.get("question_date")

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
            "question_date": question_date,
            "search_duration_ms": search_duration * 1000,
        }

    except Exception as e:
        print(f"Error searching for question '{question}': {e}")
        return None


async def evaluate_client(client_name, client, data, oai_client, num_runs=3):
    """
    Evaluate a client on the LongMemEval dataset
    """
    print(f"\n=== Evaluating {client_name} ===")

    results_dir = f"/Users/niu/code/Memos/evaluation/evaluation_openclaw/results/lme/{client_name}"
    os.makedirs(results_dir, exist_ok=True)

    response_path = f"{results_dir}/{client_name}_lme_responses.json"
    judged_path = f"{results_dir}/{client_name}_lme_judged.json"

    responses = {}

    # Process each user's data
    for user_idx, user_data in enumerate(data):
        user_id = f"lme_exp_user_{user_idx}"
        user_responses = []

        print(f"Processing user {user_id}...")

        # Step 1: Add conversation to memory
        conversation_sessions = user_data.get("haystack_sessions", [])
        haystack_dates = user_data.get("haystack_dates", [])

        if len(conversation_sessions) != len(haystack_dates):
            print(
                f"Warning: Mismatch between number of sessions ({len(conversation_sessions)}) and dates ({len(haystack_dates)})"
            )
            continue

        messages = process_conversation(conversation_sessions, haystack_dates)

        time.sleep(60)  # Add delay to avoid rate limiting

        print(
            f"Adding {len(messages)} messages to memory for {user_id}...current time: {datetime.now()}"
        )
        start_time = time.time()

        try:
            # Extract timestamp from messages (use the first one if available)
            timestamp = messages[0]["timestamp"] if messages else time.time()
            # Add messages to memory
            client.add(messages, user_id, timestamp, batch_size=100)
            add_duration = time.time() - start_time
            print(f"Added messages in {add_duration:.2f} seconds")
        except Exception as e:
            print(f"Error adding messages: {e}")
            continue

        time.sleep(60)

        # Step 2: Process QA pair
        print(f"Processing QA pair for {user_id}...current time: {datetime.now()}")

        # Get event loop
        loop = asyncio.get_event_loop()

        # Process QA pair
        qa_result = await process_qa_pair(client, user_data, user_id, loop)
        if qa_result:
            user_responses.append(qa_result)

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

        for response in tqdm(
            user_responses, desc=f"Grading responses for {user_id}", unit="response"
        ):
            question = response.get("question")
            answer = response.get("answer")
            ground_truth = response.get("golden_answer")
            question_date = response.get("question_date")
            search_duration_ms = response.get("search_duration_ms", 0.0)

            if not ground_truth:
                continue

            # Grade the response
            grading_tasks = [
                lme_grader(oai_client, question, ground_truth, answer) for _ in range(num_runs)
            ]
            judgments = await asyncio.gather(*grading_tasks)
            judgments_dict = {f"judgment_{i + 1}": j for i, j in enumerate(judgments)}

            # Calculate NLP metrics
            nlp_metrics = calculate_nlp_metrics(ground_truth, answer, "", ["lexical", "semantic"])

            graded_response = {
                "user_id": user_id,
                "question": question,
                "question_date": question_date,
                "golden_answer": ground_truth,
                "answer": answer,
                "llm_judgments": judgments_dict,
                "nlp_metrics": nlp_metrics,
                "search_duration_ms": search_duration_ms,
                "total_duration_ms": search_duration_ms,
            }
            graded_responses.append(graded_response)

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
        choices=["openclaw", "openclaw_memos", "openclaw_qmd"],
        default="openclaw_qmd",  # TODO
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
    load_dotenv("/Users/niu/code/Memos/evaluation/evaluation_openclaw/.env")

    # Initialize OpenAI client for grading
    from openai import OpenAI

    oai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))

    # Load LongMemEval dataset
    data_path = (
        "/Users/niu/code/Memos/evaluation/evaluation_openclaw/data/longmemeval/longmemeval_s.json"
    )
    with open(data_path) as f:
        data = json.load(f)
        length = len(data)
        data = data[: length // 10]

    print(f"Loaded {len(data)} users from LongMemEval dataset")

    # Initialize client
    if args.client == "openclaw":
        client = OpenclawClient(
            apikey="bb8804823a5140e82324a5417545e42a85b023c95abe77b5",
            baseurl="http://47.116.127.182:18789",
        )
        client_name = "openclaw"
    elif args.client == "openclaw_memos":
        client = OpenclawMemOSClient(
            apikey="bb8804823a5140e82324a5417545e42a85b023c95abe77b5",
            baseurl="http://47.116.195.3:18789",
        )
        client_name = "openclaw_memos"
    elif args.client == "openclaw_qmd":
        client = OpenclawClient(
            apikey="bb8804823a5140e82324a5417545e42a85b023c95abe77b5",
            baseurl="http://47.116.200.71:18789",
        )
        client_name = "openclaw_qmd"
    else:
        raise ValueError

    # Run evaluation
    await evaluate_client(client_name, client, data, oai_client, args.num_runs)

    print("\n=== Evaluation Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
