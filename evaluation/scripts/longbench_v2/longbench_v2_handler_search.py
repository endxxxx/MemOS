import argparse
import json
import os
import sys
import time
import traceback

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm


# Add project root and src to sys.path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src"))

from evaluation.scripts.utils.metrics import Metrics
from memos.api import handlers
from memos.api.handlers.add_handler import AddHandler
from memos.api.handlers.base_handler import HandlerDependencies
from memos.api.handlers.search_handler import SearchHandler
from memos.api.product_models import APIADDRequest, APISearchRequest


load_dotenv()
memos_knowledgebase_id = os.getenv("MEMOS_KNOWLEDGEBASE_ID_LONGBENCH_V2")

# Initialize handlers
print("=" * 80)
print("Initializing service components...")
print("=" * 80)
# Use init_server from handlers as in add_from_handler.py
components = handlers.init_server()
dependencies = HandlerDependencies.from_init_server(components)
search_handler = SearchHandler(dependencies)
add_handler = AddHandler(dependencies)


def add_memories(add_req: APIADDRequest):
    """
    Add memories using the local AddHandler.
    Included to satisfy the requirement of using 'add' way from add_from_handler.py.
    """
    return add_handler.handle_add_memories(add_req)


def search_memories(search_req: APISearchRequest):
    """
    Search memories using the local SearchHandler.
    """
    return search_handler.handle_search_memories(search_req)


def _load_dataset_jsonl(dataset_path: Path) -> list[dict]:
    samples: list[dict] = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def _load_existing_results(output_path: Path) -> tuple[list[dict], set[str]]:
    if not output_path.exists():
        return [], set()
    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            ids = {str(r.get("_id")) for r in data if r.get("_id")}
            return data, ids
        if isinstance(data, dict) and isinstance(data.get("results"), list):
            rows = data.get("results") or []
            ids = {str(r.get("_id")) for r in rows if r.get("_id")}
            return rows, ids
    except Exception:
        return [], set()
    return [], set()


def _save_json_list(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({"results": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def handler_search_one(sample: dict, top_k: int, version_dir: str, search_mode: str) -> dict:
    sample_id = str(sample.get("_id"))
    user_id = version_dir + "_" + sample_id
    question = sample.get("question") or ""
    choices = {
        "A": sample.get("choice_A") or "",
        "B": sample.get("choice_B") or "",
        "C": sample.get("choice_C") or "",
        "D": sample.get("choice_D") or "",
    }

    # Construct search request
    readable_cube_ids = []
    if memos_knowledgebase_id:
        readable_cube_ids.append(memos_knowledgebase_id)

    req = APISearchRequest(
        query=str(question),
        user_id=user_id,
        top_k=top_k,
        mode=search_mode,
        readable_cube_ids=readable_cube_ids if readable_cube_ids else None,
    )

    try:
        # Use the local handler search function
        response = search_memories(req)

        # Parse results
        # SearchHandler returns a SearchResponse object, data is in response.data
        results_data = response.data
        memories = []

        # Extract memories from the structured response
        if results_data and isinstance(results_data, dict):
            if "text_mem" in results_data:
                for bucket in results_data["text_mem"]:
                    for mem in bucket.get("memories", []):
                        mem_content = mem.get("memory", "")
                        if mem_content:
                            memories.append(mem_content)
            # Also check for 'memory_detail_list' just in case structure differs
            elif "memory_detail_list" in results_data:
                for m in results_data["memory_detail_list"]:
                    mem_content = m.get("memory_value", "")
                    if mem_content:
                        memories.append(mem_content)

    except Exception as e:
        print(f"Search failed for sample {sample_id}: {e}")
        traceback.print_exc()
        memories = []

    print(f"[Handler Search] sample_id: {sample_id} search memories: {len(memories)}")

    return {
        "_id": sample_id,
        "domain": sample.get("domain"),
        "sub_domain": sample.get("sub_domain"),
        "difficulty": sample.get("difficulty"),
        "length": sample.get("length"),
        "question": question,
        "choices": choices,
        "answer": sample.get("answer"),
        "memories_used": memories,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Longbench-v2 Product Search via Local Handlers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # lib argument is kept for compatibility but not strictly used for switching libs
    parser.add_argument(
        "--lib", "-b", default="memos-handler", help="Product name (default: memos-handler)"
    )
    parser.add_argument(
        "--dataset-path",
        "-s",
        default="evaluation/data/longbench_v2/longbenchv2_train.json",
        help="Path to JSON file containing samples",
    )
    parser.add_argument("--workers", "-c", type=int, default=5, help="Concurrency (default: 5)")
    parser.add_argument(
        "--timeout", type=float, default=120.0, help="Request timeout in seconds (default: 120)"
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=30,
        help="Number of results to return per search (default: 20)",
    )
    parser.add_argument("--version-dir", "-v", default=None, help="Version directory name")
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Limit number of samples to process",
    )
    parser.add_argument(
        "--mode", "-m", type=str, default="fast", help="Search mode (default: fast)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("Longbench-v2 Product Search (Local Handler)")
    print("=" * 60)

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    dataset = _load_dataset_jsonl(dataset_path)
    if args.limit is not None:
        dataset = dataset[: args.limit]

    version = args.version_dir if args.version_dir else "default"
    base_dir = os.path.join("evaluation", "results", "longbench_v2", version)
    output_dir = base_dir
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{args.lib}_search_results.json"
    output_path = Path(os.path.join(output_dir, output_filename))

    results, processed_ids = _load_existing_results(output_path)
    pending = [s for s in dataset if str(s.get("_id")) not in processed_ids]
    if not pending:
        print("All samples processed.")
        return

    metrics = Metrics()
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:

        def do_search(sample: dict) -> dict:
            st = time.perf_counter()
            r = handler_search_one(
                sample, args.top_k, args.version_dir if args.version_dir else "default", args.mode
            )
            dur = time.perf_counter() - st
            r["duration_ms"] = dur * 1000
            metrics.record(dur, True)
            return r

        futures = [executor.submit(do_search, sample) for sample in pending]
        for idx, f in enumerate(
            tqdm(as_completed(futures), total=len(futures), desc="Searching"), start=1
        ):
            try:
                r = f.result()
                results.append(r)
                if idx % 20 == 0:
                    _save_json_list(output_path, results)
            except Exception as e:
                metrics.record(0.0, False, str(e))
                print(f"[Search] Error: {e}")
                traceback.print_exc()

    _save_json_list(output_path, results)
    print(f"[Search] saved {len(results)} rows to {output_path}")

    total_duration = time.time() - start_time
    summary = metrics.summary()
    combined_obj = {
        "perf": {
            "summary": summary,
            "total_duration": total_duration,
            "config": {
                "workers": args.workers,
                "top_k": args.top_k,
                "dataset_path": str(dataset_path),
                "limit": args.limit,
                "mode": args.mode,
            },
        },
        "results": results,
    }
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(json.dumps(combined_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, output_path)

    print("\n" + "=" * 60)
    print("Search finished! Statistics:")
    print("=" * 60)
    print(f"Total duration: {total_duration:.2f}s")
    print(f"Success: {summary['counts']['success']} / Failed: {summary['counts']['failed']}")

    if summary["stats"]:
        stats = summary["stats"]
        qps = stats["count"] / total_duration if total_duration > 0 else 0
        print(f"QPS: {qps:.2f}")
        print("Latency stats (ms):")
        print(f"  Mean: {stats['mean']:.2f}")
        print(f"  Median: {stats['median']:.2f}")
        print(f"  Min: {stats['min']:.2f}")
        print(f"  Max: {stats['max']:.2f}")
        print(f"  P95: {stats['p95']:.2f}")
        print(f"  P99: {stats['p99']:.2f}")

    if summary["errors"]:
        print("\nError stats:")
        for error, count in sorted(summary["errors"].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  [{count} times] {error[:100]}...")


if __name__ == "__main__":
    main()
