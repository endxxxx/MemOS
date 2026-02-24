import os
import sys


# Add project root to sys.path
sys.path.append(os.getcwd())

import argparse
import json
import time
import uuid

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from evaluation.scripts.utils.metrics import Metrics
from memos.api import handlers
from memos.api.handlers.add_handler import AddHandler
from memos.api.handlers.base_handler import HandlerDependencies
from memos.api.product_models import APIADDRequest


load_dotenv()
MEMOS_KNOWLEDGEBASE_ID_LONGBENCH_V2 = os.getenv("MEMOS_KNOWLEDGEBASE_ID_LONGBENCH_V2")

# Initialize Handler
print("Initializing HandlerDependencies...")
components = handlers.init_server()
handler_dependencies = HandlerDependencies.from_init_server(components)
add_handler = AddHandler(handler_dependencies)
print("Handler initialized.")


def add_memories(add_req: APIADDRequest):
    return add_handler.handle_add_memories(add_req)


def _load_dataset_jsonl(dataset_path: Path) -> list[dict]:
    # Default to the specific file if not exists, as requested by user
    # But user said "load from evaluation/data/longbench_v2/longbenchv2_train.json"
    # The main function sets this as default argument.

    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        return []

    samples: list[dict] = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return samples


def _load_added_ids(records_path: Path) -> dict[str, str | None]:
    if not records_path.exists():
        return {}
    try:
        obj = json.loads(records_path.read_text(encoding="utf-8"))
        added = obj.get("added") if isinstance(obj, dict) else None
        if isinstance(added, dict):
            return {str(k): (str(v) if v is not None else None) for k, v in added.items()}
    except Exception:
        return {}
    return {}


def _save_added_ids(
    records_path: Path,
    added: dict[str, str | None],
    perf: dict | None = None,
) -> None:
    records_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = records_path.with_suffix(records_path.suffix + ".tmp")

    obj = {
        "added": dict(sorted(added.items())),
    }
    if perf is not None:
        obj["perf"] = perf

    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, records_path)


def ingest_context(
    sample: dict,
    url_prefix: str,
    async_mode: str = "sync",
    version_dir: str | None = None,
) -> tuple[str, str]:
    sample_id = str(sample.get("_id"))
    user_id = version_dir + "_" + sample_id if version_dir else sample_id

    # Construct file URL
    file_url = f"{url_prefix.rstrip('/')}/{sample_id}.txt"
    file_id = uuid.uuid4().hex

    # Determine writable cube ids
    # If MEMOS_KNOWLEDGEBASE_ID_LONGBENCH_V2 is set, use it.
    # Otherwise, it might default to user_id based cube.
    # But for LongBench V2, we usually want to use the specific knowledge base if available?
    # The original script uses memos_knowledgebase_id for memos-api.
    writable_cube_ids = [MEMOS_KNOWLEDGEBASE_ID_LONGBENCH_V2]

    # Construct Request
    # Following add_from_handler.py example for file addition
    req = APIADDRequest.model_validate(
        {
            "user_id": user_id,
            "session_id": "default_session",
            "async_mode": async_mode,
            "writable_cube_ids": writable_cube_ids,
            "messages": [
                {
                    "file": {
                        "file_data": file_url,
                        "file_id": file_id,
                        "filename": f"{sample_id}.txt",
                    },
                    "type": "file",
                }
            ],
        }
    )

    # Call Handler
    result = add_memories(req)

    # Extract result info
    # The result.data is a list of results.
    # We return sample_id and the file_id (or memory id)
    # In original script: file_id = result["data"][0]["id"]
    # Here result.data is likely a list of added memory items.

    # If result.data is list, take the first one's ID?
    # Or just return the file_id we generated?
    # The original script returns file_id obtained from API.
    # Let's inspect result structure if possible, but safely assume success means we can return our file_id or the one from result.

    res_file_id = file_id
    if result.data and isinstance(result.data, list) and len(result.data) > 0:
        # Assuming the first item corresponds to our file add
        item = result.data[0]
        if isinstance(item, dict):
            res_file_id = item.get("id", file_id)
        elif hasattr(item, "id"):
            res_file_id = item.id

    return sample_id, res_file_id


def parse_args():
    parser = argparse.ArgumentParser(
        description="LongBench-v2 Handler Add Concurrent Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # lib argument is kept for compatibility but effectively ignored or must be 'memos-handler'
    parser.add_argument(
        "--lib", "-b", default="memos-handler", help="Product name (default: memos-handler)"
    )

    parser.add_argument("--workers", "-w", type=int, default=5, help="Concurrency (default: 5)")

    parser.add_argument(
        "--async-mode", default="sync", choices=["sync", "async"], help="Async mode (default: sync)"
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
        "--url-prefix",
        "-u",
        default="https://memos-knowledge-base-file-pre.oss-cn-shanghai.aliyuncs.com/longbench_v2_text_files/",
        help="URL prefix to be prepended to filenames",
    )

    parser.add_argument(
        "--dataset_path",
        "-p",
        default="evaluation/data/longbench_v2/longbenchv2_train.json",
        help="Dataset path",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("LongBench-v2 Handler Add Tool")
    print("=" * 60)

    dataset_path = Path(args.dataset_path)
    print(f"Loading dataset from {dataset_path}...")
    dataset = _load_dataset_jsonl(dataset_path)
    if not dataset:
        print("No samples found or file does not exist.")
        return

    if args.limit is not None:
        dataset = dataset[: args.limit]

    version_dir_name = args.version_dir if args.version_dir else "default"
    base_dir = Path(f"evaluation/results/longbench_v2/{version_dir_name}")
    version_output_dir = str(base_dir)
    os.makedirs(version_output_dir, exist_ok=True)

    # Use lib name in output file
    output_path = os.path.join(version_output_dir, f"{args.lib}_add_results.json")
    output_path = Path(output_path)

    added_ids: dict[str, str | None] = _load_added_ids(output_path)
    pending = [s for s in dataset if str(s.get("_id")) not in added_ids]

    print(
        f"[Add] lib={args.lib} total={len(dataset)} pending={len(pending)} workers={args.workers}"
    )
    if not pending:
        return

    metrics = Metrics()

    def do_ingest(sample):
        start_time = time.perf_counter()
        try:
            sample_id, file_id = ingest_context(
                sample,
                args.url_prefix,
                args.async_mode,
                version_dir_name,  # Use version_dir_name as user_id prefix
            )
            duration = time.perf_counter() - start_time
            metrics.record(duration, True)
            return sample_id, file_id
        except Exception as e:
            duration = time.perf_counter() - start_time
            metrics.record(duration, False, str(e))
            raise e

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(do_ingest, sample) for sample in pending]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Adding"):
            try:
                sid, fid = f.result()
                if sid:
                    sid = str(sid)
                    added_ids[sid] = str(fid) if fid else None
                    if len(added_ids) % 10 == 0:
                        _save_added_ids(output_path, added_ids)

            except Exception as e:
                print(f"[Add] Error: {e}")

    _save_added_ids(output_path, added_ids)
    print(f"[Add] saved records to {output_path}")

    total_duration = time.time() - start_time
    summary = metrics.summary()

    _save_added_ids(
        output_path,
        added_ids,
        perf={
            "summary": summary,
            "total_duration": total_duration,
            "config": {
                "workers": args.workers,
                "async_mode": args.async_mode,
                "dataset_path": str(args.dataset_path),
            },
        },
    )

    print("\n" + "=" * 60)
    print("Ingestion finished! Statistics:")
    print("=" * 60)
    print(f"Total duration: {total_duration:.2f}s")
    print(f"Success: {summary['counts']['success']} / Failed: {summary['counts']['failed']}")

    if summary["stats"]:
        stats = summary["stats"]
        qps = stats["count"] / total_duration if total_duration > 0 else 0
        print(f"QPS: {qps:.2f}")

    if summary["errors"]:
        print("\nError stats:")
        for error, count in sorted(summary["errors"].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  [{count} times] {error[:100]}...")


if __name__ == "__main__":
    main()
