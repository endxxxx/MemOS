import argparse
import json
import os
import threading
import time
import traceback
import uuid

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from tqdm import tqdm

from evaluation.scripts.utils.client import get_lib_client
from evaluation.scripts.utils.metrics import Metrics


load_dotenv()
fastgpt_dataset_id = os.getenv("FASTGPT_DATASET_ID_LONGBENCH_V2")
memos_knowledgebase_id = os.getenv("MEMOS_KNOWLEDGEBASE_ID_LONGBENCH_V2")
dify_dataset_id = os.getenv("DIFY_DATASET_ID_LONGBENCH_V2")
coze_space_id = os.getenv("COZE_SPACE_ID")
coze_dataset_id = os.getenv("COZE_DATASET_ID_LONGBENCH_V2")
_records_write_lock = threading.Lock()


def _load_dataset_jsonl(dataset_path: Path) -> list[dict]:
    if not dataset_path.exists():
        dataset = load_dataset("zai-org/LongBench-v2", split="train")
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dataset_path, "w", encoding="utf-8") as f:
            for i in range(len(dataset)):
                s = dataset[i]
                row = {
                    "_id": s.get("_id") or s.get("id") or str(i),
                    "domain": s.get("domain"),
                    "sub_domain": s.get("sub_domain"),
                    "difficulty": s.get("difficulty"),
                    "length": s.get("length"),
                    "question": s.get("question"),
                    "choice_A": s.get("choice_A"),
                    "choice_B": s.get("choice_B"),
                    "choice_C": s.get("choice_C"),
                    "choice_D": s.get("choice_D"),
                    "answer": s.get("answer"),
                    "context": s.get("context") or s.get("document") or s.get("documents"),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Successfully saved dataset to {dataset_path}")

    samples: list[dict] = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def _load_added_ids(records_path: Path) -> dict[str, str | None]:
    """
    Load added records as a mapping: sample_id -> file_id (or None).
    """
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
    with _records_write_lock:
        records_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = records_path.with_suffix(records_path.suffix + f".tmp.{uuid.uuid4().hex}")
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


def ingest_coze_documents(client) -> None:
    import base64

    merged_dir = os.path.abspath("evaluation/data/longbench_v2/merged_txt")
    records_path = Path(
        "evaluation/results/longbench_v2/coze_longbench_v2_0211/coze_add_results.json"
    )
    existing_map = _load_added_ids(records_path)
    try:
        names = sorted(
            [n for n in os.listdir(merged_dir) if n.startswith("doc_") and n.endswith(".txt")]
        )
    except Exception:
        names = []

    for i in range(0, len(names), 5):
        batch_names = names[i : i + 5]
        batch_added = {}
        for fname in batch_names:
            if isinstance(existing_map, dict) and existing_map.get(fname):
                continue
            full_path = os.path.join(merged_dir, fname)
            with open(full_path, "rb") as f:
                file_b64 = base64.b64encode(f.read()).decode("utf-8")
            document_bases = [
                {
                    "source_info": {
                        "file_base64": file_b64,
                        "file_type": "txt",
                        "document_source": 0,
                    },
                    "name": fname,
                }
            ]

            print(fname)
            result = client.create_document(
                dataset_id=coze_dataset_id,
                document_bases=document_bases,
            )
            print(result)

            rid = ""
            if isinstance(result, dict):
                code = result.get("code")
                if code is None or code == 0:
                    infos = result.get("document_infos")
                    if isinstance(infos, list) and infos:
                        first = infos[0]
                        rid = str(first.get("document_id") or first.get("id") or "")
                    if not rid:
                        data = result.get("data")
                        if isinstance(data, dict):
                            rid = str(data.get("id") or data.get("document_id") or "")
                        elif isinstance(data, list) and data:
                            item = data[0]
                            rid = str(item.get("id") or item.get("document_id") or "")
            if rid:
                batch_added[fname] = rid
        existing_map2 = _load_added_ids(records_path)
        existing_map2.update(batch_added)
        _save_added_ids(records_path, existing_map2)


def ingest_context(
    client,
    sample: dict,
    lib: str,
    url_prefix: str,
    mode: str = "fine",
    async_mode: str = "sync",
    version_dir: str | None = None,
) -> tuple[str, str]:
    sample_id = str(sample.get("_id"))
    user_id = version_dir + "_" + sample_id
    context = sample.get("context") or ""
    ts = int(time.time())
    file_url = f"{url_prefix.rstrip('/')}/{sample_id}.txt"

    file_id = ""
    if lib == "memos-api" or lib == "memos-api-online":
        result = client.upload_file(memos_knowledgebase_id, file_url)
        file_id = result["data"][0]["id"]
    if lib == "fastgpt":
        result = client.upload_file(dataset_id=fastgpt_dataset_id, file_url=file_url)
        file_id = result["data"]["collectionId"]
    if lib == "mem0":
        chunker = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=2048, chunk_overlap=128
        )
        chunks = [p for p in chunker.split_text(context or "") if p.strip()]

        messages = [{"role": "user", "content": p} for p in chunks]
        client.add(messages=messages, user_id=user_id, timestamp=ts, batch_size=10)

    if lib == "supermemory":
        client.add(content=context, user_id=user_id)

    if lib == "dify":
        documents_dir = os.path.abspath("evaluation/data/longbench_v2/documents")
        os.makedirs(documents_dir, exist_ok=True)
        file_path = os.path.join(documents_dir, f"{sample_id}.txt")

        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(context)

        result = client.upload_file(
            dataset_id=dify_dataset_id, file_url=file_path, mime_type="text/plain"
        )
        file_id = result["batch"]

    return sample_id, file_id


def parse_args():
    parser = argparse.ArgumentParser(
        description="LongBench-v2 Product Add Concurrent Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--lib", "-b", required=True, help="Product name to evaluate")

    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8001",
        help="MemOS API URL (default: http://127.0.0.1:8001)",
    )

    parser.add_argument("--workers", "-w", type=int, default=5, help="Concurrency (default: 10)")

    parser.add_argument(
        "--timeout", type=float, default=1200, help="Request timeout in seconds (default: 120)"
    )

    parser.add_argument(
        "--mode", default="fine", choices=["fine", "fast"], help="Processing mode (default: fine)"
    )

    parser.add_argument(
        "--async-mode", default="sync", choices=["sync", "async"], help="Async mode (default: sync)"
    )

    parser.add_argument("--version-dir", "-v", default=None, help="Version directory name")

    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Limit number of samples to process (for testing, default all)",
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


def main() -> None:
    args = parse_args()
    print("=" * 60)
    print("LongBench-v2 Product Add Concurrent Tool")
    print("=" * 60)

    dataset_path = Path(args.dataset_path)
    dataset = _load_dataset_jsonl(dataset_path)
    if args.limit is not None:
        dataset = dataset[: args.limit]

    base_dir = "evaluation/results/longbench_v2"
    version_output_dir = os.path.join(
        base_dir, args.version_dir if args.version_dir else "version_default"
    )
    os.makedirs(version_output_dir, exist_ok=True)
    output_path = os.path.join(version_output_dir, f"{args.lib}_add_results.json")
    output_path = Path(output_path)

    added_ids: dict[str, str | None] = _load_added_ids(output_path)
    pending = [s for s in dataset if str(s.get("_id")) not in added_ids]

    print(
        f"[Add] lib={args.lib} total={len(dataset)} pending={len(pending)} workers={args.workers}"
    )
    if not pending:
        return

    client = get_lib_client(args.lib)
    metrics = Metrics()

    if args.lib == "coze":
        ingest_coze_documents(client)
        return

    def do_ingest(sample):
        start_time = time.perf_counter()
        try:
            sample_id, file_id = ingest_context(
                client,
                sample,
                args.lib,
                args.url_prefix,
                args.mode,
                args.async_mode,
                args.version_dir,
            )
            duration = time.perf_counter() - start_time
            metrics.record(duration, True)
            return sample_id, file_id
        except Exception as e:
            traceback.print_exc()
            duration = time.perf_counter() - start_time
            metrics.record(duration, False, str(e))
            raise e

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(do_ingest, sample) for sample in pending]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Adding"):
            try:
                sid, fid = f.result()
                if sid and fid:
                    sid = str(sid)
                    added_ids[sid] = str(fid)
                    if len(added_ids) % 10 == 0:
                        _save_added_ids(output_path, added_ids)

            except Exception as e:
                print(f"[Add] Error: {e}")
                traceback.print_exc()

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
                "mode": args.mode,
                "async_mode": args.async_mode,
                "dataset_path": args.dataset_path,
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
