#!/usr/bin/env python3

import argparse
import json
import os
import threading
import time

from dotenv import load_dotenv

from evaluation.scripts.utils.client import get_lib_client
from evaluation.scripts.utils.metrics import Metrics


load_dotenv()

fastgpt_dataset_id = os.getenv("FASTGPT_DATASET_ID_MM_LONGBENCH")
memos_knowledgebase_id = os.getenv("MEMOS_KNOWLEDGEBASE_ID_MM_LONGBENCH")
dify_dataset_id = os.getenv("DIFY_DATASET_ID_MM_LONGBENCH")
coze_dataset_id = os.getenv("COZE_DATASET_ID_MM_LONGBENCH")


def read_filenames(filepath: str) -> list[str]:
    """
    Read filename list from file
    Supports one filename per line, automatically removes empty lines and whitespace
    """
    filenames = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                filenames.append(line)
    return filenames


def run_concurrent_add(
    lib: str,
    filenames: list[str],
    url_prefix: str,
    user_prefix: str,
    workers: int,
    mode: str | None = "fine",
    async_mode: str | None = "sync",
    output_path: str | None = None,
    existing_added: dict | None = None,
    config: dict | None = None,
) -> dict:
    """
    Execute concurrent add operations

    Args:
        lib: Client name
        filenames: List of filenames
        url_prefix: URL prefix
        user_prefix: User ID prefix
        workers: Concurrency
        mode: Mode
        async_mode: Async mode
        output_path: Path to save output
        existing_added: Dictionary of already added files
        config: Configuration dictionary

    Returns:
        Statistics result
    """

    client = get_lib_client(lib)
    metrics = Metrics()
    total_files = len(filenames)
    completed = 0
    completed_lock = threading.Lock()
    file_write_lock = threading.Lock()

    added_ids: dict[str, str] = {}
    all_added_ids = dict(existing_added) if existing_added else {}

    start_time = time.time()

    def save_checkpoint():
        if not output_path:
            return

        try:
            current_duration = time.time() - start_time
            summary = metrics.summary()

            checkpoint_data = {
                "summary": summary,
                "total_duration": current_duration,
                "config": config,
                "added": dict(sorted(all_added_ids.items())),
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"Warning: Failed to save checkpoint: {e}")

    def add_single_file(filename: str, doc_id: str = ""):
        nonlocal completed

        if lib in ("dify", "coze"):
            file_url = os.path.abspath(
                os.path.join("evaluation/data/mmlongbench/documents", filename)
            )
        else:
            file_url = f"{url_prefix.rstrip('/')}/{filename}"

        start_time = time.perf_counter()

        result = None
        try:
            current_file_id = None
            if lib == "memos-api-online":
                result = client.upload_file(memos_knowledgebase_id, file_url)
                if isinstance(result, dict):
                    data = result.get("data") or []
                    if isinstance(data, list) and data:
                        first = data[0] if isinstance(data[0], dict) else {}
                        fid = first.get("id")
                        if fid:
                            current_file_id = str(fid)
            elif lib == "fastgpt":
                result = client.upload_file(dataset_id=fastgpt_dataset_id, file_url=file_url)
                current_file_id = result["data"]["collectionId"]
            elif lib == "dify":
                result = client.upload_file(dataset_id=dify_dataset_id, file_url=file_url)
                current_file_id = result["batch"]
            elif lib == "coze":
                if not coze_dataset_id:
                    raise RuntimeError("COZE_DATASET_ID_MM_LONGBENCH not set")
                import base64

                with open(file_url, "rb") as f:
                    file_b64 = base64.b64encode(f.read()).decode("utf-8")
                document_bases = [
                    {
                        "source_info": {
                            "file_base64": file_b64,
                            "file_type": "pdf",
                            "document_source": 0,
                        },
                        "name": filename,
                    }
                ]
                result = client.create_document(
                    dataset_id=coze_dataset_id,
                    document_bases=document_bases,
                    chunk_strategy=None,
                    format_type=0,
                )
                try:
                    if isinstance(result, dict):
                        code = result.get("code")
                        if code is not None and code != 0:
                            current_file_id = None
                        else:
                            infos = result.get("document_infos")
                            if isinstance(infos, list) and infos:
                                first = infos[0]
                                current_file_id = str(
                                    first.get("document_id") or first.get("id") or ""
                                )
                            if not current_file_id:
                                data = result.get("data")
                                if isinstance(data, dict):
                                    current_file_id = str(
                                        data.get("id") or data.get("document_id") or ""
                                    )
                                elif isinstance(data, list) and data:
                                    item = data[0]
                                    current_file_id = str(
                                        item.get("id") or item.get("document_id") or ""
                                    )
                    if not current_file_id:
                        resp_list = client.list_documents(dataset_id=coze_dataset_id)
                        if isinstance(resp_list, dict):
                            items = resp_list.get("data") or resp_list.get("documents") or []
                            for it in items:
                                name = str(it.get("name") or "")
                                doc_id_val = str(it.get("id") or it.get("document_id") or "")
                                if name == filename and doc_id_val:
                                    current_file_id = doc_id_val
                                    break
                except Exception:
                    pass

            if current_file_id:
                with file_write_lock:
                    added_ids[filename] = current_file_id
                    all_added_ids[filename] = current_file_id
                    save_checkpoint()

            duration = time.perf_counter() - start_time
            metrics.record(duration, True)

            with completed_lock:
                completed += 1
                print(
                    f"[{completed}/{total_files}] ✓ Success: {filename} ({duration * 1000:.0f}ms)"
                )

            return True, result

        except Exception as e:
            duration = time.perf_counter() - start_time
            error_msg = str(e)
            metrics.record(duration, False, error_msg)

            with completed_lock:
                completed += 1
                print(f"[{completed}/{total_files}] ✗ Failed: {filename} - {error_msg[:100]}")

            return False, error_msg

    print(f"\nStarting concurrent add for {total_files} files...")
    print(f"Concurrency: {workers}")
    print(f"Version Dir: {user_prefix}")
    print(f"URL prefix: {url_prefix}")
    print("-" * 60)

    start_time = time.time()

    results = []
    batch_size = 5
    pause_seconds = 5
    total_batches = (len(filenames) + batch_size - 1) // batch_size
    for batch_index in range(total_batches):
        start = batch_index * batch_size
        end = min(start + batch_size, len(filenames))
        current_batch = filenames[start:end]
        print(f"[Batch {batch_index + 1}/{total_batches}] size={len(current_batch)}")
        for filename in current_batch:
            doc_id = filename[:-3] + ".pdf"
            success, result = add_single_file(filename, doc_id)
            results.append({"filename": filename, "success": success, "result": result})
        save_checkpoint()
        if batch_index < total_batches - 1:
            print(f"[Batch {batch_index + 1}] pause {pause_seconds}s before next batch")
            time.sleep(pause_seconds)

    end_time = time.time()
    total_duration = end_time - start_time

    # Print statistics
    summary = metrics.summary()

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
        print("\nError statistics:")
        for error, count in sorted(summary["errors"].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  [{count} times] {error[:100]}...")

    return {
        "summary": summary,
        "total_duration": total_duration,
        "results": results,
        "added": added_ids,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="MMLongbench-doc Product Add Concurrent Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--lib", "-b", required=True, help="Product name to evaluate")
    parser.add_argument(
        "--filenames-file",
        "-f",
        default="evaluation/data/mmlongbench/pdf_file_list.txt",
        help="Path to text file containing list of filenames (one per line)",
    )

    parser.add_argument(
        "--url-prefix",
        "-u",
        default="https://memos-knowledge-base-file-pre.oss-cn-shanghai.aliyuncs.com/mmlongbench_pdf_files/",
        help="URL prefix to be prepended to filenames",
    )

    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8001",
        help="MemOS API address (default: http://127.0.0.1:8001)",
    )

    parser.add_argument("--workers", "-w", type=int, default=5, help="Concurrency (default: 5)")

    parser.add_argument(
        "--timeout", type=float, default=1200, help="Request timeout in seconds (default: 120)"
    )

    parser.add_argument(
        "--mode", default="fine", choices=["fine", "fast"], help="Processing mode (default: fine)"
    )

    parser.add_argument(
        "--async-mode", default="sync", choices=["sync", "async"], help="Async mode (default: sync)"
    )

    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Limit number of samples to process (for testing, default all)",
    )

    parser.add_argument("--version-dir", "-v", default=None, help="Version directory name")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("MMLongbench-doc Product Add Concurrent Tool")
    print("=" * 60)

    # Read filename list
    print(f"\nReading filename list: {args.filenames_file}")
    try:
        filenames = read_filenames(args.filenames_file)
        print(f"Read {len(filenames)} filenames")
        if len(filenames) == 0:
            print("Error: Filename list is empty!")
            return

        # Show first few filenames
        print("First 5 filenames:")
        for fn in filenames[:5]:
            print(f"  - {fn}")
        if len(filenames) > 5:
            print(f"  ... and {len(filenames) - 5} more files")

    except FileNotFoundError:
        print(f"Error: File not found {args.filenames_file}")
        return
    except Exception as e:
        print(f"Error: Failed to read file - {e}")
        return

    if args.limit is not None:
        filenames = filenames[: args.limit]

    # Determine output file path
    base_dir = "evaluation/results/mmlongbench"
    version_output_dir = os.path.join(base_dir, args.version_dir if args.version_dir else "default")
    os.makedirs(version_output_dir, exist_ok=True)
    output_path = os.path.join(version_output_dir, f"{args.lib}_add_results.json")

    existing_added: dict[str, str] = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, encoding="utf-8") as f:
                obj = json.load(f)
            added_obj = obj.get("added") if isinstance(obj, dict) else None
            if isinstance(added_obj, dict):
                existing_added = {str(k): str(v) for k, v in added_obj.items() if v is not None}
        except Exception:
            existing_added = {}

    if existing_added:
        before = len(filenames)
        filenames = [fn for fn in filenames if fn not in existing_added]
        print(
            f"[Resume] found {len(existing_added)} successful files in checkpoint, "
            f"skip {before - len(filenames)} files, pending={len(filenames)}"
        )

    if not filenames:
        print("[Add] no pending files, nothing to do.")
        return

    config = {
        "filenames_file": args.filenames_file,
        "url_prefix": args.url_prefix,
        "api_url": args.api_url,
        "concurrency": args.workers,
        "mode": args.mode,
        "async_mode": args.async_mode,
        "version_dir": args.version_dir,
    }

    # Execute concurrent add
    result = run_concurrent_add(
        lib=args.lib,
        filenames=filenames,
        url_prefix=args.url_prefix,
        user_prefix=args.version_dir,
        workers=args.workers,
        mode=args.mode,
        async_mode=args.async_mode,
        output_path=output_path,
        existing_added=existing_added,
        config=config,
    )

    # Save results to file
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            # Remove non-serializable content
            output_data = {
                "summary": result.get("summary"),
                "total_duration": result.get("total_duration"),
                "config": {
                    "filenames_file": args.filenames_file,
                    "url_prefix": args.url_prefix,
                    "api_url": args.api_url,
                    "concurrency": args.workers,
                    "mode": args.mode,
                    "async_mode": args.async_mode,
                    "version_dir": args.version_dir,
                },
            }
            added = result.get("added") or {}
            merged_added: dict[str, str] = {}
            merged_added.update(existing_added)
            if isinstance(added, dict) and added:
                for k, v in added.items():
                    if v is None:
                        continue
                    merged_added[str(k)] = str(v)
            if merged_added:
                output_data["added"] = dict(sorted(merged_added.items()))
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
