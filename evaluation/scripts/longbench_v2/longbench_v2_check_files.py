import argparse
import json
import os

from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from evaluation.scripts.utils.client import MemosApiOnlineClient, get_lib_client


load_dotenv()
# Knowledgebase ID used when re-uploading LongBench-v2 files
memos_knowledgebase_id = os.getenv("MEMOS_KNOWLEDGEBASE_ID_LONGBENCH_V2")
dify_dataset_id = os.getenv("DIFY_DATASET_ID_LONGBENCH_V2")


def _load_added_ids(records_path: Path) -> dict[str, str | None]:
    """
    Load mapping from sample_id (version-prefixed user id) to file_id from add_results.json.
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


def _check_file_status(
    client: MemosApiOnlineClient, lib: str, dataset_id: str, added_ids: dict, batch_size: int
) -> dict[str, dict[str, str | None]]:
    """
    Phase 1: Query file processing status for all given file_ids in batches.
    Returns file_id -> {name, size, status}.
    """
    file_status: dict[str, dict[str, str | None]] = {}

    if lib == "dify":
        page = 1
        limit = 20
        has_more = True

        while has_more:
            try:
                resp = client.get_dataset_documents(dataset_id, page=page, limit=limit)
                data = resp.get("data", [])
                has_more = resp.get("has_more", False)
                page += 1

                for item in data:
                    file_name = item.get("name")
                    if not file_name:
                        continue

                    file_status[file_name] = {
                        "name": file_name,
                        "status": item.get("indexing_status"),
                        "error": item.get("error"),
                        "id": item.get("id"),
                    }
            except Exception as e:
                print(f"[Check] error fetching dify documents page {page}: {e}")
                break
        return file_status

    file_ids = sorted({fid for fid in added_ids.values() if fid})
    for i in tqdm(range(0, len(file_ids), batch_size), desc="Checking files"):
        batch = file_ids[i : i + batch_size]
        try:
            resp = client.check_file(dataset_id, batch)
        except Exception as e:
            print(f"[Check] error for batch starting at {i}: {e}")
            continue
        if not isinstance(resp, dict):
            continue
        data = resp.get("data") or {}
        details = data.get("file_detail_list") or []
        for item in details:
            if not isinstance(item, dict):
                continue
            fid = item.get("id")
            if not fid:
                continue
            file_status[str(fid)] = {
                "name": item.get("name"),
                "size": item.get("size"),
                "status": item.get("status"),
            }
    return file_status


def _reupload_failed_files(
    client: MemosApiOnlineClient,
    file_status: dict[str, dict[str, str | None]],
    added_ids: dict[str, str | None],
    url_prefix: str,
    lib: str,
) -> list[dict[str, str | None]]:
    """
    Phase 2: Re-upload files whose status == PROCESSING_FAILED (or 'error' for Dify).
    LongBench-v2 user_id is '<version>_<sample_id>', so the file URL uses the last segment.
    Returns a list of per-file reupload results for auditing.
    """
    fid_to_user: dict[str, str] = {}
    if lib != "dify":
        for uid, fid in added_ids.items():
            if fid:
                fid_to_user[str(fid)] = str(uid)

    reupload_results: list[dict[str, str | None]] = []

    if lib == "dify":
        failed_files = [
            filename for filename, info in file_status.items() if (info.get("status") == "error")
        ]
    else:
        failed_files = [
            fid for fid, info in file_status.items() if (info.get("status") == "PROCESSING_FAILED")
        ]

    for item in tqdm(failed_files, desc="Reuploading failed files"):
        if lib == "dify":
            filename = item
            # sample_id is filename without extension (assuming .txt)
            # Actually we just need the local path
            file_url = os.path.abspath(
                os.path.join("evaluation/data/longbench_v2/documents", filename)
            )

            # For reporting
            uid = filename  # using filename as uid for reporting context
            fid = file_status[filename].get("id")

            if not os.path.exists(file_url):
                reupload_results.append(
                    {
                        "old_file_id": fid,
                        "user_id": uid,
                        "new_file_id": None,
                        "ok": "false",
                        "error": "local_file_not_found",
                    }
                )
                continue
        else:
            fid = item
            uid = fid_to_user.get(fid)
            if not uid:
                reupload_results.append(
                    {
                        "old_file_id": fid,
                        "user_id": None,
                        "new_file_id": None,
                        "ok": "false",
                        "error": "user_id_not_found",
                    }
                )
                continue
            file_url = f"{url_prefix.rstrip('/')}/{uid.split('_')[-1]}.txt"

        try:
            if lib == "dify":
                resp = client.upload_file(dify_dataset_id or "", file_url, mime_type="text/plain")
                new_id = resp.get("batch")  # Dify returns batch id
            else:
                resp = client.upload_file(memos_knowledgebase_id or "", file_url)
                new_id = None
                if isinstance(resp, dict):
                    data = resp.get("data") or {}
                    if isinstance(data, list) and data:
                        first = data[0] if isinstance(data[0], dict) else {}
                        new_id = str(first.get("id")) if first.get("id") else None

            reupload_results.append(
                {
                    "old_file_id": fid
                    if lib != "dify"
                    else filename,  # Use filename as identifier for Dify old_id context
                    "user_id": uid,
                    "new_file_id": new_id,
                    "ok": "true",
                    "error": None,
                }
            )
        except Exception as e:
            reupload_results.append(
                {
                    "old_file_id": fid if lib != "dify" else filename,
                    "user_id": uid,
                    "new_file_id": None,
                    "ok": "false",
                    "error": str(e),
                }
            )
    return reupload_results


def main(argv: list[str] | None = None) -> None:
    """
    Orchestrate file status checking and failed-file reupload for LongBench-v2 memos-api-online runs.
    """
    parser = argparse.ArgumentParser(
        description="Check LongBench-v2 memos-api-online file status and reupload failed."
    )
    parser.add_argument("--lib", type=str, default="memos-api-online")
    parser.add_argument("--version-dir", "-v", default=None)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument(
        "--url-prefix",
        "-u",
        default="https://memos-knowledge-base-file-pre.oss-cn-shanghai.aliyuncs.com/longbench_v2_text_files/",
    )
    args = parser.parse_args(argv)

    if args.lib != "memos-api-online" and args.lib != "dify":
        print(f"Only memos-api-online and dify are supported, got lib={args.lib}")
        return

    version = args.version_dir if args.version_dir else "default"
    base_dir = Path(f"evaluation/results/longbench_v2/{version}")
    output_dir = base_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records_path = output_dir / f"{args.lib}_add_results.json"
    print(f"[Check] loading records from {records_path}")

    added_ids = _load_added_ids(records_path)
    file_ids = sorted({fid for fid in added_ids.values() if fid})

    if args.lib != "dify":
        print(f"[Check] total file ids: {len(file_ids)}")
        if not file_ids:
            return

    client = get_lib_client(args.lib)
    batch_size = max(1, args.batch_size)

    file_status = {}
    if args.lib == "dify":
        file_status = _check_file_status(client, args.lib, dify_dataset_id, added_ids, 1)
    else:
        file_status = _check_file_status(
            client, args.lib, memos_knowledgebase_id, added_ids, batch_size
        )
    reupload_results = _reupload_failed_files(
        client, file_status, added_ids, args.url_prefix, args.lib
    )

    # Update added records with new file ids
    if reupload_results:
        try:
            obj: dict = {}
            if records_path.exists():
                txt = records_path.read_text(encoding="utf-8")
                if txt:
                    parsed = json.loads(txt)
                    if isinstance(parsed, dict):
                        obj = parsed
            added_obj: dict[str, str | None] = {}
            if isinstance(obj.get("added"), dict):
                added_obj = {
                    str(k): (str(v) if v is not None else None) for k, v in obj["added"].items()
                }
            else:
                added_obj = dict(added_ids)
            for item in reupload_results:
                if item.get("ok") == "true" and item.get("user_id") and item.get("new_file_id"):
                    added_obj[str(item["user_id"])] = str(item["new_file_id"])
            obj["added"] = dict(sorted(added_obj.items()))
            tmp_r = records_path.with_suffix(records_path.suffix + ".tmp")
            tmp_r.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp_r, records_path)
            print(f"[Update] updated add_results with new file ids -> {records_path}")
        except Exception as e:
            print(f"[Update] failed to update add_results: {e}")

    output_path = output_dir / f"{args.lib}_file_status.json"

    if args.lib == "dify":
        to_save = {
            "lib": args.lib,
            "version_dir": args.version_dir,
            "total": len(file_status),
            "file_detail_list": file_status,
            "reupload_results": reupload_results,
        }
    else:
        file_detail_list = [{"id": fid, **(file_status.get(fid) or {})} for fid in file_ids]
        to_save = {
            "lib": args.lib,
            "version_dir": args.version_dir,
            "total": len(file_detail_list),
            "file_detail_list": file_detail_list,
            "reupload_results": reupload_results,
        }

    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(json.dumps(to_save, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, output_path)
    print(f"[Check] saved file status for {len(file_status)} files to {output_path}")


if __name__ == "__main__":
    main()
