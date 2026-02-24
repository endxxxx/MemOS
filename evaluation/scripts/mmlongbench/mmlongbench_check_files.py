import argparse
import json
import os

from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from evaluation.scripts.utils.client import MemosApiOnlineClient, get_lib_client


load_dotenv()
memos_knowledgebase_id = os.getenv("MEMOS_KNOWLEDGEBASE_ID_MM_LONGBENCH")
dify_dataset_id = os.getenv("DIFY_DATASET_ID_MM_LONGBENCH")


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


def _check_file_status(
    client: MemosApiOnlineClient, lib: str, dataset_id: str, added_ids: dict, batch_size: int
) -> dict[str, dict[str, str | None]]:
    file_status: dict[str, dict[str, str | None]] = {}

    if lib == "dify":
        # Dify implementation using get_dataset_documents
        page = 1
        limit = 20  # Adjust limit as needed
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
                        "id": item.get("id"),  # Keep Dify ID for reference if needed
                    }
            except Exception as e:
                print(f"[Check] error fetching dify documents page {page}: {e}")
                break

        return file_status

    file_ids = sorted({fid for fid in added_ids.values() if fid})
    print(f"[Check] total file ids: {len(file_ids)}")
    if not file_ids:
        return {}

    for i in tqdm(range(0, len(file_ids), batch_size), desc="Checking files"):
        batch = file_ids[i : i + batch_size]
        try:
            resp = client.check_file(dataset_id, batch)
        except Exception as e:
            print(f"[Check] error for batch starting at {i}: {e}")
            continue

        if lib == "memos-api-online":
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
    fid_to_filename: dict[str, str] = {}

    if lib != "dify":
        for filename, fid in added_ids.items():
            if fid:
                fid_to_filename[str(fid)] = str(filename)

    reupload_results: list[dict[str, str | None]] = []

    if lib == "dify":
        # For dify, keys in file_status are filenames
        failed_files = [
            filename for filename, info in file_status.items() if (info.get("status") == "error")
        ]
    else:
        # For memos, keys in file_status are file_ids
        failed_ids = [
            fid for fid, info in file_status.items() if (info.get("status") == "PROCESSING")
        ]
        failed_files = []  # Not used for memos logic below which iterates failed_ids

    iterator = failed_files if lib == "dify" else failed_ids

    for item in tqdm(iterator, desc="Reuploading failed files"):
        if lib == "dify":
            filename = item  # item is filename
            fid = file_status[filename].get(
                "id"
            )  # get the dify doc id if needed, though we reupload by file path
        else:
            fid = item  # item is fid
            filename = fid_to_filename.get(fid)

        if not filename:
            reupload_results.append(
                {
                    "old_file_id": fid,
                    "filename": None,
                    "new_file_id": None,
                    "ok": "false",
                    "error": "filename_not_found",
                }
            )
            continue

        if lib == "dify":
            file_url = os.path.abspath(
                os.path.join("evaluation/data/mmlongbench/documents", filename)
            )
        else:
            file_url = f"{url_prefix.rstrip('/')}/{filename}"

        try:
            if lib == "dify":
                resp = client.upload_file(dify_dataset_id or "", file_url)
                new_id = resp.get("batch")
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
                    "old_file_id": fid,
                    "filename": filename,
                    "new_file_id": new_id,
                    "ok": "true",
                    "error": None,
                }
            )
        except Exception as e:
            reupload_results.append(
                {
                    "old_file_id": fid,
                    "filename": filename,
                    "new_file_id": None,
                    "ok": "false",
                    "error": str(e),
                }
            )

    return reupload_results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Check MMLongbench memos-api-online file status and reupload failed files."
    )
    parser.add_argument("--lib", type=str, default="memos-api-online")
    parser.add_argument("--version-dir", "-v", default=None)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument(
        "--url-prefix",
        "-u",
        default="https://memos-knowledge-base-file-pre.oss-cn-shanghai.aliyuncs.com/mmlongbench_pdf_files/",
    )
    args = parser.parse_args(argv)

    version = args.version_dir if args.version_dir else "default"
    base_dir = Path(f"evaluation/results/mmlongbench/{version}")
    output_dir = base_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records_path = output_dir / f"{args.lib}_add_results.json"
    print(f"[Check] loading records from {records_path}")

    added_ids = _load_added_ids(records_path)
    file_ids = sorted({fid for fid in added_ids.values() if fid})

    client = get_lib_client(args.lib)
    batch_size = max(1, args.batch_size)

    file_status = {}
    if args.lib == "dify":
        file_status = _check_file_status(client, args.lib, dify_dataset_id, added_ids, 1)
    elif args.lib == "memos-api-online":
        file_status = _check_file_status(
            client, args.lib, memos_knowledgebase_id, added_ids, batch_size
        )

    reupload_results = _reupload_failed_files(
        client, file_status, added_ids, args.url_prefix, args.lib
    )
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
                if item.get("ok") == "true" and item.get("filename") and item.get("new_file_id"):
                    added_obj[str(item["filename"])] = str(item["new_file_id"])
            obj["added"] = dict(sorted(added_obj.items()))
            tmp_r = records_path.with_suffix(records_path.suffix + ".tmp")
            tmp_r.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp_r, records_path)
            print(f"[Update] updated add_results with new file ids -> {records_path}")
        except Exception as e:
            print(f"[Update] failed to update add_results: {e}")

    output_path = output_dir / f"{args.lib}_file_status.json"

    if args.lib == "dify":
        file_detail_list = []
        for filename in sorted(added_ids.keys()):
            info = file_status.get(filename) or {}
            detail = {"filename": filename, "batch_id": added_ids.get(filename), **info}
            file_detail_list.append(detail)
    else:
        file_detail_list = [{"id": fid, **(file_status.get(fid) or {})} for fid in file_ids]

    result_obj = {
        "lib": args.lib,
        "version_dir": args.version_dir,
        "total": len(file_detail_list),
        "file_detail_list": file_detail_list,
        "reupload_results": reupload_results,
    }
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(json.dumps(result_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, output_path)
    print(f"[Check] saved file status for {len(file_status)} files to {output_path}")


if __name__ == "__main__":
    main()
