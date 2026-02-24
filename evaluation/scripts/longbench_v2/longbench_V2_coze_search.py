import argparse
import json
import os
import time
import traceback

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from dotenv import load_dotenv
from tqdm import tqdm


load_dotenv()


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


def _extract_coze_answer(resp_json: dict) -> str:
    if not isinstance(resp_json, dict):
        return ""
    candidates: list[str] = []
    if "data" in resp_json and isinstance(resp_json["data"], dict):
        data = resp_json["data"]
        if isinstance(data.get("messages"), list):
            for m in data.get("messages", []):
                if isinstance(m, dict):
                    c = m.get("content")
                    if isinstance(c, str) and c.strip():
                        candidates.append(c.strip())
    if isinstance(resp_json.get("messages"), list):
        for m in resp_json.get("messages", []):
            if isinstance(m, dict):
                c = m.get("content")
                if isinstance(c, str) and c.strip():
                    candidates.append(c.strip())
    if isinstance(resp_json.get("output_text"), str) and resp_json.get("output_text").strip():
        candidates.append(resp_json["output_text"].strip())
    for k in ("result", "text", "answer"):
        v = resp_json.get(k)
        if isinstance(v, str) and v.strip():
            candidates.append(v.strip())
    return candidates[-1] if candidates else ""


def _parse_sse_log_file(log_path: Path) -> list[dict]:
    try:
        content = log_path.read_text(encoding="utf-8")
    except Exception:
        return []
    events: list[tuple[str, str]] = []
    event_name = None
    data_lines: list[str] = []
    for raw in content.splitlines():
        line = raw.strip("\r")
        if not line:
            if event_name is not None:
                events.append((event_name, "\n".join(data_lines)))
            event_name, data_lines = None, []
            continue
        if line.startswith("event:"):
            event_name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
    if event_name is not None:
        events.append((event_name, "\n".join(data_lines)))
    chosen = None
    for ev, data_str in reversed(events):
        if ev.endswith("message.completed"):
            chosen = (ev, data_str)
            break
    if chosen is None:
        for ev, data_str in reversed(events):
            if ev.endswith("message.delta"):
                chosen = (ev, data_str)
                break
    if chosen is None:
        return []
    _, data_str = chosen
    try:
        data_obj = json.loads(data_str)
    except Exception:
        return []
    chunk = data_obj.get("content")
    if isinstance(chunk, dict):
        obj = chunk
    elif isinstance(chunk, str):
        try:
            obj = json.loads(chunk)
        except Exception:
            return []
    else:
        return []
    outputs = obj.get("output", [])
    items: list[dict] = []
    if isinstance(outputs, list):
        for item in outputs:
            if isinstance(item, dict):
                text = item.get("output")
                if isinstance(text, str):
                    doc_id = item.get("documentId")
                    items.append(
                        {"document_id": doc_id if isinstance(doc_id, str) else None, "output": text}
                    )
    return items


def coze_workflow_chat(
    base_url: str,
    api_key: str,
    workflow_id: str,
    question: str,
    timeout: float = 60.0,
    sample_id: str | None = None,
    debug_dir: Path | None = None,
) -> tuple[list[str], list[dict]]:
    url = f"{base_url.rstrip('/')}/v1/workflows/chat"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    payload = {
        "workflow_id": workflow_id,
        "parameters": {
            "CONVERSATION_NAME": "Default",
            "USER_INPUT": "",
            "input": question,
        },
        "additional_messages": [
            {
                "content_type": "text",
                "role": "user",
                "type": "question",
                "content": question,
            }
        ],
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout, stream=True)
    resp.raise_for_status()
    ctype = resp.headers.get("content-type", "")
    if "text/event-stream" not in ctype:
        try:
            body = resp.json()
        except Exception:
            body = {"text": resp.text}
        if isinstance(body, dict) and body.get("code") not in (None, 0, 200):
            msg = body.get("msg") or body
            raise RuntimeError(f"COZE ERROR {body.get('code')}: {msg}")
        ans = _extract_coze_answer(body)
        texts = [ans] if ans else []
        raw_items = [{"document_id": None, "output": ans}] if ans else []
        return texts, raw_items

    def iter_sse_events(r):
        event_name = None
        data_lines = []
        for raw in r.iter_lines(decode_unicode=True):
            if raw is None:
                continue
            line = raw.strip("\r")
            if not line:
                if event_name is not None:
                    yield event_name, "\n".join(data_lines)
                event_name, data_lines = None, []
                continue
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
        if event_name is not None:
            yield event_name, "\n".join(data_lines)

    last_completed_outputs: list | None = None
    last_delta_outputs: list | None = None
    content_buf = ""
    raw_deltas: list[str] = []

    for event, data_str in iter_sse_events(resp):
        if not data_str:
            continue
        if event.endswith("message.delta"):
            block_lines = ["event: " + event]
            for ln in data_str.splitlines():
                block_lines.append("data: " + ln)
            block_lines.append("")  # blank line to separate SSE events
            raw_deltas.append("\n".join(block_lines))
        try:
            data_obj = json.loads(data_str)
        except Exception:
            continue
        if event.endswith(("message.delta", "message.completed")):
            chunk = data_obj.get("content")
            if isinstance(chunk, dict):
                obj = chunk
            elif isinstance(chunk, str):
                content_buf += chunk
                try:
                    obj = json.loads(content_buf)
                    content_buf = ""
                except Exception:
                    continue
            else:
                obj = None
            if isinstance(obj, dict):
                outputs = obj.get("output", [])
                if isinstance(outputs, list):
                    if event.endswith("message.completed"):
                        last_completed_outputs = outputs
                    elif event.endswith("message.delta"):
                        last_delta_outputs = outputs

    if debug_dir and sample_id:
        try:
            debug_dir.mkdir(parents=True, exist_ok=True)
            tmp = (debug_dir / f"sse_{sample_id}.log").with_suffix(".log.tmp")
            tmp.write_text("\n".join(raw_deltas), encoding="utf-8")
            os.replace(tmp, debug_dir / f"sse_{sample_id}.log")
        except Exception:
            pass

    if debug_dir and sample_id:
        log_path = debug_dir / f"sse_{sample_id}.log"
        parsed_items = _parse_sse_log_file(log_path)
        if parsed_items:
            texts = [it["output"] for it in parsed_items]
            return texts, parsed_items

    final_outputs = (
        last_completed_outputs if last_completed_outputs is not None else (last_delta_outputs or [])
    )
    if not final_outputs and raw_deltas:
        for s in reversed(raw_deltas):
            try:
                data_lines = []
                for ln in s.splitlines():
                    if ln.startswith("data: "):
                        data_lines.append(ln[len("data: ") :])
                data_str = "\n".join(data_lines)
                data_obj = json.loads(data_str)
                chunk = data_obj.get("content")
                if isinstance(chunk, dict):
                    obj = chunk
                elif isinstance(chunk, str):
                    obj = json.loads(chunk)
                else:
                    obj = None
                if (
                    isinstance(obj, dict)
                    and isinstance(obj.get("output"), list)
                    and obj.get("output")
                ):
                    final_outputs = obj.get("output")
                    break
            except Exception:
                continue
    raw_items = []
    for item in final_outputs:
        if isinstance(item, dict):
            text = item.get("output")
            if isinstance(text, str):
                doc_id = item.get("documentId")
                raw_items.append(
                    {"document_id": doc_id if isinstance(doc_id, str) else None, "output": text}
                )
    texts = [it["output"] for it in raw_items]
    return texts, raw_items


def search_one(sample: dict, version_dir: str, cfg: dict) -> dict:
    sample_id = str(sample.get("_id"))
    question = sample.get("question") or ""
    choices = {
        "A": sample.get("choice_A") or "",
        "B": sample.get("choice_B") or "",
        "C": sample.get("choice_C") or "",
        "D": sample.get("choice_D") or "",
    }
    texts, raw_items = coze_workflow_chat(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        workflow_id=cfg["workflow_id"],
        question=str(question),
        timeout=cfg["timeout"],
        sample_id=sample_id,
        debug_dir=cfg.get("debug_dir"),
    )
    return {
        "_id": sample_id,
        "domain": sample.get("domain"),
        "sub_domain": sample.get("sub_domain"),
        "difficulty": sample.get("difficulty"),
        "length": sample.get("length"),
        "question": question,
        "choices": choices,
        "answer": sample.get("answer"),
        "memories_used": texts,
        "coze_doc_outputs": raw_items,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Longbench-v2 Coze Workflow Search Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset-path", "-s", default=None, help="Ignored")
    parser.add_argument("--workers", "-c", type=int, default=None, help="Ignored")
    parser.add_argument("--version-dir", "-v", default=None, help="Version directory name")
    parser.add_argument(
        "--limit", "-l", type=int, default=None, help="Limit number of samples to process"
    )
    parser.add_argument(
        "--coze-base-url", default=os.getenv("COZE_BASE_URL", "https://api.coze.cn")
    )
    parser.add_argument("--coze-api-key", default=None)
    parser.add_argument("--coze-workflow-id", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coze_api_key = os.getenv("COZE_API_KEY") or args.coze_api_key or ""
    coze_workflow_id = os.getenv("COZE_WORKFLOW_ID_LONGBENCH_V2") or args.coze_workflow_id or ""
    if not coze_api_key:
        raise RuntimeError("COZE_API_KEY is required in environment")
    if not coze_workflow_id:
        raise RuntimeError("COZE_WORKFLOW_ID_LONGBENCH_V2 is required in environment")

    dataset_path = Path("evaluation/data/longbench_v2/longbenchv2_train.json")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    dataset = _load_dataset_jsonl(dataset_path)
    if args.limit is not None:
        dataset = dataset[: args.limit]

    version = args.version_dir or "default"
    output_dir = Path("evaluation/results/longbench_v2") / version
    os.makedirs(output_dir, exist_ok=True)
    output_filename = "coze_search_results.json"
    output_path = output_dir / output_filename

    results, processed_ids = _load_existing_results(output_path)
    pending = [s for s in dataset if str(s.get("_id")) not in processed_ids]
    if not pending:
        return

    start_time = time.time()
    cfg = {
        "base_url": args.coze_base_url,
        "api_key": coze_api_key,
        "workflow_id": coze_workflow_id,
        "timeout": 60.0,
        "debug_dir": output_dir / "sse",
    }

    workers = 8
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(search_one, sample, args.version_dir, cfg) for sample in pending]
        for idx, f in enumerate(
            tqdm(as_completed(futures), total=len(futures), desc="Coze Searching"), start=1
        ):
            try:
                r = f.result()
                results.append(r)
                if idx % 20 == 0:
                    _save_json_list(output_path, results)
            except Exception as e:
                print(f"[Coze Search] Error: {e}")
                traceback.print_exc()

    _save_json_list(output_path, results)
    total_duration = time.time() - start_time
    combined_obj = {
        "perf": {
            "summary": {
                "counts": {"success": len(results), "failed": 0},
                "stats": None,
                "errors": {},
            },
            "total_duration": total_duration,
            "config": {
                "workers": workers,
                "dataset_path": str(dataset_path),
                "limit": args.limit,
            },
        },
        "results": results,
    }
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(json.dumps(combined_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, output_path)


if __name__ == "__main__":
    main()
