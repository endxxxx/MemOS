import argparse
import json
import os

from contextlib import ExitStack


MAX_MB_DEFAULT = 4.5
FILES_COUNT_DEFAULT = 300


def read_samples(path: str) -> list[tuple[str, bytes]]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            _id = str(obj.get("_id") or obj.get("id") or obj.get("sample_id") or "")
            ctx = obj.get("context")
            if ctx is None:
                ctx = obj.get("document")
            if ctx is None:
                docs = obj.get("documents")
                if isinstance(docs, list):
                    ctx = "\n\n".join([str(x) for x in docs])
            if ctx is None:
                ctx = ""
            b = ctx.encode("utf-8")
            items.append((_id or str(len(items)), b))
    return items


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def pack(items: list[tuple[str, bytes]], out_dir: str, files_count: int, max_bytes: int):
    ensure_dir(out_dir)
    files = []
    for i in range(files_count):
        name = os.path.join(out_dir, f"doc_{i + 1:03d}.txt")
        files.append({"path": name, "size": 0})
    with ExitStack() as stack:
        handles = {f["path"]: stack.enter_context(open(f["path"], "wb")) for f in files}
        index = {}
        cursor = 0
        sep_template = b"\n\n==== ID: %s ====\n\n"
        for _id, ctx_bytes in items:
            sep = sep_template % _id.encode("utf-8")
            overhead = len(sep)
            if len(ctx_bytes) + overhead <= max_bytes:
                placed = False
                for j in range(files_count):
                    f = files[(cursor + j) % files_count]
                    if f["size"] + overhead + len(ctx_bytes) <= max_bytes:
                        h = handles[f["path"]]
                        h.write(sep)
                        h.write(ctx_bytes)
                        f["size"] += overhead + len(ctx_bytes)
                        index.setdefault(_id, set()).add(os.path.basename(f["path"]))
                        cursor = (cursor + 1) % files_count
                        placed = True
                        break
                if not placed:
                    raise RuntimeError("容量不足，无法在限定的文件数与大小内完成合并")
            else:
                remaining = ctx_bytes
                while remaining:
                    placed = False
                    for j in range(files_count):
                        f = files[(cursor + j) % files_count]
                        file_space = max_bytes - f["size"] - overhead
                        if file_space <= 0:
                            continue
                        chunk = remaining[:file_space]
                        try:
                            chunk_decoded = chunk.decode("utf-8")
                            chunk_bytes = chunk_decoded.encode("utf-8")
                        except UnicodeDecodeError:
                            chunk_decoded = chunk.decode("utf-8", errors="ignore")
                            chunk_bytes = chunk_decoded.encode("utf-8")
                        if not chunk_bytes:
                            continue
                        h = handles[f["path"]]
                        h.write(sep)
                        h.write(chunk_bytes)
                        f["size"] += overhead + len(chunk_bytes)
                        index.setdefault(_id, set()).add(os.path.basename(f["path"]))
                        cursor = (cursor + 1) % files_count
                        remaining = remaining[len(chunk_bytes) :]
                        placed = True
                        break
                    if not placed:
                        raise RuntimeError("容量不足，无法在限定的文件数与大小内完成合并")
    index_path = os.path.join(out_dir, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({k: sorted(v) for k, v in index.items()}, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="/Users/tianxingshi/Desktop/projects/MemOS/evaluation/data/longbench_v2/longbenchv2_train.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/Users/tianxingshi/Desktop/projects/MemOS/evaluation/data/longbench_v2/merged_txt",
    )
    parser.add_argument("--files_count", type=int, default=FILES_COUNT_DEFAULT)
    parser.add_argument("--max_mb", type=float, default=MAX_MB_DEFAULT)
    args = parser.parse_args()
    items = read_samples(args.input)
    max_bytes = int(args.max_mb * 1024 * 1024)
    pack(items, args.output_dir, args.files_count, max_bytes)


if __name__ == "__main__":
    main()
