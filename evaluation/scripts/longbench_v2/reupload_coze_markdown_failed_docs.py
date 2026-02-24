import base64
import os

from pathlib import Path

from dotenv import load_dotenv

from evaluation.scripts.longbench_v2.longbench_v2_ingestion import _load_added_ids, _save_added_ids
from evaluation.scripts.utils.client import get_lib_client


def main() -> None:
    load_dotenv()
    dataset_id = os.getenv("COZE_DATASET_ID_LONGBENCH_V2")
    client = get_lib_client("coze")

    md_dir = Path("evaluation/data/longbench_v2/md_files")

    records_path = Path(
        "evaluation/results/longbench_v2/coze_longbench_v2_0211/coze_add_results.json"
    )
    existing_map = _load_added_ids(records_path)

    names = sorted([n for n in os.listdir(md_dir) if n.endswith(".md")])
    for i in range(0, len(names), 5):
        batch_names = names[i : i + 5]
        batch_added = {}
        for md_name in batch_names:
            if isinstance(existing_map, dict) and existing_map.get(md_name):
                continue
            md_path = md_dir / md_name
            with open(md_path, "rb") as f:
                file_b64 = base64.b64encode(f.read()).decode("utf-8")
            document_bases = [
                {
                    "source_info": {
                        "file_base64": file_b64,
                        "file_type": "md",
                        "document_source": 0,
                    },
                    "name": md_name,
                }
            ]

            print(md_name)
            result = client.create_document(
                dataset_id=dataset_id,
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
                batch_added[md_name] = rid
        existing_map2 = _load_added_ids(records_path)
        existing_map2.update(batch_added)
        _save_added_ids(records_path, existing_map2)


if __name__ == "__main__":
    main()
