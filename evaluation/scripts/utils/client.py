import json
import os
import re
import sys
import time
import uuid

from contextlib import suppress
from datetime import datetime

import requests

from dotenv import load_dotenv


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()


def with_retry(retries=2, delay=2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < retries - 1:
                        func_name = getattr(func, "__name__", "Operation")
                        print(f"[Retry] {func_name} failed: {e}. Retrying in {current_delay}s...")
                        time.sleep(current_delay)
                        current_delay *= 2
                    else:
                        raise e

        return wrapper

    return decorator


def get_lib_client(lib: str):
    if lib == "mem0":
        return Mem0Client(enable_graph=False)
    if lib == "supermemory":
        return SupermemoryClient()
    if lib == "memos-api-online":
        return MemosApiOnlineClient()
    if lib == "fastgpt":
        return FastGPTClient()
    if lib == "memos":
        return MemosApiClient()
    if lib == "dify":
        return DifyClient()
    if lib == "coze":
        return CozeClient()
    raise ValueError(f"Unknown library: {lib}")


class ZepClient:
    def __init__(self):
        from zep_cloud.client import Zep

        api_key = os.getenv("ZEP_API_KEY")
        self.client = Zep(api_key=api_key)

    @with_retry()
    def add(self, messages, user_id, timestamp):
        iso_date = datetime.fromtimestamp(timestamp).isoformat()
        for msg in messages:
            self.client.graph.add(
                data=msg.get("role") + ": " + msg.get("content"),
                type="message",
                created_at=iso_date,
                group_id=user_id,
            )

    @with_retry()
    def search(self, query, user_id, top_k):
        search_results = (
            self.client.graph.search(
                query=query, group_id=user_id, scope="nodes", reranker="rrf", limit=top_k
            ),
            self.client.graph.search(
                query=query, group_id=user_id, scope="edges", reranker="cross_encoder", limit=top_k
            ),
        )

        nodes = search_results[0].nodes
        edges = search_results[1].edges
        return nodes, edges


class Mem0Client:
    def __init__(self, enable_graph=False):
        from mem0 import MemoryClient

        self.client = MemoryClient(api_key=os.getenv("MEM0_API_KEY"))
        self.enable_graph = enable_graph

    @with_retry()
    def add(self, messages, user_id, timestamp, batch_size=2):
        for i in range(0, len(messages), batch_size):
            batch_messages = messages[i : i + batch_size]
            if self.enable_graph:
                self.client.add(
                    messages=batch_messages,
                    timestamp=timestamp,
                    user_id=user_id,
                    enable_graph=True,
                )
            else:
                self.client.add(
                    messages=batch_messages,
                    timestamp=timestamp,
                    user_id=user_id,
                    infer=False,
                )

    @with_retry()
    def search(self, query, user_id, top_k):
        res = self.client.search(
            query=query,
            top_k=top_k,
            user_id=user_id,
            enable_graph=self.enable_graph,
            filters={"AND": [{"user_id": f"{user_id}"}]},
        )
        return res


class MemobaseClient:
    def __init__(self):
        from memobase import MemoBaseClient

        self.client = MemoBaseClient(
            project_url=os.getenv("MEMOBASE_PROJECT_URL"), api_key=os.getenv("MEMOBASE_API_KEY")
        )

    @with_retry()
    def add(self, messages, user_id, batch_size=2):
        """
        messages = [{"role": "assistant", "content": data, "created_at": iso_date}]
        """
        from memobase import ChatBlob

        real_uid = self.string_to_uuid(user_id)
        user = self.client.get_or_create_user(real_uid)
        for i in range(0, len(messages), batch_size):
            batch_messages = messages[i : i + batch_size]
            _ = user.insert(ChatBlob(messages=batch_messages), sync=True)

    @with_retry()
    def search(self, query, user_id, top_k):
        real_uid = self.string_to_uuid(user_id)
        user = self.client.get_user(real_uid, no_get=True)
        memories = user.context(
            max_token_size=top_k * 100,
            chats=[{"role": "user", "content": query}],
            event_similarity_threshold=0.2,
            fill_window_with_events=True,
        )
        return memories

    @with_retry()
    def delete_user(self, user_id):
        from memobase.error import ServerError

        real_uid = self.string_to_uuid(user_id)
        with suppress(ServerError):
            self.client.delete_user(real_uid)

    def string_to_uuid(self, s: str, salt="memobase_client"):
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, s + salt))


class MemosApiClient:
    """Product Add API 封装"""

    def __init__(self, timeout: float = 600.0):
        self.base_url = os.getenv("MEMOS_URL")
        self.headers = {"Content-Type": "application/json"}
        self.timeout = timeout

    @with_retry()
    def add(
        self,
        messages,
        user_id,
        writable_cube_ids: list[str],
        source_type: str,
        mode: str,
        async_mode: str,
    ):
        """
        调用 /product/add 接口

        Args:
            messages: 添加记忆信息
            user_id: 用户ID
            writable_cube_ids: 可写cube ID列表
            source_type: 来源类型
            mode: 模式 (fine/coarse)
            async_mode: 异步模式 (sync/async)
        """
        url = f"{self.base_url}/product/add"

        payload = {
            "user_id": user_id,
            "writable_cube_ids": writable_cube_ids,
            "messages": messages,
            "info": {"source_type": source_type},
            "mode": mode,
            "async_mode": async_mode,
        }

        response = requests.post(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=self.headers,
            timeout=self.timeout,
        )

        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text}")

        body = response.json()
        if body.get("code") is not None and body.get("code") != 200:
            raise RuntimeError(f"BUSINESS ERROR {body.get('code')}: {response.text}")

        return body

    @with_retry()
    def search(self, query, user_id, readable_cube_ids: list[str], top_k: str, mode: str):
        """
        调用 /product/search 接口

        Args:
            query: 搜索查询
            user_id: 用户ID
            readable_cube_ids: 可读cube ID列表, 默认为[user_id]
            top_k: 返回结果数量
        """

        url = f"{self.base_url}/product/search"

        if readable_cube_ids is None:
            readable_cube_ids = [user_id]

        payload = {
            "query": query,
            "user_id": user_id,
            "readable_cube_ids": readable_cube_ids,
            "top_k": top_k,
            "mode": mode,
        }

        response = requests.post(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=self.headers,
            timeout=self.timeout,
        )

        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text}")

        return response.json()


class MemosApiOnlineClient:
    def __init__(self):
        self.memos_url = os.getenv("MEMOS_ONLINE_URL")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {os.environ['MEMOS_API_KEY']}",
        }

    @with_retry()
    def add(
        self,
        messages,
        user_id,
        conv_id,
        batch_size: int = 2,
        mode: str = "fine",
        async_mode: str = "async",
    ):
        url = f"{self.memos_url}/add/message"
        for i in range(0, len(messages), batch_size):
            batch_messages = messages[i : i + batch_size]
            payload = json.dumps(
                {
                    "user_id": user_id,
                    "conversation_id": conv_id,
                    "messages": batch_messages,
                }
            )
            resp = requests.request("POST", url, data=payload, headers=self.headers)
            resp.raise_for_status()

    @with_retry()
    def search(
        self,
        query: str,
        user_id: str,
        top_k: int,
        mode: str = "fast",
        knowledgebase_ids: list[str] | None = None,
    ):
        """Search memories."""
        url = f"{self.memos_url}/search/memory"
        data = {
            "query": query,
            "user_id": user_id,
            "memory_limit_number": top_k,
            "knowledgebase_ids": knowledgebase_ids,
            "mode": mode,
        }
        resp = requests.post(url, headers=self.headers, json=data, timeout=60)
        resp.raise_for_status()
        return resp.json()

    @with_retry()
    def upload_file(self, knowledgebase_id: str, file_url: str):
        """Upload file."""
        url = f"{self.memos_url}/add/knowledgebase-file"
        data = {
            "knowledgebase_id": knowledgebase_id,
            "file": [
                {
                    "content": file_url,
                }
            ],
        }
        resp = requests.post(url, headers=self.headers, json=data, timeout=60)
        resp.raise_for_status()
        return resp.json()

    @with_retry()
    def check_file(self, dataset_id: str, file_ids: list[str]):
        """Check file state."""
        url = f"{self.memos_url}/get/knowledgebase-file"
        data = {"file_ids": file_ids}
        resp = requests.post(url, headers=self.headers, json=data, timeout=60)
        resp.raise_for_status()
        return resp.json()


class SupermemoryClient:
    def __init__(self):
        from supermemory import Supermemory

        self.client = Supermemory(api_key=os.getenv("SUPERMEMORY_API_KEY"))

        self.api_key = os.getenv("SUPERMEMORY_API_KEY")
        if not self.api_key:
            raise ValueError(
                "SUPERMEMORY_API_KEY environment variable is not set. Please set it in your .env file or environment."
            )
        self.add_url = "https://api.supermemory.ai/v3/documents"
        self.search_url = "https://api.supermemory.ai/v3/search"

    def _sanitize_tag(self, s: str) -> str:
        t = str(s).strip()
        t = os.path.splitext(t)[0]
        t = t.replace(" ", "_")
        t = re.sub(r"[^A-Za-z0-9_-]", "_", t)
        t = re.sub(r"[_-]+", "_", t)
        t = t.strip("_")
        t = t.lower()
        if not re.match(r"^[a-z0-9]", t or ""):
            t = f"tag_{t}" if t else "tag_default"
        return t

    @with_retry()
    def add(
        self, content: str | None = None, user_id: str | None = None, messages: list | None = None
    ):
        if messages:
            content = "\n".join(
                f"{msg.get('chat_time', '')} {msg.get('role', '')}: {msg.get('content', '')}"
                for msg in messages
            )

            self.client.memories.add(content=content, container_tag=user_id)
            return

        payload = {
            "content": content,
            "raw": content,
            "containerTag": self._sanitize_tag(user_id),
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        resp = requests.post(self.add_url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()

    @with_retry()
    def search(self, query: str, user_id: str, top_k: int):
        payload = {
            "q": query,
            "limit": top_k,
            "containerTags": [self._sanitize_tag(user_id)],
            "rerank": True,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(self.search_url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        chunk_list = []
        res = [entry.get("chunks") for entry in data.get("results", [])]
        for chunks in res:
            for chunk in chunks:
                chunk_list.append(chunk["content"])

        return chunk_list


class MemuClient:
    def __init__(self):
        from memu import MemuClient

        self.memu_client = MemuClient(
            base_url="https://api.memu.so", api_key=os.getenv("MEMU_API_KEY")
        )
        self.agent_id = "assistant_001"

    def add(self, messages, user_id, iso_date):
        try:
            response = self.memu_client.memorize_conversation(
                conversation=messages,
                user_id=user_id,
                user_name=user_id,
                agent_id=self.agent_id,
                agent_name=self.agent_id,
                session_date=iso_date,
            )
            self.wait_for_completion(response.item_id)
        except Exception as error:
            print("❌ Error saving conversation:", error)

    def search(self, query, user_id, top_k):
        user_memories = self.memu_client.retrieve_related_memory_items(
            user_id=user_id, agent_id=self.agent_id, query=query, top_k=top_k, min_similarity=0.1
        )
        res = [m.memory.content for m in user_memories.related_memories]
        return res

    @with_retry()
    def wait_for_completion(self, task_id):
        while True:
            status = self.memu_client.get_task_status(task_id)
            if status.status in ["SUCCESS", "FAILURE", "REVOKED"]:
                break
            time.sleep(2)


class FastGPTClient:
    def __init__(self):
        self.base_url = os.getenv("FASTGPT_BASE_URL")
        self.api_key = os.getenv("FASTGPT_API_KEY")

    @with_retry()
    def create_dataset(self, dataset_name: str):
        url = f"{self.base_url}/core/dataset/create"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "name": dataset_name,
        }
        resp = requests.post(url, headers=headers, json=data, timeout=30)
        resp.raise_for_status()
        dataset_id = resp.json()["data"]
        return dataset_id

    @with_retry()
    def delete_dataset(self, dataset_id: str):
        url = f"{self.base_url}/core/dataset/delete?id={dataset_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = requests.delete(url, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()

    @with_retry()
    def add_content(self, dataset_id: str, content: str, collection_name: str):
        url = f"{self.base_url}/core/dataset/collection/create/text"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "text": content,
            "datasetId": dataset_id,
            "name": collection_name,
            "trainingType": "chunk",
            "chunkSettingMode": "auto",
        }
        resp = requests.post(url, headers=headers, json=data, timeout=60)
        resp.raise_for_status()
        return resp.json()

    @with_retry()
    def upload_file(self, dataset_id: str, file_url: str):
        url = f"{self.base_url}/proApi/core/dataset/collection/create/externalFileUrl"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "externalFileUrl": file_url,
            "externalFileId": file_url,
            "datasetId": dataset_id,
            "trainingType": "chunk",
            "chunkSize": 512,
        }
        resp = requests.post(url, headers=headers, json=data, timeout=60)
        resp.raise_for_status()
        return resp.json()

    @with_retry()
    def batch_add_content(self, collection_id: str, data: list[str]):
        url = f"{self.base_url}/core/dataset/data/pushData"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {"collectionId": collection_id, "data": [{"q": d} for d in data]}
        resp = requests.post(url, headers=headers, json=data, timeout=30)
        resp.raise_for_status()
        return resp.json()

    @with_retry()
    def search(self, dataset_id: str, query: str, top_k: int):
        url = f"{self.base_url}/core/dataset/searchTest"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {"datasetId": dataset_id, "text": query, "searchMode": "embedding"}
        resp = requests.post(url, headers=headers, json=data, timeout=30)

        resp.raise_for_status()
        result = resp.json()
        data_list = result["data"]["list"]
        return data_list

    @with_retry()
    def create_collection(self, dataset_id: str, collection_name: str):
        url = f"{self.base_url}/core/dataset/collection/create"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {"datasetId": dataset_id, "name": collection_name, "type": "virtual"}
        resp = requests.post(url, headers=headers, json=data, timeout=30)
        resp.raise_for_status()
        collection_id = resp.json()["data"]
        return collection_id


class DifyClient:
    def __init__(self):
        self.base_url = os.getenv("DIFY_BASE_URL")
        self.api_key = os.getenv("DIFY_API_KEY")

    @with_retry()
    def create_dataset(self, dataset_name: str, top_k: int):
        url = f"{self.base_url}/datasets"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "name": dataset_name,
            "indexing_technique": "high_quality",
            "permission": "only_me",
            "embedding_model_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "retrieval_model": {
                "search_method": "hybrid_search",
                "top_k": top_k,
                "reranking_enable": False,
                "score_threshold_enabled": False,
            },
        }
        resp = requests.post(url, headers=headers, json=data)
        resp.raise_for_status()
        dataset_id = resp.json()["id"]
        return dataset_id

    @with_retry()
    def delete_dataset(self, dataset_id: str):
        url = f"{self.base_url}/datasets/{dataset_id}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        resp = requests.delete(url, headers=headers)
        resp.raise_for_status()
        return resp.json()


class CozeClient:
    def __init__(self):
        self.base_url = os.getenv("COZE_BASE_URL") or "https://api.coze.cn"
        self.api_key = os.getenv("COZE_API_KEY")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.open_api_headers = {
            **self.headers,
            "Agw-Js-Conv": "str",
        }

    @with_retry()
    def create_dataset(self, name: str, space_id: str, format_type: int = 0):
        url = f"{self.base_url}/v1/datasets"
        data = {
            "name": name,
            "space_id": space_id,
            "format_type": format_type,
        }
        resp = requests.post(url, headers=self.headers, json=data, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def create_document(
        self,
        dataset_id: str,
        document_bases: list,
    ):
        url = f"{self.base_url}/open_api/knowledge/document/create"
        payload = {
            "dataset_id": dataset_id,
            "chunk_strategy": {
                "chunk_type": 0,
                "separator": "#",
                "max_tokens": 800,
                "remove_extra_spaces": True,
                "remove_urls_emails": True,
                "caption_type": 0,
            },
            "format_type": 0,
            "document_bases": document_bases,
        }

        resp = requests.post(url, headers=self.open_api_headers, json=payload, timeout=300)
        resp.raise_for_status()
        body = resp.json()
        return body

    @with_retry()
    def list_documents(self, dataset_id: str):
        url = f"{self.base_url}/open_api/knowledge/document/list"
        payload = {"dataset_id": dataset_id}
        resp = requests.post(url, headers=self.open_api_headers, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()

    @with_retry()
    def process_status(self, dataset_id: str, document_ids: list[str]):
        url = f"{self.base_url}/v1/datasets/{dataset_id}/process"
        payload = {"document_ids": document_ids}
        resp = requests.post(url, headers=self.headers, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()


if __name__ == "__main__":
    messages = [
        {"role": "user", "content": "杭州西湖有什么好玩的"},
        {"role": "assistant", "content": "杭州西湖有好多松鼠，还有断桥"},
    ]
    user_id = "lme_exper_user_default_499"
    iso_date = "2023-05-01T00:00:00.000Z"
    query = "杭州西湖有什么"
    top_k = 5

    # MEMOS-API
    client = MemosApiOnlineClient()
    for m in messages:
        m["created_at"] = iso_date
    client.add(messages, user_id, user_id)
    memories = client.search(query, user_id, top_k)
    print(memories)
