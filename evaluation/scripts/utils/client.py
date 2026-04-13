import json
import os
import sys
import time
import uuid

from contextlib import suppress
from datetime import datetime

import requests

from dotenv import load_dotenv


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()


class ZepClient:
    def __init__(self):
        from zep_cloud.client import Zep

        api_key = os.getenv("ZEP_API_KEY")
        self.client = Zep(api_key=api_key)

    def add(self, messages, user_id, timestamp):
        iso_date = datetime.fromtimestamp(timestamp).isoformat()
        for msg in messages:
            self.client.graph.add(
                data=msg.get("role") + ": " + msg.get("content"),
                type="message",
                created_at=iso_date,
                group_id=user_id,
            )

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

    def add(self, messages, user_id, timestamp, batch_size=2):
        max_retries = 5
        for i in range(0, len(messages), batch_size):
            batch_messages = messages[i : i + batch_size]
            for attempt in range(max_retries):
                try:
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
                        )
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(2**attempt)
                    else:
                        raise e

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

    def add(self, messages, user_id, batch_size=2):
        """
        messages = [{"role": "assistant", "content": data, "created_at": iso_date}]
        """
        from memobase import ChatBlob

        real_uid = self.string_to_uuid(user_id)
        user = self.client.get_or_create_user(real_uid)
        for i in range(0, len(messages), batch_size):
            batch_messages = messages[i : i + batch_size]
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    _ = user.insert(ChatBlob(messages=batch_messages), sync=True)
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(2**attempt)
                    else:
                        raise e

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

    def delete_user(self, user_id):
        from memobase.error import ServerError

        real_uid = self.string_to_uuid(user_id)
        with suppress(ServerError):
            self.client.delete_user(real_uid)

    def string_to_uuid(self, s: str, salt="memobase_client"):
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, s + salt))


class MemosApiClient:
    def __init__(self):
        self.memos_url = os.getenv("MEMOS_URL")
        self.headers = {"Content-Type": "application/json", "Authorization": os.getenv("MEMOS_KEY")}

    def add(self, messages, user_id, conv_id, batch_size: int = 9999):
        """
        messages = [{"role": "assistant", "content": data, "chat_time": date_str}]
        """
        url = f"{self.memos_url}/product/add"
        added_memories = []
        for i in range(0, len(messages), batch_size):
            batch_messages = messages[i : i + batch_size]
            payload = json.dumps(
                {
                    "messages": batch_messages,
                    "user_id": user_id,
                    "mem_cube_id": user_id,
                    "conversation_id": conv_id,
                }
            )
            response = requests.request("POST", url, data=payload, headers=self.headers)
            assert response.status_code == 200, response.text
            assert json.loads(response.text)["message"] == "Memory added successfully", (
                response.text
            )
            added_memories += json.loads(response.text)["data"]
        return added_memories

    def search(self, query, user_id, top_k):
        """Search memories."""
        url = f"{self.memos_url}/product/search"
        payload = json.dumps(
            {
                "query": query,
                "user_id": user_id,
                "mem_cube_id": user_id,
                "conversation_id": "",
                "top_k": top_k,
                "mode": os.getenv("SEARCH_MODE", "fast"),
                "include_preference": True,
                "pref_top_k": 6,
                "relativity": 0,
            },
            ensure_ascii=False,
        )
        response = requests.request("POST", url, data=payload, headers=self.headers)
        assert response.status_code == 200, response.text
        assert json.loads(response.text)["message"] == "Search completed successfully", (
            response.text
        )
        return json.loads(response.text)["data"]


class MemosApiOnlineClient:
    def __init__(self):
        self.memos_url = os.getenv("MEMOS_ONLINE_URL")
        self.headers = {"Content-Type": "application/json", "Authorization": os.getenv("MEMOS_KEY")}

    def add(self, messages, user_id, conv_id=None, batch_size: int = 9999):
        url = f"{self.memos_url}/add/message"
        for i in range(0, len(messages), batch_size):
            batch_messages = messages[i : i + batch_size]
            payload = json.dumps(
                {
                    "messages": batch_messages,
                    "user_id": user_id,
                    "conversation_id": conv_id,
                }
            )

            max_retries = 5
            for attempt in range(max_retries):
                try:
                    response = requests.request("POST", url, data=payload, headers=self.headers)
                    assert response.status_code == 200, response.text
                    assert json.loads(response.text)["message"] == "ok", response.text
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(2**attempt)
                    else:
                        raise e

    def search(self, query, user_id, top_k):
        """Search memories."""
        url = f"{self.memos_url}/search/memory"
        payload = json.dumps(
            {
                "query": query,
                "user_id": user_id,
                "memory_limit_number": top_k,
                "mode": os.getenv("SEARCH_MODE", "fast"),
                "include_preference": True,
                "pref_top_k": 6,
            }
        )

        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = requests.request("POST", url, data=payload, headers=self.headers)
                assert response.status_code == 200, response.text
                assert json.loads(response.text)["message"] == "ok", response.text
                text_mem_res = json.loads(response.text)["data"]["memory_detail_list"]
                pref_mem_res = json.loads(response.text)["data"]["preference_detail_list"]
                preference_note = json.loads(response.text)["data"]["preference_note"]
                for i in text_mem_res:
                    i.update({"memory": i.pop("memory_value")})
                explicit_pref_string = "Explicit Preference:"
                implicit_pref_string = "\n\nImplicit Preference:"
                explicit_idx = 0
                implicit_idx = 0
                for pref in pref_mem_res:
                    if pref["preference_type"] == "explicit_preference":
                        explicit_pref_string += f"\n{explicit_idx + 1}. {pref['preference']}"
                        explicit_idx += 1
                    if pref["preference_type"] == "implicit_preference":
                        implicit_pref_string += f"\n{implicit_idx + 1}. {pref['preference']}"
                        implicit_idx += 1

                return {
                    "text_mem": [{"memories": text_mem_res}],
                    "pref_string": explicit_pref_string + implicit_pref_string + preference_note,
                }

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    raise e


class SupermemoryClient:
    def __init__(self):
        from supermemory import Supermemory

        self.client = Supermemory(api_key=os.getenv("SUPERMEMORY_API_KEY"))

    def add(self, messages, user_id):
        content = "\n".join(
            [f"{msg['chat_time']} {msg['role']}: {msg['content']}" for msg in messages]
        )
        max_retries = 5
        for attempt in range(max_retries):
            try:
                self.client.memories.add(content=content, container_tag=user_id)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    raise e

    def search(self, query, user_id, top_k):
        max_retries = 10
        for attempt in range(max_retries):
            try:
                results = self.client.search.memories(
                    q=query,
                    container_tag=user_id,
                    threshold=0,
                    rerank=True,
                    rewrite_query=True,
                    limit=top_k,
                )
                context = "\n\n".join([r.memory for r in results.results])
                return context
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    raise e


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

    def wait_for_completion(self, task_id):
        while True:
            status = self.memu_client.get_task_status(task_id)
            if status.status in ["SUCCESS", "FAILURE", "REVOKED"]:
                break
            time.sleep(2)


class OpenVikingClient:
    def __init__(self):
        import requests

        self.requests = requests
        self.base_url = os.getenv("OPENVIKING_URL", "http://localhost:1933")
        self.session_cache = {}

    def add(self, messages, user_id, timestamp, batch_size=50):
        max_retries = 5
        session_id = f"{user_id}_session"

        # Create session if it doesn't exist
        if session_id not in self.session_cache:
            for attempt in range(max_retries):
                try:
                    # Check if session exists
                    response = self.requests.get(f"{self.base_url}/api/v1/sessions/{session_id}")
                    if response.status_code == 404:
                        # Create new session
                        response = self.requests.post(f"{self.base_url}/api/v1/sessions")
                        response.raise_for_status()
                        session_data = response.json()
                        self.session_cache[session_id] = session_data["result"]["session_id"]
                    elif response.status_code == 200:
                        # Session already exists
                        self.session_cache[session_id] = session_id
                    else:
                        response.raise_for_status()
                    break
                except Exception:
                    if attempt < max_retries - 1:
                        time.sleep(2**attempt)
                    else:
                        # If session creation fails, continue with session_id
                        self.session_cache[session_id] = session_id
                        break

        # Get session ID
        current_session_id = self.session_cache[session_id]

        # Add messages to session
        for msg in messages:
            for attempt in range(max_retries):
                try:
                    response = self.requests.post(
                        f"{self.base_url}/api/v1/sessions/{current_session_id}/messages",
                        json={"role": msg.get("role", "user"), "content": msg.get("content", "")},
                    )
                    response.raise_for_status()
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(2**attempt)
                    else:
                        raise e

        # Commit session
        for attempt in range(max_retries):
            try:
                response = self.requests.post(
                    f"{self.base_url}/api/v1/sessions/{current_session_id}/commit"
                )
                response.raise_for_status()
                commit_data = response.json()
                task_id = commit_data["result"].get("task_id")

                # Wait for task to complete
                if task_id:
                    for _task_attempt in range(max_retries):
                        try:
                            task_response = self.requests.get(
                                f"{self.base_url}/api/v1/tasks/{task_id}"
                            )
                            task_response.raise_for_status()
                            task_status = task_response.json()["result"]["status"]
                            if task_status == "completed":
                                break
                            time.sleep(1)
                        except Exception:
                            time.sleep(1)
                break
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    # If commit fails, continue anyway
                    break

    def search(self, query, user_id, top_k):
        max_retries = 5
        session_id = f"{user_id}_session"

        # Get session ID from cache
        current_session_id = self.session_cache.get(session_id, session_id)

        for attempt in range(max_retries):
            try:
                # Send HTTP request to search using correct API format
                response = self.requests.post(
                    f"{self.base_url}/api/v1/search/search",
                    json={"query": query, "limit": top_k, "session_id": current_session_id},
                )
                response.raise_for_status()
                search_results = response.json()

                # Extract and format results
                results = []

                # Check for different types of results
                if "result" in search_results:
                    result_data = search_results["result"]
                    for item in result_data["memories"]:
                        if item.get("content"):
                            results.append(item["content"])
                        elif "uri" in item:
                            try:
                                # Read content from URI
                                read_response = self.requests.get(
                                    f"{self.base_url}/api/v1/content/read?uri={item['uri']}"
                                )
                                if read_response.status_code == 200:
                                    read_data = read_response.json()
                                    if "result" in read_data:
                                        results.append(read_data["result"])
                            except Exception:
                                pass

                return results
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    raise e


class VikingClient:
    def __init__(self):
        import requests

        self.requests = requests
        self.base_url = "https://api-knowledgebase.mlp.cn-beijing.volces.com"
        self.api_key = os.getenv("VIKING_MEMORY_API_KEY")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def add(self, messages, user_id, timestamp, batch_size=50):
        max_retries = 5
        url = f"{self.base_url}/api/memory/session/add"

        for i in range(0, len(messages), batch_size):
            batch_messages = messages[i : i + batch_size]
            for attempt in range(max_retries):
                try:
                    payload = {
                        "collection_name": "locomo_eval",
                        "project_name": "default",
                        "messages": batch_messages,
                        "metadata": {
                            "default_user_id": user_id,
                            "default_user_name": user_id,
                            "default_assistant_id": "assistant_01",
                            "default_assistant_name": "Robot",
                            "time": int(timestamp * 1000),  # 转换为毫秒
                        },
                    }

                    response = self.requests.post(url, headers=self.headers, json=payload)
                    response.raise_for_status()
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(2**attempt)
                    else:
                        raise e

    def search(self, query, user_id, top_k):
        max_retries = 5
        url = f"{self.base_url}/api/memory/get_context"

        for attempt in range(max_retries):
            try:
                payload = {
                    "collection_name": "locomo_eval",
                    "project_name": "default",
                    "conversation_id": f"conversation_{user_id}",
                    "query": query,
                    "event_search_config": {
                        "filter": {"user_id": user_id, "memory_type": ["event_v1"]},
                        "limit": top_k,
                        "time_decay_config": {"weight": 0.5, "no_decay_period": 3},
                    },
                    "profile_search_config": {
                        "filter": {"user_id": user_id, "memory_type": ["profile_v1"]},
                        "limit": 1,
                    },
                }

                response = self.requests.post(url, headers=self.headers, json=payload)
                response.raise_for_status()
                search_results = response.json()

                # Extract and format results
                results = []
                if "data" in search_results:
                    # 根据实际返回结构提取结果
                    # 这里需要根据实际 API 返回结构进行调整
                    # 暂时返回整个结果
                    results.append(str(search_results["data"]))

                return results
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    raise e


if __name__ == "__main__":
    messages = [
        {"role": "user", "content": "杭州西湖有什么好玩的"},
        {"role": "assistant", "content": "杭州西湖有好多松鼠，还有断桥"},
    ]
    user_id = "test_user"
    iso_date = "2023-05-01T00:00:00.000Z"
    timestamp = 1682899200
    query = "杭州西湖有什么"
    top_k = 5

    # MEMOS-API
    client = MemosApiClient()
    for m in messages:
        m["created_at"] = iso_date
    client.add(messages, user_id, user_id)
    memories = client.search(query, user_id, top_k)
    print(memories)
