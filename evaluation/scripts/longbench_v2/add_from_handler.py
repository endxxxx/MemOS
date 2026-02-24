import os
import random as _random
import socket
import time

from dotenv import load_dotenv

from memos.api import handlers
from memos.api.handlers.add_handler import AddHandler
from memos.api.handlers.base_handler import HandlerDependencies
from memos.api.handlers.search_handler import SearchHandler
from memos.api.product_models import APIADDRequest, APISearchRequest
from memos.log import get_logger


load_dotenv()

logger = get_logger(__name__)

# Instance ID for identifying this server instance in logs and responses
INSTANCE_ID = f"{socket.gethostname()}:{os.getpid()}:{_random.randint(1000, 9999)}"


graph_db_backend = os.getenv("NEO4J_BACKEND", "nebular").lower()
print("graph_db_backend: ", graph_db_backend)

# Initialize all server components (graph_db 会自动传入 mem_reader)
print("=" * 80)
print("初始化服务组件...")
print("=" * 80)
components = handlers.init_server()

# 获取关键组件
mem_reader = components["mem_reader"]
graph_db = components["graph_db"]
embedder = components["embedder"]

# Create dependency container
dependencies = HandlerDependencies.from_init_server(components)

# Initialize all handlers with dependency injection
search_handler = SearchHandler(dependencies)
add_handler = AddHandler(dependencies)


def add_memories(add_req: APIADDRequest):
    """
    Add memories for a specific user.
    This endpoint uses the class-based AddHandler for better code organization.
    """
    return add_handler.handle_add_memories(add_req)


def search_memories(search_req: APISearchRequest):
    return search_handler.handle_search_memories(search_req)


if __name__ == "__main__":
    user_id = "asdfasdfasdfasdfasdfasdfasdfsd"
    cube_id = "cube" + user_id
    req = APIADDRequest.model_validate(
        {
            "user_id": user_id,
            "session_id": "default_session",
            "async_mode": "sync",
            "writable_cube_ids": [cube_id],
            "messages": [
                {
                    "file": {
                        "file_data": "https://memos-knowledge-base-file-pre.oss-cn-shanghai.aliyuncs.com/longbench_v2_text_files/66ebbe4f5a08c7b9b35de533.txt",
                        "file_id": "c07da7508178fa8d54805b030be7610f",
                        "filename": "6707f349bb02136c067d13b9.txt",
                    },
                    "type": "file",
                }
            ],
        }
    )
    init_time = time.time()
    res = add_memories(req)
    print(res)
    print(f"time duration: {time.time() - init_time}")
