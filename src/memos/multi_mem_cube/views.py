from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol


if TYPE_CHECKING:
    from memos.api.product_models import APIADDRequest, APIFeedbackRequest, APISearchRequest
    from memos.search.memory_type_router import MemorySearchPlan


class MemCubeView(Protocol):
    """
    A high-level cube view used by AddHandler.
    It may wrap a single logical cube or multiple cubes,
    but exposes a unified add_memories interface.
    """

    def add_memories(self, add_req: APIADDRequest) -> list[dict[str, Any]]:
        """
        Process add_req, extract memories and write them into one or more cubes.

        Returns:
            A list of memory dicts, each item should at least contain:
            - memory
            - memory_id
            - memory_type
            - cube_id
        """
        ...

    def search_memories(
        self,
        search_req: APISearchRequest,
        memory_search_plan: MemorySearchPlan | None = None,
    ) -> dict[str, Any]:
        """
        Process search_req, read memories from one or more cubes and search them.

        Returns:
            A list of memory dicts, each item should at least contain:
            - memory
            - memory_id
            - memory_type
            - cube_id
        """
        ...

    def feedback_memories(self, feedback_req: APIFeedbackRequest) -> dict[str, Any]:
        """
        Process feedback_req, read memories from one or more cubes and feedback them.

        Returns:
            A list of memory dicts, each item should at least contain:
            - memory
            - memory_id
            - memory_type
            - cube_id
        """
        ...
