from unittest.mock import Mock

import pytest

from memos.api.handlers.search_handler import SearchHandler
from memos.api.product_models import APISearchRequest
from memos.search.memory_type_router import MemoryRoutingScene, route_memory_search


def _request(**kwargs) -> APISearchRequest:
    payload = {"query": "unmatched", "user_id": "user-1"}
    payload.update(kwargs)
    return APISearchRequest(**payload)


def test_search_request_defaults_to_normal_memory_routing_scene():
    assert _request().memory_routing_scene == "normal"


def test_search_request_rejects_unknown_memory_routing_scene():
    with pytest.raises(ValueError, match="memory_routing_scene"):
        _request(memory_routing_scene="unknown")


def test_search_handler_uses_request_memory_routing_scene(monkeypatch):
    monkeypatch.setenv("MEMOS_MEMORY_TYPE_ROUTING_ENABLED", "true")
    handler = SearchHandler.__new__(SearchHandler)
    handler.logger = Mock()

    normal = handler._resolve_memory_route(_request(memory_routing_scene="normal"))
    minimal = handler._resolve_memory_route(_request(memory_routing_scene="minimal"))

    assert normal is not None
    assert normal.plan.search_longterm is True
    assert minimal is not None
    assert minimal.plan.searches_any_memory is False


def test_search_handler_memory_routing_flag_can_disable_routing(monkeypatch):
    monkeypatch.setenv("MEMOS_MEMORY_TYPE_ROUTING_ENABLED", "false")
    handler = SearchHandler.__new__(SearchHandler)
    handler.logger = Mock()

    assert handler._resolve_memory_route(_request()) is None


def test_system_route_disables_auxiliary_retrieval_and_skips_search():
    req = _request(query="[cron:123 task] run", internet_search=True)
    decision = route_memory_search(req.query, MemoryRoutingScene.NORMAL)

    SearchHandler._apply_memory_route(req, decision)

    assert req.include_preference is False
    assert req.include_skill_memory is False
    assert req.search_tool_memory is False
    assert req.internet_search is False
    assert SearchHandler._can_skip_memory_search(req, decision) is True


def test_preference_route_does_not_override_explicit_request_exclusion():
    req = _request(query="请按照我的偏好推荐", include_preference=False)
    decision = route_memory_search(req.query, MemoryRoutingScene.NORMAL)

    SearchHandler._apply_memory_route(req, decision)

    assert req.include_preference is False
    assert decision.plan.search_preference is True
