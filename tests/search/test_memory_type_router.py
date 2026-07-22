import pytest

from memos.search.memory_type_router import (
    MemoryIntent,
    MemoryRoutingScene,
    route_memory_search,
)


@pytest.mark.parametrize(
    ("query", "expected_intent", "expected_reason"),
    [
        ("你还记得我之前说过的项目吗？", MemoryIntent.GENERAL, "explicit_memory"),
        ("请按照我的偏好推荐一部电影", MemoryIntent.PREFERENCE, "explicit_preference"),
        (
            "你还记得我之前说过什么吗？请按照我的偏好推荐",
            MemoryIntent.BOTH,
            "explicit_memory_and_preference",
        ),
        ("我最近想换手机，帮我推荐一下", MemoryIntent.PREFERENCE, "implicit_personalization"),
        ("解释什么是向量数据库", MemoryIntent.NONE, "standalone_task"),
    ],
)
def test_route_memory_search_matches_documented_rules(
    query: str,
    expected_intent: MemoryIntent,
    expected_reason: str,
):
    decision = route_memory_search(query, MemoryRoutingScene.NORMAL)

    assert decision.intent is expected_intent
    assert decision.reason == expected_reason
    assert decision.matched_rules


@pytest.mark.parametrize(
    "query",
    [
        "HEARTBEAT",
        "[cron:1a516e4a-5c29-4634-9d3e-98c5576b533c 每小时同步] run task",
        ("System: [2026-07-04 01:00:00 UTC] reminder\n\nA scheduled reminder has been triggered."),
    ],
)
def test_route_memory_search_recognizes_system_templates(query: str):
    decision = route_memory_search(query, MemoryRoutingScene.NORMAL)

    assert decision.intent is MemoryIntent.SYSTEM
    assert decision.plan.search_working is False
    assert decision.plan.search_longterm is False
    assert decision.plan.search_user is False
    assert decision.plan.search_preference is False


def test_route_memory_search_does_not_treat_cron_discussion_as_system():
    decision = route_memory_search("请解释 cron 是什么", MemoryRoutingScene.NORMAL)

    assert decision.intent is MemoryIntent.NONE
    assert decision.reason == "standalone_task"


def test_route_memory_search_uses_scene_default_for_unmatched_query():
    normal = route_memory_search("今天天气不错", MemoryRoutingScene.NORMAL)
    minimal = route_memory_search("今天天气不错", MemoryRoutingScene.MINIMAL)

    assert normal.intent is MemoryIntent.GENERAL
    assert normal.reason == "normal_default"
    assert normal.plan.search_working is True
    assert normal.plan.search_longterm is True
    assert normal.plan.search_user is True
    assert normal.plan.search_preference is False

    assert minimal.intent is MemoryIntent.NONE
    assert minimal.reason == "minimal_default"
    assert minimal.plan.search_working is False
    assert minimal.plan.search_longterm is False
    assert minimal.plan.search_user is False
    assert minimal.plan.search_preference is False


def test_route_memory_search_applies_denials_after_positive_intent():
    decision = route_memory_search(
        "你还记得我之前说过的需求吗？请按照我的偏好推荐，但不要参考历史对话",
        MemoryRoutingScene.NORMAL,
    )

    assert decision.intent is MemoryIntent.PREFERENCE
    assert decision.plan.search_working is True
    assert decision.plan.search_longterm is False
    assert decision.plan.search_user is False
    assert decision.plan.search_preference is True
    assert "deny_general" in decision.matched_rules


def test_route_memory_search_maps_fully_denied_intent_to_scene_none_plan():
    normal = route_memory_search(
        "请按照我的偏好推荐，但不要参考我的偏好",
        MemoryRoutingScene.NORMAL,
    )
    minimal = route_memory_search(
        "请按照我的偏好推荐，但不要参考我的偏好",
        MemoryRoutingScene.MINIMAL,
    )

    assert normal.intent is MemoryIntent.NONE
    assert normal.plan.search_working is True
    assert minimal.intent is MemoryIntent.NONE
    assert minimal.plan.search_working is False
