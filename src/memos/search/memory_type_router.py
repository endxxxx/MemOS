from __future__ import annotations

import re

from dataclasses import dataclass
from enum import Enum


class MemoryIntent(str, Enum):
    SYSTEM = "SYSTEM"
    NONE = "NONE"
    GENERAL = "GENERAL"
    PREFERENCE = "PREFERENCE"
    BOTH = "BOTH"


class MemoryRoutingScene(str, Enum):
    NORMAL = "normal"
    MINIMAL = "minimal"


@dataclass(frozen=True)
class MemorySearchPlan:
    search_working: bool
    search_longterm: bool
    search_user: bool
    search_preference: bool

    @property
    def searches_any_memory(self) -> bool:
        return any(
            (
                self.search_working,
                self.search_longterm,
                self.search_user,
                self.search_preference,
            )
        )


@dataclass(frozen=True)
class MemoryRouteDecision:
    intent: MemoryIntent
    reason: str
    matched_rules: tuple[str, ...]
    plan: MemorySearchPlan


NORMAL_SEARCH_PLANS = {
    MemoryIntent.SYSTEM: MemorySearchPlan(False, False, False, False),
    MemoryIntent.NONE: MemorySearchPlan(True, False, False, False),
    MemoryIntent.GENERAL: MemorySearchPlan(True, True, True, False),
    MemoryIntent.PREFERENCE: MemorySearchPlan(True, False, False, True),
    MemoryIntent.BOTH: MemorySearchPlan(True, True, True, True),
}

MINIMAL_SEARCH_PLANS = {
    MemoryIntent.SYSTEM: MemorySearchPlan(False, False, False, False),
    MemoryIntent.NONE: MemorySearchPlan(False, False, False, False),
    MemoryIntent.GENERAL: MemorySearchPlan(True, True, True, False),
    MemoryIntent.PREFERENCE: MemorySearchPlan(True, False, False, True),
    MemoryIntent.BOTH: MemorySearchPlan(True, True, True, True),
}

SCENE_SEARCH_PLANS = {
    MemoryRoutingScene.NORMAL: NORMAL_SEARCH_PLANS,
    MemoryRoutingScene.MINIMAL: MINIMAL_SEARCH_PLANS,
}


def _patterns(*items: tuple[str, str]) -> tuple[tuple[str, re.Pattern[str]], ...]:
    return tuple((name, re.compile(pattern, re.IGNORECASE | re.DOTALL)) for name, pattern in items)


SYSTEM_PATTERNS = _patterns(
    ("system_heartbeat", r"^\s*HEARTBEAT(?:_OK)?\s*$"),
    ("system_cron", r"^\s*\[cron:[^\]]+\]"),
    (
        "system_scheduled_reminder",
        r"^\s*System:\s*\[[^\]]+\][\s\S]*A scheduled reminder has been triggered",
    ),
)

EXPLICIT_MEMORY_PATTERNS = _patterns(
    ("explicit_memory_remember", r"你还记得"),
    ("explicit_memory_previous", r"我之前(说过|提过|告诉过)"),
    ("explicit_memory_history", r"(根据|结合|回顾).{0,8}(之前|历史|过去)"),
    ("explicit_memory_last_time", r"(我)?上次(我们|我|说过|提过|讨论|聊过|做过)"),
    ("explicit_memory_en_remember", r"\bdo you remember\b"),
    (
        "explicit_memory_en_history",
        r"\bbased on (our|my) (history|previous)\b",
    ),
)

EXPLICIT_PREFERENCE_PATTERNS = _patterns(
    (
        "explicit_preference_named",
        r"(根据|按照|结合).{0,8}(我的)?(偏好|喜好|习惯|口味|风格)",
    ),
    ("explicit_preference_suitable", r"(适合|更适合)我"),
    ("explicit_preference_for_me", r"为我(推荐|选择|定制|规划)"),
    ("explicit_preference_personalized", r"个性化(推荐|方案)"),
    (
        "explicit_preference_en_named",
        r"\bbased on my (preferences|taste|habits)\b",
    ),
    (
        "explicit_preference_en_personalized",
        r"\b(personalized|tailored) (recommendation|plan)\b",
    ),
)

RECOMMENDATION_PATTERNS = _patterns(
    (
        "recommendation_term",
        r"推荐|建议|选哪个|哪个好|怎么选|帮我选|安排|规划|制定|"
        r"\brecommend\b|\bsuggest\b|\bwhich one\b|\bchoose\b|\bplan\b",
    ),
)

SELF_CONTEXT_PATTERNS = _patterns(
    (
        "self_context",
        r"我想|我要|我准备|我打算|我最近|我平时|我的|适合我|帮我|"
        r"\bI want\b|\bI need\b|\bfor me\b|\bmy\b",
    ),
)

STANDALONE_PATTERNS = _patterns(
    ("standalone_command", r"^(翻译|计算|解释|总结|改写)"),
    ("standalone_definition", r"(是什么|什么意思|怎么定义)[？?]?$"),
    (
        "standalone_command_en",
        r"^(translate|calculate|define|summarize|rewrite)\b",
    ),
)

DENY_PREFERENCE_PATTERNS = _patterns(
    (
        "deny_preference",
        r"(不要|不用|无需|别).{0,8}(参考|结合|考虑|使用).{0,8}"
        r"(我的)?(偏好|喜好|习惯|口味|风格)",
    ),
    ("deny_preference_ignore", r"(忽略|不要使用).{0,8}(偏好|个性化信息)"),
)

DENY_GENERAL_PATTERNS = _patterns(
    (
        "deny_general",
        r"(不要|不用|无需|别).{0,8}(参考|结合|使用).{0,8}"
        r"(历史|之前的对话|过去的记忆)",
    ),
    (
        "deny_general_ignore",
        r"(忽略|忘掉).{0,8}(之前|历史|过去).{0,8}(内容|对话|记忆)",
    ),
)


def _matched_rules(
    query: str, patterns: tuple[tuple[str, re.Pattern[str]], ...]
) -> tuple[str, ...]:
    return tuple(name for name, pattern in patterns if pattern.search(query))


def _intent_requirements(intent: MemoryIntent) -> tuple[bool, bool]:
    return (
        intent in (MemoryIntent.GENERAL, MemoryIntent.BOTH),
        intent in (MemoryIntent.PREFERENCE, MemoryIntent.BOTH),
    )


def _intent_from_requirements(need_general: bool, need_preference: bool) -> MemoryIntent:
    if need_general and need_preference:
        return MemoryIntent.BOTH
    if need_general:
        return MemoryIntent.GENERAL
    if need_preference:
        return MemoryIntent.PREFERENCE
    return MemoryIntent.NONE


def route_memory_search(
    query: str,
    scene: MemoryRoutingScene | str = MemoryRoutingScene.NORMAL,
) -> MemoryRouteDecision:
    resolved_scene = MemoryRoutingScene(scene)
    normalized_query = query.strip()

    system_matches = _matched_rules(normalized_query, SYSTEM_PATTERNS)
    if system_matches:
        return MemoryRouteDecision(
            intent=MemoryIntent.SYSTEM,
            reason="system_template",
            matched_rules=system_matches,
            plan=SCENE_SEARCH_PLANS[resolved_scene][MemoryIntent.SYSTEM],
        )

    deny_general_matches = _matched_rules(normalized_query, DENY_GENERAL_PATTERNS)
    deny_preference_matches = _matched_rules(normalized_query, DENY_PREFERENCE_PATTERNS)
    explicit_memory_matches = _matched_rules(normalized_query, EXPLICIT_MEMORY_PATTERNS)
    explicit_preference_matches = _matched_rules(normalized_query, EXPLICIT_PREFERENCE_PATTERNS)

    if explicit_memory_matches and explicit_preference_matches:
        intent = MemoryIntent.BOTH
        reason = "explicit_memory_and_preference"
        positive_matches = (*explicit_memory_matches, *explicit_preference_matches)
    elif explicit_memory_matches:
        intent = MemoryIntent.GENERAL
        reason = "explicit_memory"
        positive_matches = explicit_memory_matches
    elif explicit_preference_matches:
        intent = MemoryIntent.PREFERENCE
        reason = "explicit_preference"
        positive_matches = explicit_preference_matches
    else:
        recommendation_matches = _matched_rules(normalized_query, RECOMMENDATION_PATTERNS)
        self_context_matches = _matched_rules(normalized_query, SELF_CONTEXT_PATTERNS)
        if recommendation_matches and self_context_matches:
            intent = MemoryIntent.PREFERENCE
            reason = "implicit_personalization"
            positive_matches = (*recommendation_matches, *self_context_matches)
        else:
            standalone_matches = _matched_rules(normalized_query, STANDALONE_PATTERNS)
            if standalone_matches:
                intent = MemoryIntent.NONE
                reason = "standalone_task"
                positive_matches = standalone_matches
            elif resolved_scene is MemoryRoutingScene.NORMAL:
                intent = MemoryIntent.GENERAL
                reason = "normal_default"
                positive_matches = ("normal_default",)
            else:
                intent = MemoryIntent.NONE
                reason = "minimal_default"
                positive_matches = ("minimal_default",)

    need_general, need_preference = _intent_requirements(intent)
    if deny_general_matches:
        need_general = False
    if deny_preference_matches:
        need_preference = False
    effective_intent = _intent_from_requirements(need_general, need_preference)

    return MemoryRouteDecision(
        intent=effective_intent,
        reason=reason,
        matched_rules=(
            *positive_matches,
            *deny_general_matches,
            *deny_preference_matches,
        ),
        plan=SCENE_SEARCH_PLANS[resolved_scene][effective_intent],
    )
