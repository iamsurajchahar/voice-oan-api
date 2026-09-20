"""Issue #271: a previous turn's documents must not answer this turn's question.

A caller asked about cattle-shed subsidy and was given diarrhoea treatment
advice — the clinical documents from the turn before were still sitting in the
model's context, and they read richer than whatever the scheme lookup returned.
`trim_history` now keeps the replies but lets the documents behind older ones
fall away, so there is nothing stale to reach for.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from app.utils import trim_history


def _turn(n: int, *, tool: bool = True):
    """One user turn: question, optional search round-trip, spoken answer."""
    messages = [ModelRequest(parts=[UserPromptPart(content=f"question {n}")])]
    if tool:
        call_id = f"call-{n}"
        messages.append(
            ModelResponse(parts=[ToolCallPart(tool_name="search_documents", args={"query": f"q{n}"}, tool_call_id=call_id)])
        )
        messages.append(
            ModelRequest(parts=[ToolReturnPart(tool_name="search_documents", content=f"documents {n}", tool_call_id=call_id)])
        )
    messages.append(ModelResponse(parts=[TextPart(content=f"answer {n}")]))
    return messages


def _history(turns: int, **kwargs):
    return [m for n in range(1, turns + 1) for m in _turn(n, **kwargs)]


def _contents(history):
    return [getattr(p, "content", None) for m in history for p in m.parts]


def _tool_call_ids(history):
    return [
        p.tool_call_id
        for m in history
        for p in m.parts
        if getattr(p, "part_kind", "") in ("tool-call", "tool-return", "retry-prompt")
    ]


def test_only_the_recent_turns_keep_their_documents():
    trimmed = trim_history(_history(4), include_system_prompts=False, tool_return_turns=2)
    assert _tool_call_ids(trimmed) == ["call-3", "call-3", "call-4", "call-4"]
    assert "documents 1" not in _contents(trimmed)
    assert "documents 4" in _contents(trimmed)


def test_every_answer_survives_the_scoping():
    trimmed = trim_history(_history(4), include_system_prompts=False, tool_return_turns=1)
    spoken = _contents(trimmed)
    for n in range(1, 5):
        assert f"question {n}" in spoken
        assert f"answer {n}" in spoken


def test_zero_drops_every_previous_turns_documents():
    trimmed = trim_history(_history(3), include_system_prompts=False, tool_return_turns=0)
    assert _tool_call_ids(trimmed) == []
    assert "answer 3" in _contents(trimmed)


def test_none_keeps_everything_as_before():
    trimmed = trim_history(_history(4), include_system_prompts=False, tool_return_turns=None)
    assert _tool_call_ids(trimmed) == [f"call-{n}" for n in (1, 1, 2, 2, 3, 3, 4, 4)]


def test_calls_and_returns_are_always_dropped_together():
    # An orphaned tool-call is a hard error for the model API, so the scoping
    # has to remove both sides or neither.
    trimmed = trim_history(_history(4), include_system_prompts=False, tool_return_turns=1)
    ids = _tool_call_ids(trimmed)
    assert ids.count("call-4") == 2
    assert all(i == "call-4" for i in ids)


def test_a_retry_prompt_is_scoped_with_its_call():
    history = [
        ModelRequest(parts=[UserPromptPart(content="question 1")]),
        ModelResponse(parts=[ToolCallPart(tool_name="search_documents", args={}, tool_call_id="call-1")]),
        ModelRequest(parts=[RetryPromptPart(content="INVALID_QUERY", tool_name="search_documents", tool_call_id="call-1")]),
        ModelResponse(parts=[TextPart(content="answer 1")]),
        *_turn(2),
    ]
    trimmed = trim_history(history, include_system_prompts=False, tool_return_turns=1)
    assert _tool_call_ids(trimmed) == ["call-2", "call-2"]


def test_turns_without_tools_are_untouched():
    history = _history(3, tool=False)
    trimmed = trim_history(history, include_system_prompts=False, tool_return_turns=1)
    assert _contents(trimmed) == _contents(trim_history(history, include_system_prompts=False))


def test_voice_passes_the_configured_bound():
    from app.config import settings

    assert settings.history_tool_return_turns >= 0
