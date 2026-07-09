"""Unit tests for the kimi-native transcript forwarder.

Covers the stateful row planner against kimi's real ``wire.jsonl`` event schema
(turn.prompt, step.begin/end, content.part text/think, tool.call/tool.result,
turn.cancel), the request-body shapes, the line-offset state round-trip, and
workspace/recency session discovery. The live POST loop is exercised by the e2e
gate, not here.
"""

from __future__ import annotations

import json
from pathlib import Path

from omnigent.kimi_native_forwarder import (
    _discover_wire,
    _ForwardState,
    _FunctionCallPost,
    _FunctionOutputPost,
    _MessagePost,
    _plan_row,
    _post_body,
    _read_new_rows,
    _read_state,
    _ReasoningPost,
    _StatusPost,
    _TurnState,
    _write_state,
    clear_kimi_bridge_state,
)

_AGENT = "kimi-native-ui"

# --- wire-row builders (mirror the shapes seen in a real wire.jsonl) ---------


def _loop(etype: str, **fields: object) -> dict[str, object]:
    return {"type": "context.append_loop_event", "event": {"type": etype, **fields}}


def _part(turn_id: str | None, part_type: str, text: str, uuid: str = "u") -> dict[str, object]:
    event: dict[str, object] = {
        "type": "content.part",
        "uuid": uuid,
        "part": {"type": part_type, "text": text},
    }
    if turn_id is not None:
        event["turnId"] = turn_id
    return {"type": "context.append_loop_event", "event": event}


def _think(turn_id: str | None, think: str, uuid: str = "u") -> dict[str, object]:
    event: dict[str, object] = {
        "type": "content.part",
        "uuid": uuid,
        "part": {"type": "think", "think": think},
    }
    if turn_id is not None:
        event["turnId"] = turn_id
    return {"type": "context.append_loop_event", "event": event}


def _tool_call(
    turn_id: str, call_id: str, name: str, args: dict[str, object]
) -> dict[str, object]:
    return _loop(
        "tool.call", turnId=turn_id, toolCallId=call_id, uuid=call_id, name=name, args=args
    )


def _tool_result(call_id: str, output: str) -> dict[str, object]:
    # tool.result carries no turnId — only the call id and the output.
    return _loop("tool.result", toolCallId=call_id, parentUuid=call_id, result={"output": output})


def _user_prompt(text: str) -> dict[str, object]:
    return {
        "type": "turn.prompt",
        "input": [{"type": "text", "text": text}],
        "origin": {"kind": "user"},
    }


def _plan(line_no: int, row: dict[str, object], state: _TurnState | None = None):
    return _plan_row(line_no, row, state or _TurnState())


class TestPlanRowMessages:
    def test_user_prompt_keeps_own_line_id(self) -> None:
        posts, state = _plan(4, _user_prompt("what is in this repo?"))
        assert posts == [_MessagePost(4, "user", "what is in this repo?", "kimi:turn:4")]
        assert state == _TurnState()  # user prompt carries no turnId → no turn state

    def test_non_user_prompt_skipped(self) -> None:
        row = {
            "type": "turn.prompt",
            "input": [{"type": "text", "text": "x"}],
            "origin": {"kind": "system"},
        }
        assert _plan(0, row) == ([], _TurnState())

    def test_assistant_text_uses_turn_id(self) -> None:
        posts, state = _plan(30, _part("3", "text", "hello"), _TurnState("3", True))
        assert posts == [_MessagePost(30, "assistant", "hello", "kimi:turn:3")]
        assert state.turn_id == "3"

    def test_assistant_text_falls_back_to_uuid_without_turn(self) -> None:
        # A content.part with no turnId and no carried turn → per-event id.
        posts, _ = _plan(9, _part(None, "text", "hi", uuid="67ce67f7"))
        assert posts == [_MessagePost(9, "assistant", "hi", "kimi:67ce67f7")]

    def test_think_part_becomes_reasoning(self) -> None:
        posts, state = _plan(30, _think("3", "let me think"), _TurnState("3", True))
        assert posts == [_ReasoningPost(30, "let me think")]
        assert state.turn_id == "3"

    def test_noise_rows_skipped(self) -> None:
        for row in (
            {"type": "metadata", "protocol_version": 1},
            {"type": "usage.record", "usage": {}},
            {"type": "llm.request"},
            {"type": "context.append_message", "message": {"role": "user", "content": []}},
        ):
            assert _plan(0, row) == ([], _TurnState())


class TestPlanRowStatus:
    def test_step_begin_posts_running_with_id_once(self) -> None:
        posts, state = _plan(28, _loop("step.begin", turnId="3", step=1))
        assert posts == [_StatusPost(28, "running", "kimi:turn:3")]
        assert state == _TurnState(turn_id="3", running=True)
        # A second step.begin in the same turn does not re-post running.
        posts2, state2 = _plan_row(35, _loop("step.begin", turnId="3", step=2), state)
        assert posts2 == []
        assert state2 == _TurnState(turn_id="3", running=True)

    def test_step_end_tool_use_keeps_running(self) -> None:
        row = _loop("step.end", turnId="3", step=1, finishReason="tool_use")
        posts, state = _plan_row(33, row, _TurnState("3", True))
        assert posts == []
        assert state == _TurnState(turn_id="3", running=True)

    def test_step_end_end_turn_posts_idle_and_resets(self) -> None:
        row = _loop("step.end", turnId="3", step=2, finishReason="end_turn")
        posts, state = _plan_row(38, row, _TurnState("3", True))
        assert posts == [_StatusPost(38, "idle", "kimi:turn:3")]
        assert state == _TurnState()

    def test_step_end_without_running_resets_silently(self) -> None:
        row = _loop("step.end", turnId="0", step=1, finishReason="end_turn")
        posts, state = _plan_row(10, row, _TurnState("0", False))
        assert posts == []
        assert state == _TurnState()

    def test_turn_cancel_posts_idle_when_running(self) -> None:
        posts, state = _plan_row(18, {"type": "turn.cancel"}, _TurnState("1", True))
        assert posts == [_StatusPost(18, "idle", "kimi:turn:1")]
        assert state == _TurnState()

    def test_turn_cancel_noop_when_idle(self) -> None:
        assert _plan(18, {"type": "turn.cancel"}) == ([], _TurnState())


class TestPlanRowTools:
    def test_tool_call_becomes_function_call(self) -> None:
        row = _tool_call("3", "Agent:0", "Agent", {"q": "x"})
        posts, state = _plan_row(31, row, _TurnState("3", True))
        args = json.dumps({"q": "x"}, ensure_ascii=True)
        assert posts == [_FunctionCallPost(31, "Agent:0", "Agent", args, "kimi:turn:3")]
        assert state.turn_id == "3"

    def test_tool_result_uses_carried_turn_id(self) -> None:
        # tool.result has no turnId; it must inherit the remembered turn.
        posts, _ = _plan_row(32, _tool_result("Agent:0", "22:00"), _TurnState("3", True))
        assert posts == [_FunctionOutputPost(32, "Agent:0", "22:00", "kimi:turn:3")]


# --- golden replay of a real multi-step, tool-using turn (turn 3) ------------

_GOLDEN_TURN = [
    (26, _user_prompt("help me find when norway and england quarter final starts")),
    (27, {"type": "context.append_message", "message": {"role": "user", "content": []}}),
    (28, _loop("step.begin", turnId="3", step=1)),
    (29, {"type": "llm.request"}),
    (30, _part("3", "text", "I will perform a web search.")),
    (31, _tool_call("3", "Agent:0", "Agent", {"description": "web search"})),
    (32, _tool_result("Agent:0", "starts at 22:00 on Sat 11 Jul")),
    (33, _loop("step.end", turnId="3", step=1, finishReason="tool_use")),
    (34, {"type": "usage.record", "usage": {}}),
    (35, _loop("step.begin", turnId="3", step=2)),
    (36, {"type": "llm.request"}),
    (37, _part("3", "text", "The match starts at 22:00 on Sat 11 Jul.")),
    (38, _loop("step.end", turnId="3", step=2, finishReason="end_turn")),
    (39, {"type": "usage.record", "usage": {}}),
]


class TestGoldenTurn:
    def _replay(self):
        state = _TurnState()
        posts: list[object] = []
        for line_no, row in _GOLDEN_TURN:
            batch, state = _plan_row(line_no, row, state)
            posts.extend(batch)
        return posts, state

    def test_emits_expected_sequence(self) -> None:
        posts, state = self._replay()
        kinds = [type(p).__name__ for p in posts]
        assert kinds == [
            "_MessagePost",  # user prompt
            "_StatusPost",  # running
            "_MessagePost",  # assistant narration
            "_FunctionCallPost",  # Agent call
            "_FunctionOutputPost",  # Agent result
            "_MessagePost",  # final answer
            "_StatusPost",  # idle
        ]
        statuses = [p.status for p in posts if isinstance(p, _StatusPost)]
        assert statuses == ["running", "idle"]
        assert state == _TurnState()  # turn fully released

    def test_assistant_side_shares_one_response_id(self) -> None:
        posts, _ = self._replay()
        ids: set[str] = set()
        for p in posts:
            if isinstance(p, (_FunctionCallPost, _FunctionOutputPost)):
                ids.add(p.response_id)
            elif isinstance(p, _MessagePost) and p.role == "assistant":
                ids.add(p.response_id)
            elif isinstance(p, _StatusPost) and p.response_id:
                ids.add(p.response_id)
        assert ids == {"kimi:turn:3"}

    def test_tool_call_and_output_share_call_id(self) -> None:
        posts, _ = self._replay()
        call = next(p for p in posts if isinstance(p, _FunctionCallPost))
        out = next(p for p in posts if isinstance(p, _FunctionOutputPost))
        assert call.call_id == out.call_id == "Agent:0"


class TestPostBody:
    def test_running_status_carries_response_id(self) -> None:
        body = _post_body(_StatusPost(28, "running", "kimi:turn:3"), _AGENT)
        assert body == {
            "type": "external_session_status",
            "data": {"status": "running", "response_id": "kimi:turn:3"},
        }

    def test_status_without_response_id_omits_it(self) -> None:
        body = _post_body(_StatusPost(0, "idle", None), _AGENT)
        assert body == {"type": "external_session_status", "data": {"status": "idle"}}

    def test_reasoning_body(self) -> None:
        body = _post_body(_ReasoningPost(30, "hmm"), _AGENT)
        assert body == {
            "type": "external_output_reasoning_delta",
            "data": {"delta": "hmm", "started": True},
        }

    def test_user_message_body(self) -> None:
        body = _post_body(_MessagePost(4, "user", "hi", "kimi:turn:4"), _AGENT)
        assert body["type"] == "external_conversation_item"
        assert body["data"] == {
            "item_type": "message",
            "item_data": {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            "response_id": "kimi:turn:4",
        }

    def test_assistant_message_carries_agent(self) -> None:
        body = _post_body(_MessagePost(30, "assistant", "yo", "kimi:turn:3"), _AGENT)
        assert body["data"]["item_data"] == {
            "role": "assistant",
            "content": [{"type": "output_text", "text": "yo"}],
            "agent": _AGENT,
        }

    def test_function_call_body(self) -> None:
        post = _FunctionCallPost(31, "Agent:0", "Agent", '{"q": "x"}', "kimi:turn:3")
        body = _post_body(post, _AGENT)
        assert body["data"] == {
            "item_type": "function_call",
            "item_data": {
                "agent": _AGENT,
                "name": "Agent",
                "arguments": '{"q": "x"}',
                "call_id": "Agent:0",
            },
            "response_id": "kimi:turn:3",
        }

    def test_function_output_body(self) -> None:
        post = _FunctionOutputPost(32, "Agent:0", "done", "kimi:turn:3")
        body = _post_body(post, _AGENT)
        assert body["data"] == {
            "item_type": "function_call_output",
            "item_data": {"call_id": "Agent:0", "output": "done"},
            "response_id": "kimi:turn:3",
        }


class TestReadNewRows:
    def _wire(self, tmp_path: Path) -> Path:
        rows = [
            {"type": "metadata", "protocol_version": 1},
            _user_prompt("hi"),
            _think("0", "…"),
            _part("0", "text", "hello!"),
        ]
        p = tmp_path / "wire.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        return p

    def test_returns_line_indexed_rows(self, tmp_path: Path) -> None:
        rows = _read_new_rows(self._wire(tmp_path), 0)
        assert [(n, r["type"]) for n, r in rows] == [
            (0, "metadata"),
            (1, "turn.prompt"),
            (2, "context.append_loop_event"),
            (3, "context.append_loop_event"),
        ]

    def test_offset_skips_already_seen(self, tmp_path: Path) -> None:
        rows = _read_new_rows(self._wire(tmp_path), 2)
        assert [n for n, _ in rows] == [2, 3]

    def test_missing_file_is_empty(self, tmp_path: Path) -> None:
        assert _read_new_rows(tmp_path / "nope.jsonl", 0) == []


class TestState:
    def test_round_trip_and_clear(self, tmp_path: Path) -> None:
        assert _read_state(tmp_path) is None
        _write_state(tmp_path, _ForwardState(wire_path="/x/wire.jsonl", last_line=7))
        loaded = _read_state(tmp_path)
        assert loaded is not None
        assert loaded.wire_path == "/x/wire.jsonl"
        assert loaded.last_line == 7
        clear_kimi_bridge_state(tmp_path)
        assert _read_state(tmp_path) is None


class TestDiscoverWire:
    def _make_session(
        self, home: Path, session_dir_name: str, work_dir: str, *, mtime: float
    ) -> Path:
        wire = home / "sessions" / "wd_x" / session_dir_name / "agents" / "main" / "wire.jsonl"
        wire.parent.mkdir(parents=True, exist_ok=True)
        wire.write_text("{}\n", encoding="utf-8")
        import os

        os.utime(wire, (mtime, mtime))
        idx = home / "session_index.jsonl"
        index_row = {"sessionDir": str(wire.parent.parent.parent), "workDir": work_dir}
        with idx.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(index_row) + "\n")
        return wire

    def test_picks_newest_matching_workspace(self, tmp_path: Path) -> None:
        home = tmp_path / "kimi-code-home"
        home.mkdir()
        self._make_session(home, "session_old", "/ws", mtime=1000.0)
        newest = self._make_session(home, "session_new", "/ws", mtime=2000.0)
        self._make_session(home, "session_other", "/different", mtime=3000.0)
        assert _discover_wire(home, "/ws", launch_epoch_ms=0) == newest

    def test_none_before_any_session(self, tmp_path: Path) -> None:
        home = tmp_path / "kimi-code-home"
        home.mkdir()
        assert _discover_wire(home, "/ws", launch_epoch_ms=0) is None

    def test_ignores_sessions_before_launch(self, tmp_path: Path) -> None:
        home = tmp_path / "kimi-code-home"
        home.mkdir()
        self._make_session(home, "session_stale", "/ws", mtime=1000.0)
        assert _discover_wire(home, "/ws", launch_epoch_ms=9_000_000_000_000) is None
