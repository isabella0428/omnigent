"""Mirror a kimi-native TUI session's transcript into the Omnigent web chat.

The kimi-native harness launches the interactive ``kimi`` TUI in a tmux pane and
injects web-UI turns into it (see :mod:`omnigent.kimi_native_bridge`). The TUI's
reply renders live in the embedded terminal, but — unlike the SDK ``KimiExecutor``
— nothing flows the assistant's response back into Omnigent's conversation
transcript (the chat bubbles). This module closes that gap, the kimi analog of
:mod:`omnigent.cursor_native_forwarder`.

Data source: kimi persists each session to an append-only JSONL "wire" log at
``$KIMI_CODE_HOME/sessions/<wd_…>/<session_…>/agents/main/wire.jsonl``. The
native harness points ``KIMI_CODE_HOME`` at ``<bridge_dir>/kimi-code-home`` whose
``sessions/`` is symlinked to the user's global store, so several workspaces'
sessions share the tree; we disambiguate by ``workDir`` (via ``session_index.jsonl``)
and recency. Relevant wire events:

- ``turn.prompt`` (``origin.kind == "user"``) → a user message.
- ``context.append_loop_event`` wraps the streamed turn. Its ``event.type`` is
  one of: ``step.begin`` / ``step.end`` (step boundaries, carrying ``turnId`` and
  a terminal ``finishReason``); ``content.part`` where ``part.type == "text"`` is
  an assistant message and ``part.type == "think"`` is reasoning (mirrored as a
  transient ``external_output_reasoning_delta`` from ``part["think"]``);
  ``tool.call`` and ``tool.result`` (a built-in tool invocation and its output).
- ``turn.cancel`` → the turn was interrupted.

Each turn is mirrored so the web chat matches the TUI: user/assistant text as
``external_conversation_item`` messages, think blocks as reasoning deltas, tool
calls as ``function_call`` / ``function_call_output`` items, and
``external_session_status`` ``running`` / ``idle`` edges bracketing the turn. All
of a turn's assistant-side items and its status edges share one ``response_id``
(``kimi:turn:<turnId>``) so the web renders in-flight tools as *live* cards
(spinner + ticking timer) rather than static ones.

kimi's wire has no ``turn.end`` event, and ``tool.result`` carries no ``turnId``,
so the forwarder is a *stateful* tailer: it remembers the active turn across
lines (:class:`_TurnState`) and treats a ``step.end`` whose ``finishReason`` is
not ``tool_use`` (or a ``turn.cancel``) as the turn's end. A per-session line
offset is persisted in ``<bridge_dir>/kimi_forwarder.json`` so restarts resume
without double-posting.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path

import httpx

_logger = logging.getLogger(__name__)

#: Poll cadence for new wire-log lines (matches cursor_native_forwarder).
_POLL_INTERVAL_S = 0.25
#: Persisted forwarder state (discovered wire path + high-water line count).
_STATE_FILE = "kimi_forwarder.json"
#: Clock-skew tolerance when matching a session created at/after launch.
_DISCOVER_SKEW_MS = 10_000
#: Supervisor backoff bounds.
_BACKOFF_INITIAL_S = 1.0
_BACKOFF_MAX_S = 30.0

#: Omnigent session-event types (must match the server ingestion route; shared
#: with the codex-/opencode-native forwarders).
_EXTERNAL_ITEM = "external_conversation_item"
_EXTERNAL_STATUS = "external_session_status"
_EXTERNAL_REASONING_DELTA = "external_output_reasoning_delta"
_STATUS_RUNNING = "running"
_STATUS_IDLE = "idle"

#: kimi ``step.end.finishReason`` meaning "paused this step to run a tool, more
#: steps follow" — the one non-terminal reason. Anything else (``end_turn`` …)
#: ends the turn.
_FINISH_TOOL_USE = "tool_use"


@dataclass
class _ForwardState:
    """Durable cursor for the wire-log tail."""

    wire_path: str
    last_line: int


@dataclass
class _TurnState:
    """In-memory turn tracking for the stateful tail.

    kimi emits no ``turn.end`` event and its ``tool.result`` carries no
    ``turnId``, so the active turn must be remembered across lines. ``turn_id``
    is kimi's ``turnId`` for the live turn (e.g. ``"3"``); ``running`` records
    whether a ``running`` status edge has already been posted for it, so it
    fires once per turn rather than once per step.
    """

    turn_id: str | None = None
    running: bool = False


@dataclass
class _MessagePost:
    """A user/assistant chat bubble to mirror."""

    line_no: int
    role: str
    text: str
    response_id: str


@dataclass
class _ReasoningPost:
    """A think block to mirror as a transient ``external_output_reasoning_delta``."""

    line_no: int
    delta: str


@dataclass
class _FunctionCallPost:
    """A tool invocation to mirror as a ``function_call`` item."""

    line_no: int
    call_id: str
    name: str
    arguments: str
    response_id: str


@dataclass
class _FunctionOutputPost:
    """A tool result to mirror as a ``function_call_output`` item."""

    line_no: int
    call_id: str
    output: str
    response_id: str


@dataclass
class _StatusPost:
    """A session-status edge (``running`` / ``idle``).

    ``response_id`` names the turn on a ``running`` edge; the server records it
    as the session's ``active_response_id``, which is what keeps a mid-turn
    reconnect rendering the forwarded tool cards LIVE (the turn-start edge is
    not replayed on the SSE stream).
    """

    line_no: int
    status: str
    response_id: str | None = None


#: Anything the planner asks the loop to POST.
_Post = _MessagePost | _ReasoningPost | _FunctionCallPost | _FunctionOutputPost | _StatusPost


def clear_kimi_bridge_state(bridge_dir: Path) -> None:
    """Drop any stale forwarder state so a new terminal starts a fresh tail.

    Mirrors ``cursor_native_forwarder.clear_cursor_bridge_state``: without this,
    a re-created terminal would resume the prior session's line offset against a
    different wire log.
    """
    with contextlib.suppress(OSError):
        (bridge_dir / _STATE_FILE).unlink()


def _read_state(bridge_dir: Path) -> _ForwardState | None:
    try:
        raw = (bridge_dir / _STATE_FILE).read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    wire_path = data.get("wire_path")
    last_line = data.get("last_line")
    if isinstance(wire_path, str) and isinstance(last_line, int):
        return _ForwardState(wire_path=wire_path, last_line=last_line)
    return None


def _write_state(bridge_dir: Path, state: _ForwardState) -> None:
    payload = {"wire_path": state.wire_path, "last_line": state.last_line}
    tmp = bridge_dir / (_STATE_FILE + ".tmp")
    with contextlib.suppress(OSError):
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(bridge_dir / _STATE_FILE)


def _workdirs_for_sessions(kimi_home: Path) -> dict[str, str]:
    """Map each session dir → its ``workDir`` from ``session_index.jsonl``.

    Returns ``{}`` when the index is absent/unreadable (a brand-new home before
    kimi has written any session).
    """
    index = kimi_home / "session_index.jsonl"
    mapping: dict[str, str] = {}
    try:
        text = index.read_text(encoding="utf-8")
    except OSError:
        return mapping
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            session_dir = row.get("sessionDir")
            work_dir = row.get("workDir")
            if isinstance(session_dir, str) and isinstance(work_dir, str):
                mapping[session_dir] = work_dir
    return mapping


def _discover_wire(kimi_home: Path, workspace: str, launch_epoch_ms: int) -> Path | None:
    """Locate the wire log for *workspace*'s newest session created at/after launch.

    Globs ``sessions/*/session_*/agents/main/wire.jsonl`` under *kimi_home*,
    keeps only sessions whose ``session_index`` ``workDir`` matches *workspace*
    (when the index lists them), and returns the most-recently-modified wire log
    whose mtime is at/after ``launch_epoch_ms`` (minus skew). Returns ``None``
    until kimi has created the session.
    """
    sessions_root = kimi_home / "sessions"
    if not sessions_root.exists():
        return None
    workdirs = _workdirs_for_sessions(kimi_home)
    floor_s = (launch_epoch_ms - _DISCOVER_SKEW_MS) / 1000.0
    best: tuple[float, Path] | None = None
    for wire in sessions_root.glob("*/session_*/agents/main/wire.jsonl"):
        # session_index keys on the session dir (…/<wd_…>/<session_…>).
        session_dir = str(wire.parent.parent.parent)
        work_dir = workdirs.get(session_dir)
        # When the index doesn't list it yet, fall back to recency alone — a
        # freshly created session may not be indexed until its first turn.
        if work_dir is not None and work_dir != workspace:
            continue
        try:
            mtime = wire.stat().st_mtime
        except OSError:
            continue
        if mtime < floor_s:
            continue
        if best is None or mtime > best[0]:
            best = (mtime, wire)
    return best[1] if best is not None else None


def _input_text(blocks: object) -> str:
    """Concatenate the ``text`` of an ``input`` / ``content`` block list."""
    if not isinstance(blocks, list):
        return ""
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _event_turn_id(event: dict[str, object]) -> str | None:
    """Return the event's ``turnId`` as a string, or ``None`` when absent."""
    turn_id = event.get("turnId")
    if isinstance(turn_id, str) and turn_id:
        return turn_id
    if isinstance(turn_id, int):
        return str(turn_id)
    return None


def _turn_response_id(turn_id: str) -> str:
    """The shared per-turn response id (groups a turn's items + status edges)."""
    return f"kimi:turn:{turn_id}"


def _response_id(state: _TurnState, event: dict[str, object], line_no: int) -> str:
    """Per-turn response id, or a per-event fallback when no turn is known.

    Assistant text and tool events group under ``kimi:turn:<turnId>`` so they
    share the turn's ``running`` edge and render as one live response. A
    ``tool.result`` has no ``turnId`` and leans on the remembered turn; if even
    that is missing (e.g. a mid-turn restart resumed past ``step.begin``), fall
    back to a per-event id so the item still posts, just ungrouped.
    """
    if state.turn_id is not None:
        return _turn_response_id(state.turn_id)
    uuid = event.get("uuid")
    if isinstance(uuid, str) and uuid:
        return f"kimi:{uuid}"
    return f"kimi:line:{line_no}"


def _tool_output_text(result: object) -> str:
    """Extract the mirrored text from a ``tool.result`` ``result`` payload."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        output = result.get("output")
        if isinstance(output, str):
            return output
        return json.dumps(result, ensure_ascii=True)
    return ""


def _plan_row(
    line_no: int, row: dict[str, object], state: _TurnState
) -> tuple[list[_Post], _TurnState]:
    """Translate one wire row into Omnigent posts, threading turn state.

    Pure: returns the posts to emit and the NEXT turn state. The caller commits
    the returned state only after the posts land, so a POST failure that retries
    the row cannot lose a ``running`` edge or double-advance the turn.
    """
    row_type = row.get("type")

    if row_type == "turn.prompt":
        origin = row.get("origin")
        if isinstance(origin, dict) and origin.get("kind") != "user":
            return [], state
        text = _input_text(row.get("input"))
        if not text:
            return [], state
        # The user bubble keeps its own per-line id: turn.prompt carries no
        # turnId, and the message need not join the turn's response group — it
        # only has to precede the assistant items, which wire order guarantees.
        return [_MessagePost(line_no, "user", text, f"kimi:turn:{line_no}")], state

    if row_type == "turn.cancel":
        if state.running:
            rid = _turn_response_id(state.turn_id) if state.turn_id is not None else None
            return [_StatusPost(line_no, _STATUS_IDLE, rid)], _TurnState()
        return [], state

    if row_type != "context.append_loop_event":
        return [], state

    event = row.get("event")
    if not isinstance(event, dict):
        return [], state
    etype = event.get("type")

    # Any event carrying turnId (re-)establishes the active turn — this also
    # recovers the turn after a mid-turn restart resumed past ``step.begin``.
    turn_id = _event_turn_id(event)
    if turn_id is not None:
        state = replace(state, turn_id=turn_id)

    if etype == "step.begin":
        if state.turn_id is not None and not state.running:
            edge = _StatusPost(line_no, _STATUS_RUNNING, _turn_response_id(state.turn_id))
            return [edge], replace(state, running=True)
        return [], state

    if etype == "step.end":
        if event.get("finishReason") == _FINISH_TOOL_USE:
            return [], state  # paused for a tool; the turn continues
        posts: list[_Post] = []
        if state.running:
            rid = _turn_response_id(state.turn_id) if state.turn_id is not None else None
            posts = [_StatusPost(line_no, _STATUS_IDLE, rid)]
        return posts, _TurnState()

    if etype == "content.part":
        part = event.get("part")
        if not isinstance(part, dict):
            return [], state
        part_type = part.get("type")
        if part_type == "text":
            text = part.get("text")
            if not isinstance(text, str) or not text:
                return [], state
            rid = _response_id(state, event, line_no)
            return [_MessagePost(line_no, "assistant", text, rid)], state
        if part_type == "think":
            # Reasoning lives in ``part["think"]`` (not ``part["text"]``); mirror
            # it as a transient reasoning delta so the web UI paints a thinking
            # block — the kimi analogue of codex-native's #1254 reasoning fix.
            think = part.get("think")
            if not isinstance(think, str) or not think:
                return [], state
            return [_ReasoningPost(line_no, think)], state
        return [], state

    if etype == "tool.call":
        call_id = event.get("toolCallId") or event.get("uuid")
        name = event.get("name")
        if not isinstance(call_id, str) or not isinstance(name, str):
            return [], state
        args = event.get("args")
        arguments = json.dumps(args if isinstance(args, dict) else {}, ensure_ascii=True)
        rid = _response_id(state, event, line_no)
        return [_FunctionCallPost(line_no, call_id, name, arguments, rid)], state

    if etype == "tool.result":
        call_id = event.get("toolCallId") or event.get("parentUuid")
        if not isinstance(call_id, str):
            return [], state
        output = _tool_output_text(event.get("result"))
        rid = _response_id(state, event, line_no)
        return [_FunctionOutputPost(line_no, call_id, output, rid)], state

    return [], state


def _read_new_rows(wire_path: Path, last_line: int) -> list[tuple[int, dict[str, object]]]:
    """Parse wire-log lines beyond *last_line* into ``(line_no, row)`` pairs.

    The wire log is append-only JSONL, so a line count is a stable high-water
    mark. Non-JSON / non-object lines are dropped (they carry no line number
    forward on their own — the cursor advances as later rows are processed).
    """
    try:
        lines = wire_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[tuple[int, dict[str, object]]] = []
    for idx in range(last_line, len(lines)):
        line = lines[idx].strip()
        if not line or not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append((idx, row))
    return rows


def _post_body(post: _Post, agent_name: str) -> dict[str, object]:
    """Build the ``/events`` request body for one planned post."""
    if isinstance(post, _StatusPost):
        status_data: dict[str, object] = {"status": post.status}
        if post.response_id is not None:
            status_data["response_id"] = post.response_id
        return {"type": _EXTERNAL_STATUS, "data": status_data}
    if isinstance(post, _ReasoningPost):
        # One-shot delta with ``started`` opens a reasoning block. Kimi persists
        # completed think parts (not streamed deltas), so one delta per part.
        return {"type": _EXTERNAL_REASONING_DELTA, "data": {"delta": post.delta, "started": True}}
    if isinstance(post, _MessagePost):
        content_type = "input_text" if post.role == "user" else "output_text"
        item_data: dict[str, object] = {
            "role": post.role,
            "content": [{"type": content_type, "text": post.text}],
        }
        if post.role == "assistant":
            item_data["agent"] = agent_name
        data: dict[str, object] = {
            "item_type": "message",
            "item_data": item_data,
            "response_id": post.response_id,
        }
        return {"type": _EXTERNAL_ITEM, "data": data}
    if isinstance(post, _FunctionCallPost):
        data = {
            "item_type": "function_call",
            "item_data": {
                "agent": agent_name,
                "name": post.name,
                "arguments": post.arguments,
                "call_id": post.call_id,
            },
            "response_id": post.response_id,
        }
        return {"type": _EXTERNAL_ITEM, "data": data}
    # _FunctionOutputPost
    data = {
        "item_type": "function_call_output",
        "item_data": {"call_id": post.call_id, "output": post.output},
        "response_id": post.response_id,
    }
    return {"type": _EXTERNAL_ITEM, "data": data}


async def _emit_post(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    headers: dict[str, str],
    session_id: str,
    post: _Post,
    agent_name: str,
) -> None:
    """POST one planned event to the session's ``/events`` endpoint."""
    url = f"{base_url.rstrip('/')}/v1/sessions/{session_id}/events"
    resp = await client.post(url, headers=headers, json=_post_body(post, agent_name))
    resp.raise_for_status()


async def forward_kimi_wire_to_session(
    *,
    base_url: str,
    headers: dict[str, str],
    session_id: str,
    bridge_dir: Path,
    kimi_home: Path,
    workspace: str,
    launch_epoch_ms: int,
    agent_name: str = "kimi-native-ui",
) -> None:
    """Poll the kimi session wire log and mirror new turns into the chat.

    Runs until cancelled. Discovers the wire log lazily (kimi writes it after the
    first turn), then tails it, planning each new row into user/assistant
    messages, reasoning deltas, ``function_call`` items and ``running`` / ``idle``
    status edges, and persisting the line offset after every processed row.
    """
    state = _read_state(bridge_dir)
    wire_path = Path(state.wire_path) if state is not None else None
    last_line = state.last_line if state is not None else 0
    turn = _TurnState()
    async with httpx.AsyncClient(timeout=15.0) as client:
        while True:
            if wire_path is None or not wire_path.exists():
                discovered = await asyncio.to_thread(
                    _discover_wire, kimi_home, workspace, launch_epoch_ms
                )
                if discovered is not None and discovered != wire_path:
                    wire_path = discovered
                    last_line = 0
                    turn = _TurnState()
                    _write_state(bridge_dir, _ForwardState(str(wire_path), last_line))
            if wire_path is not None and wire_path.exists():
                rows = await asyncio.to_thread(_read_new_rows, wire_path, last_line)
                for line_no, row in rows:
                    posts, next_turn = _plan_row(line_no, row, turn)
                    delivered = True
                    for post in posts:
                        try:
                            await _emit_post(
                                client,
                                base_url=base_url,
                                headers=headers,
                                session_id=session_id,
                                post=post,
                                agent_name=agent_name,
                            )
                        except httpx.HTTPError as exc:
                            _logger.warning("kimi forwarder: POST failed (will retry): %s", exc)
                            delivered = False
                            break
                    if not delivered:
                        break
                    # Commit turn state only after the row's posts land, so a
                    # retried row re-plans against unchanged state.
                    turn = next_turn
                    last_line = line_no + 1
                    _write_state(bridge_dir, _ForwardState(str(wire_path), last_line))
            await asyncio.sleep(_POLL_INTERVAL_S)


async def supervise_kimi_forwarder(
    *,
    base_url: str,
    headers: dict[str, str],
    session_id: str,
    bridge_dir: Path,
    kimi_home: Path,
    workspace: str,
    launch_epoch_ms: int,
    agent_name: str = "kimi-native-ui",
) -> None:
    """Run :func:`forward_kimi_wire_to_session` with restart-on-crash backoff.

    Propagates :class:`asyncio.CancelledError` cleanly (terminal teardown), but
    restarts on any other exception with exponential backoff — mirrors
    ``cursor_native_forwarder.supervise_cursor_forwarder``.
    """
    backoff = _BACKOFF_INITIAL_S
    while True:
        try:
            await forward_kimi_wire_to_session(
                base_url=base_url,
                headers=headers,
                session_id=session_id,
                bridge_dir=bridge_dir,
                kimi_home=kimi_home,
                workspace=workspace,
                launch_epoch_ms=launch_epoch_ms,
                agent_name=agent_name,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            _logger.exception("kimi forwarder crashed for session %s; restarting", session_id)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _BACKOFF_MAX_S)
        else:
            return
