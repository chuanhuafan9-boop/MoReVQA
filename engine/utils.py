from __future__ import annotations

import ast
import json
import re
from typing import Any

from engine.memory import ExternalMemory
from engine.schemas import ActionCall, to_jsonable


class PlanParseError(ValueError):
    """Raised when an LLM plan cannot be parsed into safe tool calls."""


class ToolExecutionError(RuntimeError):
    """Raised when a planned tool call cannot be dispatched."""


class ProgramGenerator:
    """Generate a tool program from a prompt, matching CLOVA's engine role."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def generate(self, prompt: str) -> tuple[list[ActionCall], str]:
        raw_program = self.llm.generate(prompt)
        return parse_action_plan(raw_program), raw_program


class ProgramInterpreter:
    """Execute one generated module program against exposed tools."""

    def execute(
        self,
        stage: str,
        actions: list[ActionCall],
        api: Any,
        memory: ExternalMemory,
        strict: bool = False,
    ) -> list[Any]:
        return execute_actions(stage, actions, api, memory, strict=strict)


def parse_action_plan(text: str) -> list[ActionCall]:
    """Parse a JSON or simple function-call plan into ActionCall objects."""

    if not text or not text.strip():
        return []
    json_payload = _extract_json(text)
    if json_payload is not None:
        return _actions_from_json(json_payload)
    return _actions_from_code_like(text)


def execute_actions(
    stage: str,
    actions: list[ActionCall],
    api: Any,
    memory: ExternalMemory,
    strict: bool = False,
) -> list[Any]:
    """Execute only white-listed actions exposed by a stage API object."""

    outputs: list[Any] = []
    allowed = set(getattr(api, "allowed_tools", ()))
    for call in actions:
        if allowed and call.name not in allowed:
            message = f"{stage}: tool is not allowed: {call.name}"
            memory.add_trace(stage, "tool_error", {"call": call.to_dict(), "error": message})
            if strict:
                raise ToolExecutionError(message)
            continue
        tool = getattr(api, call.name, None)
        if tool is None or not callable(tool):
            message = f"{stage}: tool does not exist: {call.name}"
            memory.add_trace(stage, "tool_error", {"call": call.to_dict(), "error": message})
            if strict:
                raise ToolExecutionError(message)
            continue
        try:
            result = tool(*call.args, **call.kwargs)
            outputs.append(result)
            memory.add_trace(stage, "tool_output", {"call": call.to_dict(), "result": result})
        except Exception as exc:
            memory.add_trace(
                stage,
                "tool_error",
                {"call": call.to_dict(), "error": f"{type(exc).__name__}: {exc}"},
            )
            if strict:
                raise
    return outputs


def _extract_json(text: str) -> Any | None:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL | re.IGNORECASE)
    if fenced:
        stripped = fenced.group(1).strip()
    candidates = [stripped]
    first_object = _balanced_slice(stripped, "{", "}")
    first_array = _balanced_slice(stripped, "[", "]")
    if first_object:
        candidates.append(first_object)
    if first_array:
        candidates.append(first_array)
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _balanced_slice(text: str, start_char: str, end_char: str) -> str | None:
    start = text.find(start_char)
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == start_char:
            depth += 1
        elif char == end_char:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _actions_from_json(payload: Any) -> list[ActionCall]:
    if isinstance(payload, dict):
        if "calls" in payload:
            payload = payload["calls"]
        elif "plan" in payload:
            payload = payload["plan"]
        elif "actions" in payload:
            payload = payload["actions"]
        else:
            payload = [payload]
    if not isinstance(payload, list):
        raise PlanParseError("Plan JSON must be a list or contain calls/actions/plan.")
    calls: list[ActionCall] = []
    for item in payload:
        if isinstance(item, str):
            calls.extend(_actions_from_code_like(item))
            continue
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("api") or item.get("tool") or item.get("function")
        if not name:
            continue
        args = item.get("args", [])
        kwargs = item.get("kwargs", {})
        arguments = item.get("arguments")
        if arguments is not None:
            if isinstance(arguments, list):
                args = arguments
            elif isinstance(arguments, dict):
                kwargs = arguments
            else:
                args = [arguments]
        if isinstance(args, dict) and not kwargs:
            kwargs = args
            args = []
        if not isinstance(args, list):
            args = [args]
        calls.append(ActionCall(name=str(name), args=args, kwargs=dict(kwargs or {})))
    return calls


def _actions_from_code_like(text: str) -> list[ActionCall]:
    code = _strip_code_fences(text)
    calls: list[ActionCall] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        lines = [line.strip() for line in code.splitlines() if line.strip()]
        for line in lines:
            try:
                calls.extend(_actions_from_code_like(line))
            except PlanParseError:
                continue
        return calls
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            calls.append(_action_from_call_node(node.value))
    return calls


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    return text


def _action_from_call_node(node: ast.Call) -> ActionCall:
    if isinstance(node.func, ast.Name):
        name = node.func.id
    elif isinstance(node.func, ast.Attribute):
        name = node.func.attr
    else:
        raise PlanParseError("Only simple tool calls are allowed.")
    args = [_safe_literal(arg) for arg in node.args]
    kwargs = {keyword.arg: _safe_literal(keyword.value) for keyword in node.keywords if keyword.arg}
    return ActionCall(name=name, args=args, kwargs=kwargs)


def _safe_literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception as exc:
        raise PlanParseError(f"Only literal call arguments are allowed: {ast.dump(node)}") from exc


def plan_to_json(calls: list[ActionCall]) -> str:
    return json.dumps([call.to_dict() for call in calls], ensure_ascii=False, indent=2, default=to_jsonable)
