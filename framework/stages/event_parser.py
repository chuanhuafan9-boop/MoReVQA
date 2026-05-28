from __future__ import annotations

import re

from engine.config import MoreVQAConfig
from engine.utils import ProgramGenerator, ProgramInterpreter
from engine.memory import ExternalMemory
from engine.models.base import LLMBackend
from engine.schemas import ActionCall
from prompts.prompt_engineering import build_event_prompt
from tools.event import EventParsingAPI


class EventParsingStage:
    stage_name = "event_parsing"

    def __init__(self, llm: LLMBackend, config: MoreVQAConfig) -> None:
        self.generator = ProgramGenerator(llm)
        self.interpreter = ProgramInterpreter()
        self.keep_ratio = config.float("grounding", "temporal_keep_ratio", default=0.4)

    def run(self, memory: ExternalMemory) -> ExternalMemory:
        prompt = build_event_prompt(memory)
        calls, raw = self.generator.generate(prompt)
        if not calls:
            calls = heuristic_event_plan(memory.question)
            raw = "heuristic_event_plan"
        memory.set_plan(self.stage_name, calls, raw)
        api = EventParsingAPI(memory, keep_ratio=self.keep_ratio)
        self.interpreter.execute(self.stage_name, calls, api, memory)
        return memory


def heuristic_event_plan(question: str) -> list[ActionCall]:
    q = question.strip()
    lower = q.lower()
    calls: list[ActionCall] = []
    hint = _temporal_hint(lower)
    if hint != "none":
        calls.append(ActionCall("trim", [hint]))
    conjunction = _conjunction(lower)
    event = _event_phrase(q, conjunction)
    rewritten = _rewrite_question(q, conjunction)
    calls.append(ActionCall("parse_event", [conjunction, event, rewritten]))
    calls.append(ActionCall("classify", [_qa_type(lower)]))
    calls.append(ActionCall("require_ocr", [_needs_ocr(lower)]))
    return calls


def _temporal_hint(lower: str) -> str:
    if any(token in lower for token in ["near the end", "at the end", "ending", "last "]):
        return "end"
    if any(token in lower for token in ["beginning", "at the start", "first ", "initially"]):
        return "beginning"
    if "middle" in lower:
        return "middle"
    return "none"


def _conjunction(lower: str) -> str:
    for item in ["before", "after", "during", "while", "when"]:
        if re.search(rf"\b{item}\b", lower):
            return item
    return "none"


def _event_phrase(question: str, conjunction: str) -> str:
    if conjunction != "none":
        parts = re.split(rf"\b{re.escape(conjunction)}\b", question, flags=re.IGNORECASE, maxsplit=1)
        if len(parts) == 2 and parts[1].strip():
            return _clean_event(parts[1])
    cleaned = re.sub(r"^(what|why|how|where|who|when|which)\b", "", question, flags=re.IGNORECASE)
    return _clean_event(cleaned)


def _rewrite_question(question: str, conjunction: str) -> str:
    if conjunction in {"before", "after", "during", "while", "when"}:
        parts = re.split(rf"\b{re.escape(conjunction)}\b", question, flags=re.IGNORECASE, maxsplit=1)
        if parts and parts[0].strip().endswith("?"):
            return parts[0].strip()
        if parts and parts[0].strip():
            return parts[0].strip() + "?"
    return question.strip()


def _qa_type(lower: str) -> str:
    stripped = lower.strip()
    for prefix in ["why", "how", "where", "who", "when", "what"]:
        if stripped.startswith(prefix):
            return prefix
    if stripped.startswith(("is ", "are ", "was ", "were ", "does ", "do ", "did ", "can ")):
        return "yes_no"
    if "how many" in stripped or "number of" in stripped:
        return "count"
    return "unknown"


def _needs_ocr(lower: str) -> bool:
    return any(word in lower for word in ["text", "word", "written", "sign", "label", "title"])


def _clean_event(text: str) -> str:
    text = re.sub(r"[?.!]+$", "", text.strip())
    text = re.sub(r"\b(near the end|at the end|in the beginning|at the start)\b", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text or "main visible event"
