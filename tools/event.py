from __future__ import annotations

from engine.memory import ExternalMemory


class EventParsingAPI:
    allowed_tools = {"trim", "parse_event", "classify", "require_ocr", "noop"}

    def __init__(self, memory: ExternalMemory, keep_ratio: float = 0.4) -> None:
        self.memory = memory
        self.keep_ratio = keep_ratio

    def trim(self, hint: str = "none", keep_ratio: float | None = None) -> dict:
        hint = (hint or "none").strip().lower()
        keep_ratio = self.keep_ratio if keep_ratio is None else float(keep_ratio)
        frame_ids = list(self.memory.frame_ids)
        if not frame_ids or hint in {"none", "all", ""}:
            self.memory.temporal_hint = hint if hint else "none"
            return {"hint": hint, "frame_ids": frame_ids}
        count = max(1, round(len(frame_ids) * keep_ratio))
        if hint in {"beginning", "start", "early", "first"}:
            selected = frame_ids[:count]
            normalized = "beginning"
        elif hint in {"end", "ending", "late", "near end", "last"}:
            selected = frame_ids[-count:]
            normalized = "end"
        elif hint in {"middle", "mid"}:
            center = len(frame_ids) // 2
            half = count // 2
            selected = frame_ids[max(0, center - half) : min(len(frame_ids), center - half + count)]
            normalized = "middle"
        else:
            selected = frame_ids
            normalized = hint
        self.memory.temporal_hint = normalized
        self.memory.set_frame_ids(selected, "event_parsing", f"trim:{normalized}")
        return {"hint": normalized, "frame_ids": selected}

    def parse_event(
        self,
        conjunction: str = "none",
        event: str | list[str] | None = None,
        rewritten_question: str | None = None,
    ) -> dict:
        conjunction = (conjunction or "none").strip().lower()
        self.memory.conjunction = conjunction
        events = event if isinstance(event, list) else [event] if event else []
        for item in events:
            item = str(item).strip()
            if item and item not in self.memory.event_queue:
                self.memory.event_queue.append(item)
        if rewritten_question:
            self.memory.working_question = rewritten_question.strip()
        return {
            "conjunction": self.memory.conjunction,
            "event_queue": list(self.memory.event_queue),
            "working_question": self.memory.working_question,
        }

    def classify(self, qa_type: str = "unknown") -> dict:
        self.memory.qa_type = (qa_type or "unknown").strip().lower()
        return {"qa_type": self.memory.qa_type}

    def require_ocr(self, flag: bool | str = False) -> dict:
        if isinstance(flag, str):
            value = flag.strip().lower() in {"true", "yes", "1", "ocr", "required"}
        else:
            value = bool(flag)
        self.memory.require_ocr = value
        return {"require_ocr": value}

    def noop(self) -> dict:
        return {"noop": True}
