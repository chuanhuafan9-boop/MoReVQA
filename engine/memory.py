from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from engine.schemas import (
    ActionCall,
    CaptionRecord,
    GroundingRecord,
    QARecord,
    to_jsonable,
)


@dataclass
class ExternalMemory:
    """Shared state passed through M1, M2, M3, and final prediction."""

    question: str
    options: list[str] | None = None
    video_path: str | None = None
    frame_ids: list[int] = field(default_factory=list)

    # M1 Event Parsing fills these language-only fields.
    working_question: str | None = None
    temporal_hint: str | None = None
    conjunction: str = "none"
    qa_type: str = "unknown"
    require_ocr: bool = False
    event_queue: list[str] = field(default_factory=list)

    # M2 Grounding stores the frames and visual evidence selected for reasoning.
    grounded_frame_ids: list[int] = field(default_factory=list)
    grounding: list[GroundingRecord] = field(default_factory=list)

    # M3 Reasoning stores context captions and targeted VQA sub-question answers.
    captions: list[CaptionRecord] = field(default_factory=list)
    reasoning_outputs: list[QARecord] = field(default_factory=list)

    # Program plans and traces are saved so every paper stage can be inspected later.
    plans: dict[str, list[ActionCall]] = field(default_factory=dict)
    raw_plans: dict[str, str] = field(default_factory=dict)
    traces: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.working_question is None:
            self.working_question = self.question
        if self.options is not None:
            self.options = [str(option) for option in self.options]

    def active_frame_ids(self) -> list[int]:
        return list(self.grounded_frame_ids or self.frame_ids)

    def set_plan(self, stage: str, calls: list[ActionCall], raw: str) -> None:
        self.plans[stage] = calls
        self.raw_plans[stage] = raw
        self.add_trace(stage, "plan", {"calls": [call.to_dict() for call in calls], "raw": raw})

    def add_trace(self, stage: str, kind: str, payload: Any) -> None:
        self.traces.append({"stage": stage, "kind": kind, "payload": to_jsonable(payload)})

    def set_frame_ids(self, frame_ids: list[int], stage: str, reason: str) -> None:
        self.frame_ids = _unique_sorted(frame_ids)
        self.add_trace(stage, "frame_ids", {"reason": reason, "frame_ids": self.frame_ids})

    def set_grounded_frame_ids(self, frame_ids: list[int], stage: str, reason: str) -> None:
        self.grounded_frame_ids = _unique_sorted(frame_ids)
        self.add_trace(
            stage,
            "grounded_frame_ids",
            {"reason": reason, "frame_ids": self.grounded_frame_ids},
        )

    def add_caption(self, record: CaptionRecord) -> None:
        self.captions.append(record)
        self.add_trace(record.stage, "caption", record)

    def add_grounding(self, record: GroundingRecord) -> None:
        self.grounding.append(record)
        self.add_trace(record.source, "grounding", record)

    def add_qa(self, record: QARecord) -> None:
        self.reasoning_outputs.append(record)
        self.add_trace(record.source, "qa", record)

    def text_summary(self, max_captions: int = 80, max_qas: int = 80) -> str:
        lines: list[str] = []
        lines.append(f"Original question: {self.question}")
        if self.working_question and self.working_question != self.question:
            lines.append(f"Rewritten question: {self.working_question}")
        if self.options:
            lines.append("Candidate answers:")
            for idx, option in enumerate(self.options, start=1):
                lines.append(f"{idx}. {option}")
        lines.append(f"QA type: {self.qa_type}")
        lines.append(f"Temporal hint: {self.temporal_hint or 'none'}")
        lines.append(f"Conjunction: {self.conjunction or 'none'}")
        lines.append(f"Require OCR: {self.require_ocr}")
        if self.event_queue:
            lines.append("Parsed events:")
            for event in self.event_queue:
                lines.append(f"- {event}")
        if self.grounded_frame_ids:
            lines.append(f"Grounded frame ids: {self.grounded_frame_ids}")
        if self.grounding:
            lines.append("Grounding records:")
            for item in self.grounding[-10:]:
                lines.append(f"- {item.query}: {item.frame_ids}")
        if self.captions:
            lines.append("Video captions:")
            for caption in sorted(self.captions, key=lambda row: (row.timestamp, row.frame_id))[
                -max_captions:
            ]:
                lines.append(
                    f"[frame {caption.frame_id:>5} | {caption.timestamp:>7.2f}s] "
                    f"{caption.caption}"
                )
        if self.reasoning_outputs:
            lines.append("Reasoning outputs:")
            for qa in self.reasoning_outputs[-max_qas:]:
                frame = "" if qa.frame_id is None else f"[frame {qa.frame_id}] "
                lines.append(f"{frame}Q: {qa.question} A: {qa.answer}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(
            {
                "question": self.question,
                "options": self.options,
                "video_path": self.video_path,
                "frame_ids": self.frame_ids,
                "working_question": self.working_question,
                "temporal_hint": self.temporal_hint,
                "conjunction": self.conjunction,
                "qa_type": self.qa_type,
                "require_ocr": self.require_ocr,
                "event_queue": self.event_queue,
                "grounded_frame_ids": self.grounded_frame_ids,
                "captions": self.captions,
                "grounding": self.grounding,
                "reasoning_outputs": self.reasoning_outputs,
                "plans": self.plans,
                "raw_plans": self.raw_plans,
                "traces": self.traces,
            }
        )


def _unique_sorted(frame_ids: list[int]) -> list[int]:
    return sorted({int(frame_id) for frame_id in frame_ids})
