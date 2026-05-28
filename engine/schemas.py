from __future__ import annotations

"""Shared records exchanged by the execution engine and task framework."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class VideoFrame:
    frame_id: int
    timestamp: float
    image: Any
    source_path: str | None = None


@dataclass(slots=True)
class CaptionRecord:
    frame_id: int
    timestamp: float
    caption: str
    stage: str


@dataclass(slots=True)
class Detection:
    frame_id: int
    label: str
    score: float
    box: list[float] = field(default_factory=list)


@dataclass(slots=True)
class GroundingRecord:
    query: str
    frame_ids: list[int]
    scores: dict[int, float] = field(default_factory=dict)
    detections: list[Detection] = field(default_factory=list)
    source: str = "grounding"


@dataclass(slots=True)
class QARecord:
    question: str
    answer: str
    frame_id: int | None = None
    score: float | None = None
    source: str = "vqa"


@dataclass(slots=True)
class ActionCall:
    name: str
    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "args": self.args, "kwargs": self.kwargs}


@dataclass(slots=True)
class PredictionResult:
    answer: str
    raw_response: str
    confidence: float | None = None


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    return value
