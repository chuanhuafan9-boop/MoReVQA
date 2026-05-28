from __future__ import annotations

from engine.memory import ExternalMemory
from engine.models.base import DetectorBackend, ImageTextScorerBackend, VisionLanguageBackend
from engine.schemas import Detection, GroundingRecord, VideoFrame
from engine.video import select_frames


class GroundingAPI:
    allowed_tools = {"localize", "verify_action", "truncate", "noop"}

    def __init__(
        self,
        memory: ExternalMemory,
        frames: list[VideoFrame],
        detector: DetectorBackend,
        image_text_scorer: ImageTextScorerBackend,
        vqa_model: VisionLanguageBackend,
        top_k: int = 6,
    ) -> None:
        self.memory = memory
        self.frames = frames
        self.detector = detector
        self.image_text_scorer = image_text_scorer
        self.vqa_model = vqa_model
        self.top_k = top_k
        self._last_scores: dict[int, float] = {}
        self._last_detections: list[Detection] = []

    def localize(self, query: str, top_k: int | None = None) -> dict:
        query = str(query).strip()
        top_k = self.top_k if top_k is None else int(top_k)
        frames = select_frames(self.frames, self.memory.frame_ids)
        all_detections: list[Detection] = []
        scores: dict[int, float] = {}
        for frame in frames:
            detections = self.detector.detect(frame, query)
            all_detections.extend(detections)
            det_score = max((item.score for item in detections), default=0.0)
            txt_score = self.image_text_scorer.score(frame, query)
            scores[frame.frame_id] = max(det_score, txt_score)
        ranked = _top_frame_ids(scores, top_k)
        self._last_scores = scores
        self._last_detections = all_detections
        record = GroundingRecord(
            query=query,
            frame_ids=ranked,
            scores=scores,
            detections=all_detections,
            source="grounding.localize",
        )
        self.memory.add_grounding(record)
        if ranked:
            self.memory.set_grounded_frame_ids(ranked, "grounding", f"localize:{query}")
        return {"query": query, "frame_ids": ranked, "scores": scores}

    def verify_action(self, action: str, top_k: int | None = None) -> dict:
        action = str(action).strip()
        top_k = self.top_k if top_k is None else int(top_k)
        frames = select_frames(self.frames, self.memory.frame_ids)
        scores: dict[int, float] = {}
        for frame in frames:
            text_score = self.image_text_scorer.score(frame, action)
            answer = self.vqa_model.answer(
                frame,
                f"Is the following event visible: {action}? Answer yes or no.",
            )
            yes_bonus = 0.35 if _is_yes(answer) else 0.0
            scores[frame.frame_id] = min(1.0, text_score + yes_bonus)
        ranked = _top_frame_ids(scores, top_k)
        self._last_scores = scores
        record = GroundingRecord(
            query=action,
            frame_ids=ranked,
            scores=scores,
            detections=[],
            source="grounding.verify_action",
        )
        self.memory.add_grounding(record)
        if ranked:
            self.memory.set_grounded_frame_ids(ranked, "grounding", f"verify_action:{action}")
        return {"action": action, "frame_ids": ranked, "scores": scores}

    def truncate(self, relation: str = "none", margin: int = 0) -> dict:
        relation = (relation or "none").strip().lower()
        frame_ids = list(self.memory.frame_ids)
        anchors = list(self.memory.grounded_frame_ids)
        if not frame_ids or not anchors or relation in {"none", "", "and"}:
            return {"relation": relation, "frame_ids": frame_ids}
        lo, hi = min(anchors), max(anchors)
        if relation == "before":
            selected = [frame_id for frame_id in frame_ids if frame_id <= lo + margin]
        elif relation == "after":
            selected = [frame_id for frame_id in frame_ids if frame_id >= hi - margin]
        elif relation in {"during", "while", "when"}:
            selected = [frame_id for frame_id in frame_ids if lo - margin <= frame_id <= hi + margin]
        else:
            selected = frame_ids
        if selected:
            self.memory.set_frame_ids(selected, "grounding", f"truncate:{relation}")
            self.memory.set_grounded_frame_ids(selected, "grounding", f"truncate:{relation}")
        return {"relation": relation, "frame_ids": selected}

    def noop(self) -> dict:
        return {"noop": True}


def _top_frame_ids(scores: dict[int, float], top_k: int) -> list[int]:
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [frame_id for frame_id, _ in ranked[:top_k]]


def _is_yes(answer: str) -> bool:
    answer = (answer or "").strip().lower()
    return answer.startswith("yes") or answer in {"true", "visible", "present"}
