from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from engine.config import MoreVQAConfig
from engine.memory import ExternalMemory
from engine.models.registry import ModelBundle, build_model_bundle
from engine.schemas import PredictionResult, VideoFrame, to_jsonable
from engine.video import load_video_frames, uniform_sample_frames
from framework.stages import EventParsingStage, GroundingStage, PredictionStage, ReasoningStage


@dataclass(slots=True)
class MoReVQAOutput:
    prediction: PredictionResult
    memory: ExternalMemory

    def to_dict(self) -> dict:
        return {
            "prediction": to_jsonable(self.prediction),
            "memory": self.memory.to_dict(),
        }

    def save_trace(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


class MoReVQA:
    def __init__(self, config: MoreVQAConfig, models: ModelBundle | None = None) -> None:
        self.config = config.merged_with_defaults()
        self.models = models or build_model_bundle(self.config)
        self.event_stage = EventParsingStage(self.models.llm, self.config)
        self.grounding_stage = GroundingStage(
            llm=self.models.llm,
            detector=self.models.detector,
            image_text_scorer=self.models.image_text_scorer,
            vqa_model=self.models.vqa,
            config=self.config,
        )
        self.reasoning_stage = ReasoningStage(
            llm=self.models.llm,
            captioner=self.models.captioner,
            vqa_model=self.models.vqa,
            ocr_model=self.models.ocr,
            config=self.config,
        )
        self.prediction_stage = PredictionStage(self.models.llm, self.config)

    @classmethod
    def from_config_file(cls, path: str | Path, force_mock: bool = False) -> "MoReVQA":
        config = MoreVQAConfig.from_file(path).merged_with_defaults()
        models = build_model_bundle(config, force_mock=force_mock)
        return cls(config=config, models=models)

    def inference(
        self,
        question: str,
        video: str | Path | list[VideoFrame],
        options: list[str] | None = None,
    ) -> MoReVQAOutput:
        """Run the sole supported MoReVQA program on one video question."""

        if isinstance(video, list):
            return self.answer_frames(video, question, options, video_path="in-memory frames")
        return self.answer(video, question, options)

    def answer(
        self,
        video_path: str | Path,
        question: str,
        options: list[str] | None = None,
    ) -> MoReVQAOutput:
        frames = self._load_frames(video_path)
        return self.answer_frames(frames, question, options, video_path=video_path)

    def answer_frames(
        self,
        frames: list[VideoFrame],
        question: str,
        options: list[str] | None = None,
        video_path: str | Path | None = None,
    ) -> MoReVQAOutput:
        memory = ExternalMemory(
            question=question,
            options=options,
            video_path=str(video_path) if video_path is not None else None,
            frame_ids=[frame.frame_id for frame in frames],
        )
        self.event_stage.run(memory)
        self.grounding_stage.run(memory, frames)
        self.reasoning_stage.run(memory, frames)
        prediction = self.prediction_stage.run(memory)
        return MoReVQAOutput(prediction=prediction, memory=memory)

    def _load_frames(
        self,
        video_path: str | Path,
    ) -> list[VideoFrame]:
        frames = load_video_frames(
            video_path,
            sample_fps=self.config.float("video", "sample_fps", default=1.0),
            max_frames=self.config.get("video", "max_frames", default=None),
        )
        max_frames = self.config.get("video", "max_frames", default=None)
        if max_frames is not None:
            frames = uniform_sample_frames(frames, int(max_frames))
        return frames
