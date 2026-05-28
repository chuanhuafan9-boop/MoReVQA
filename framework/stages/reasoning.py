from __future__ import annotations

from engine.config import MoreVQAConfig
from engine.utils import ProgramGenerator, ProgramInterpreter
from engine.memory import ExternalMemory
from engine.models.base import LLMBackend, OCRBackend, VisionLanguageBackend
from engine.schemas import ActionCall, CaptionRecord, VideoFrame
from engine.video import uniform_sample_frames
from prompts.prompt_engineering import build_reasoning_prompt
from tools.reasoning import ReasoningAPI


class ReasoningStage:
    stage_name = "reasoning"

    def __init__(
        self,
        llm: LLMBackend,
        captioner: VisionLanguageBackend,
        vqa_model: VisionLanguageBackend,
        ocr_model: OCRBackend,
        config: MoreVQAConfig,
    ) -> None:
        self.generator = ProgramGenerator(llm)
        self.interpreter = ProgramInterpreter()
        self.captioner = captioner
        self.vqa_model = vqa_model
        self.ocr_model = ocr_model
        self.context_frames = config.int("video", "context_frames", default=16)
        self.max_frames_per_question = config.int("grounding", "verify_top_k", default=8)

    def run(self, memory: ExternalMemory, frames: list[VideoFrame]) -> ExternalMemory:
        self._add_video_context(memory, frames)
        prompt = build_reasoning_prompt(memory)
        calls, raw = self.generator.generate(prompt)
        if not calls:
            calls = heuristic_reasoning_plan(memory)
            raw = "heuristic_reasoning_plan"
        memory.set_plan(self.stage_name, calls, raw)
        api = ReasoningAPI(
            memory=memory,
            frames=frames,
            vqa_model=self.vqa_model,
            ocr_model=self.ocr_model,
            max_frames_per_question=self.max_frames_per_question,
        )
        self.interpreter.execute(self.stage_name, calls, api, memory)
        return memory

    def _add_video_context(self, memory: ExternalMemory, frames: list[VideoFrame]) -> None:
        context_frames = uniform_sample_frames(frames, self.context_frames)
        seen = {(row.stage, row.frame_id) for row in memory.captions}
        for frame in context_frames:
            if ("context", frame.frame_id) in seen:
                continue
            caption = self.captioner.caption(frame)
            memory.add_caption(
                CaptionRecord(
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp,
                    caption=caption,
                    stage="context",
                )
            )


def heuristic_reasoning_plan(memory: ExternalMemory) -> list[ActionCall]:
    question = memory.working_question or memory.question
    qa_type = memory.qa_type
    supporting: list[str] = [question]
    if qa_type == "why":
        supporting.extend(
            [
                "What is the main actor doing in these frames?",
                "What surroundings or interactions explain the event?",
            ]
        )
    elif qa_type == "how":
        supporting.append("How is the visible action being performed?")
    elif qa_type == "where":
        supporting.append("Where does the relevant event take place?")
    elif qa_type == "who":
        supporting.append("Who is involved in the relevant event?")
    elif qa_type == "when":
        supporting.append("What happens before and after the relevant event?")
    elif qa_type in {"what", "description", "unknown"}:
        supporting.append("What is happening in the relevant frames?")
    return [ActionCall("vqa", [supporting])]
