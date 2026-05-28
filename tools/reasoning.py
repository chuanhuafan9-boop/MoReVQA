from __future__ import annotations

from engine.memory import ExternalMemory
from engine.models.base import OCRBackend, VisionLanguageBackend
from engine.schemas import QARecord, VideoFrame
from engine.video import select_frames, uniform_sample_frames


class ReasoningAPI:
    allowed_tools = {"vqa", "noop"}

    def __init__(
        self,
        memory: ExternalMemory,
        frames: list[VideoFrame],
        vqa_model: VisionLanguageBackend,
        ocr_model: OCRBackend | None = None,
        max_frames_per_question: int = 8,
    ) -> None:
        self.memory = memory
        self.frames = frames
        self.vqa_model = vqa_model
        self.ocr_model = ocr_model
        self.max_frames_per_question = max_frames_per_question

    def vqa(
        self,
        question_or_questions: str | list[str],
        frame_ids: list[int] | None = None,
    ) -> dict:
        questions = (
            [question_or_questions]
            if isinstance(question_or_questions, str)
            else [str(item) for item in question_or_questions]
        )
        selected_ids = frame_ids or self.memory.active_frame_ids()
        frames = select_frames(self.frames, selected_ids)
        frames = uniform_sample_frames(frames, self.max_frames_per_question)
        outputs: list[dict] = []
        if self.memory.require_ocr and self.ocr_model is not None:
            for frame in frames:
                text = self.ocr_model.read_text(frame)
                if text:
                    record = QARecord(
                        question="OCR visible text",
                        answer=text,
                        frame_id=frame.frame_id,
                        source="reasoning.ocr",
                    )
                    self.memory.add_qa(record)
                    outputs.append(
                        {
                            "question": "OCR visible text",
                            "answer": text,
                            "frame_id": frame.frame_id,
                        }
                    )
        for question in questions:
            for frame in frames:
                answer = self.vqa_model.answer(frame, question)
                record = QARecord(
                    question=question,
                    answer=answer,
                    frame_id=frame.frame_id,
                    source="reasoning.vqa",
                )
                self.memory.add_qa(record)
                outputs.append(
                    {"question": question, "answer": answer, "frame_id": frame.frame_id}
                )
        return {"outputs": outputs}

    def noop(self) -> dict:
        return {"noop": True}
