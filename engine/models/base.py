from __future__ import annotations

from abc import ABC, abstractmethod

from engine.schemas import Detection, VideoFrame


class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> str:
        raise NotImplementedError


class VisionLanguageBackend(ABC):
    @abstractmethod
    def caption(self, frame: VideoFrame, prompt: str | None = None) -> str:
        raise NotImplementedError

    @abstractmethod
    def answer(self, frame: VideoFrame, question: str) -> str:
        raise NotImplementedError


class DetectorBackend(ABC):
    @abstractmethod
    def detect(self, frame: VideoFrame, label: str) -> list[Detection]:
        raise NotImplementedError


class ImageTextScorerBackend(ABC):
    @abstractmethod
    def score(self, frame: VideoFrame, text: str) -> float:
        raise NotImplementedError


class OCRBackend(ABC):
    @abstractmethod
    def read_text(self, frame: VideoFrame) -> str:
        raise NotImplementedError
