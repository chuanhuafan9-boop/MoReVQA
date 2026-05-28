from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, TypeVar

from engine.config import MoreVQAConfig
from engine.models.adapters import (
    CLIPScorer,
    FallbackDetector,
    FallbackImageTextScorer,
    FallbackLLM,
    FallbackOCR,
    FallbackVisionLanguageModel,
    MockDetector,
    MockImageTextScorer,
    MockLLM,
    MockOCR,
    MockVisionLanguageModel,
    OpenAIClipRN50Scorer,
    OpenAICompatibleLLM,
    OwlVitDetector,
    PaliGemma2PromptedOCR,
    PaliGemma2VisionLanguageModel,
    TransformersLLM,
    TransformersVisionLanguageModel,
)
from engine.models.base import (
    DetectorBackend,
    ImageTextScorerBackend,
    LLMBackend,
    OCRBackend,
    VisionLanguageBackend,
)


@dataclass(slots=True)
class ModelBundle:
    llm: LLMBackend
    captioner: VisionLanguageBackend
    vqa: VisionLanguageBackend
    detector: DetectorBackend
    image_text_scorer: ImageTextScorerBackend
    ocr: OCRBackend


T = TypeVar("T")


def build_model_bundle(config: MoreVQAConfig, force_mock: bool = False) -> ModelBundle:
    config = config.merged_with_defaults()
    if force_mock:
        return ModelBundle(
            llm=MockLLM(),
            captioner=MockVisionLanguageModel(),
            vqa=MockVisionLanguageModel(),
            detector=MockDetector(),
            image_text_scorer=MockImageTextScorer(),
            ocr=MockOCR(),
        )
    mock_on_missing = config.bool("runtime", "mock_on_missing", default=True)
    device = str(config.get("runtime", "device", default="auto"))
    dtype = str(config.get("runtime", "dtype", default="auto"))
    llm = _safe_build(lambda: _build_llm(config, device, dtype, force_mock), MockLLM, mock_on_missing)
    captioner_section = config.section("vision_language", "captioner")
    vqa_section = config.section("vision_language", "vqa")
    captioner = _safe_build(
        lambda: _build_vlm(captioner_section, device, dtype),
        MockVisionLanguageModel,
        mock_on_missing,
    )
    if _same_paligemma_model(captioner_section, vqa_section):
        vqa = captioner
    else:
        vqa = _safe_build(
            lambda: _build_vlm(vqa_section, device, dtype),
            MockVisionLanguageModel,
            mock_on_missing,
        )
    detector = _safe_build(
        lambda: _build_detector(config.section("grounding", "detector"), device, dtype),
        MockDetector,
        mock_on_missing,
    )
    image_text_scorer = _safe_build(
        lambda: _build_scorer(config.section("grounding", "image_text_scorer"), device, dtype),
        MockImageTextScorer,
        mock_on_missing,
    )
    ocr = _safe_build(
        lambda: _build_ocr(config.section("ocr"), vqa),
        MockOCR,
        mock_on_missing,
    )
    if mock_on_missing and not force_mock:
        llm = llm if isinstance(llm, MockLLM) else FallbackLLM(llm)
        captioner = (
            captioner
            if isinstance(captioner, MockVisionLanguageModel)
            else FallbackVisionLanguageModel(captioner)
        )
        vqa = vqa if isinstance(vqa, MockVisionLanguageModel) else FallbackVisionLanguageModel(vqa)
        detector = detector if isinstance(detector, MockDetector) else FallbackDetector(detector)
        image_text_scorer = (
            image_text_scorer
            if isinstance(image_text_scorer, MockImageTextScorer)
            else FallbackImageTextScorer(image_text_scorer)
        )
        ocr = ocr if isinstance(ocr, MockOCR) else FallbackOCR(ocr)
    return ModelBundle(
        llm=llm,
        captioner=captioner,
        vqa=vqa,
        detector=detector,
        image_text_scorer=image_text_scorer,
        ocr=ocr,
    )


def _safe_build(factory: Callable[[], T], fallback: type[T], use_fallback: bool) -> T:
    if use_fallback:
        try:
            return factory()
        except Exception:
            return fallback()
    return factory()


def _build_llm(config: MoreVQAConfig, device: str, dtype: str, force_mock: bool) -> LLMBackend:
    section = config.section("llm")
    provider = str(section.get("provider", "mock")).lower()
    if force_mock or provider == "mock":
        return MockLLM()
    if provider in {"openai_compatible", "openai-compatible", "vllm"}:
        return OpenAICompatibleLLM(
            api_base=_text(section.get("api_base", "http://127.0.0.1:8000/v1")),
            api_key=_text(section.get("api_key", "EMPTY")),
            model=_text(section.get("model", "local-model")),
            temperature=float(section.get("temperature", 0.0)),
            max_tokens=int(section.get("max_tokens", 1024)),
            timeout=int(section.get("timeout", 120)),
        )
    if provider in {"transformers", "hf", "local_transformers"}:
        return TransformersLLM(
            model_path=str(section["model_path"] if "model_path" in section else section["model"]),
            device=device,
            dtype=dtype,
            max_new_tokens=int(section.get("max_tokens", 1024)),
            temperature=float(section.get("temperature", 0.0)),
        )
    raise ValueError(f"Unsupported llm provider: {provider}")


def _build_vlm(section: dict, device: str, dtype: str) -> VisionLanguageBackend:
    provider = str(section.get("provider", "mock")).lower()
    if provider == "mock":
        return MockVisionLanguageModel()
    if provider in {"paligemma2", "paligemma-2", "paligemma2_local"}:
        return PaliGemma2VisionLanguageModel(
            model_path=_text(section.get("model_path", "google/paligemma2-10b-mix-448")),
            device=device,
            dtype=dtype,
            max_new_tokens=int(section.get("max_new_tokens", 64)),
            temperature=float(section.get("temperature", 0.0)),
            caption_prompt=_text(
                section.get("prompt", "caption en")
            ),
            vqa_prefix=_text(section.get("vqa_prefix", "answer en")),
        )
    if provider in {"transformers_vlm", "transformers", "hf"}:
        return TransformersVisionLanguageModel(
            model_path=_text(section["model_path"]),
            device=device,
            dtype=dtype,
            max_new_tokens=int(section.get("max_new_tokens", 64)),
            caption_prompt=str(section.get("prompt", "Describe the image in one concise sentence.")),
        )
    raise ValueError(f"Unsupported VLM provider: {provider}")


def _build_detector(section: dict, device: str, dtype: str) -> DetectorBackend:
    provider = str(section.get("provider", "mock")).lower()
    if provider == "mock":
        return MockDetector()
    if provider in {"owlvit", "owl-vit"}:
        return OwlVitDetector(
            model_path=_text(section["model_path"]),
            device=device,
            dtype=dtype,
            threshold=float(section.get("threshold", 0.12)),
            top_k=int(section.get("top_k", 8)),
            query_max_length=int(section.get("query_max_length", 16)),
        )
    raise ValueError(f"Unsupported detector provider: {provider}")


def _build_scorer(section: dict, device: str, dtype: str) -> ImageTextScorerBackend:
    provider = str(section.get("provider", "mock")).lower()
    if provider == "mock":
        return MockImageTextScorer()
    if provider == "clip":
        return CLIPScorer(model_path=_text(section["model_path"]), device=device, dtype=dtype)
    if provider in {"openai_clip_rn50", "clip_rn50", "clip-rn50"}:
        return OpenAIClipRN50Scorer(
            model_name=_text(section.get("model_name", "RN50")),
            device=device,
            threshold=float(section.get("threshold", 0.7)),
        )
    raise ValueError(f"Unsupported image-text scorer provider: {provider}")


def _build_ocr(section: dict, vqa_model: VisionLanguageBackend) -> OCRBackend:
    provider = str(section.get("provider", "mock")).lower()
    if provider == "mock":
        return MockOCR()
    if provider in {"paligemma2_prompted", "paligemma-2-prompted", "paligemma2"}:
        return PaliGemma2PromptedOCR(
            vqa_model=vqa_model,
            prompt=_text(section.get("prompt", "ocr")),
        )
    raise ValueError(f"Unsupported OCR provider: {provider}")


def _text(value: object) -> str:
    return os.path.expandvars(str(value))


def _same_paligemma_model(captioner: dict, vqa: dict) -> bool:
    providers = {"paligemma2", "paligemma-2", "paligemma2_local"}
    return (
        str(captioner.get("provider", "")).lower() in providers
        and str(vqa.get("provider", "")).lower() in providers
        and _text(captioner.get("model_path", "google/paligemma2-10b-mix-448"))
        == _text(vqa.get("model_path", "google/paligemma2-10b-mix-448"))
    )
