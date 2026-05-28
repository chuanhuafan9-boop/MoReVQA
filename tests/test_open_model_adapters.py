from __future__ import annotations

from pathlib import Path

from PIL import Image

import engine.models.registry as registry
from engine.config import MoreVQAConfig
from engine.models.adapters import (
    MockVisionLanguageModel,
    OpenAICompatibleLLM,
    PaliGemma2PromptedOCR,
    PaliGemma2VisionLanguageModel,
)
from engine.schemas import VideoFrame


class _Response:
    def __init__(self, data: dict) -> None:
        self.data = data

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.data


def test_qwen_vllm_uses_openai_compatible_request(monkeypatch) -> None:
    recorded: dict = {}

    def fake_post(url: str, headers: dict, json: dict, timeout: int) -> _Response:
        recorded.update({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return _Response({"choices": [{"message": {"content": '{"calls":[]}'}}]})

    monkeypatch.setattr("requests.post", fake_post)
    llm = OpenAICompatibleLLM(
        api_base="http://127.0.0.1:8000/v1",
        model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        api_key="EMPTY",
        temperature=0.0,
    )

    assert llm.generate("Return JSON.") == '{"calls":[]}'
    assert recorded["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert recorded["json"]["model"] == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert recorded["json"]["temperature"] == 0.0


def test_paligemma2_formats_official_mix_task_prefixes() -> None:
    vlm = object.__new__(PaliGemma2VisionLanguageModel)
    vlm.caption_prompt = "caption en"
    vlm.vqa_prefix = "answer en"
    vlm._generate = lambda _frame, prompt: prompt
    frame = VideoFrame(0, 0.0, Image.new("RGB", (8, 8)))

    assert vlm.caption(frame) == "caption en"
    assert vlm.answer(frame, "What is the person doing?") == "answer en What is the person doing?"


def test_paligemma2_ocr_uses_ocr_task_prompt() -> None:
    vlm = object.__new__(PaliGemma2VisionLanguageModel)
    vlm.caption_prompt = "caption en"
    vlm._generate = lambda _frame, prompt: prompt
    ocr = PaliGemma2PromptedOCR(vlm, prompt="ocr")

    assert ocr.read_text(VideoFrame(0, 0.0, Image.new("RGB", (8, 8)))) == "ocr"


def test_caption_and_vqa_share_one_paligemma_model(monkeypatch) -> None:
    builds: list[object] = []

    def fake_build_vlm(section: dict, device: str, dtype: str) -> MockVisionLanguageModel:
        builds.append(section)
        return MockVisionLanguageModel()

    monkeypatch.setattr(registry, "_build_vlm", fake_build_vlm)
    config = MoreVQAConfig(
        {
            "runtime": {"mock_on_missing": False},
            "llm": {"provider": "mock"},
            "vision_language": {
                "captioner": {
                    "provider": "paligemma2",
                    "model_path": "google/paligemma2-10b-mix-448",
                },
                "vqa": {
                    "provider": "paligemma2",
                    "model_path": "google/paligemma2-10b-mix-448",
                },
            },
        }
    )

    models = registry.build_model_bundle(config)

    assert models.captioner is models.vqa
    assert len(builds) == 1


def test_force_mock_skips_open_model_loading() -> None:
    root = Path(__file__).resolve().parents[1]
    config = MoreVQAConfig.from_file(root / "configs" / "LLM_config.yaml")

    models = registry.build_model_bundle(config, force_mock=True)

    assert isinstance(models.captioner, MockVisionLanguageModel)
    assert isinstance(models.vqa, MockVisionLanguageModel)
