from __future__ import annotations

from PIL import Image

from engine.config import MoreVQAConfig
from engine.memory import ExternalMemory
from engine.models.registry import build_model_bundle
from engine.schemas import VideoFrame
from engine.utils import ProgramGenerator, parse_action_plan
from framework.morevqa import MoReVQA


def test_parse_json_plan() -> None:
    calls = parse_action_plan('{"calls":[{"name":"trim","args":["end"]}]}')
    assert calls[0].name == "trim"
    assert calls[0].args == ["end"]


def test_parse_code_like_plan() -> None:
    calls = parse_action_plan('trim("beginning")\nclassify("why")')
    assert [call.name for call in calls] == ["trim", "classify"]


def test_memory_summary() -> None:
    memory = ExternalMemory("What is happening?", options=["a", "b"], frame_ids=[0, 1, 2])
    assert "Original question" in memory.text_summary()


def test_video_settings_use_paper_config_key() -> None:
    config = MoreVQAConfig({"video": {"context_frames": 30}})
    model = MoReVQA(config, models=build_model_bundle(config, force_mock=True))
    assert model.reasoning_stage.context_frames == 30


def test_program_generator_parses_llm_tool_calls() -> None:
    class _LLM:
        def generate(self, _prompt: str) -> str:
            return '{"calls":[{"name":"noop","args":[]}]}'

    calls, raw = ProgramGenerator(_LLM()).generate("plan")
    assert raw.startswith('{"calls"')
    assert calls[0].name == "noop"


def test_morevqa_inference_with_mock_frames() -> None:
    config = MoreVQAConfig.defaults()
    models = build_model_bundle(config, force_mock=True)
    model = MoReVQA(config, models=models)
    frames = [
        VideoFrame(frame_id=i, timestamp=float(i), image=Image.new("RGB", (16, 16)))
        for i in range(4)
    ]
    output = model.inference("What is happening?", frames, ["running", "jumping"])
    assert output.prediction.answer in {"running", "jumping"}
    assert output.memory.traces
