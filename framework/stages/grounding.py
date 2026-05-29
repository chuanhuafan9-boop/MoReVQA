from __future__ import annotations

from engine.config import MoreVQAConfig
from engine.utils import (
    ProgramGenerator,
    ProgramInterpreter,
    log_api_program,
    log_model_output,
    log_prompt,
)
from engine.memory import ExternalMemory
from engine.models.base import (
    DetectorBackend,
    ImageTextScorerBackend,
    LLMBackend,
    VisionLanguageBackend,
)
from engine.schemas import ActionCall, VideoFrame
from prompts.prompt_engineering import build_grounding_prompt
from tools.grounding import GroundingAPI


class GroundingStage:
    stage_name = "grounding"

    def __init__(
        self,
        llm: LLMBackend,
        detector: DetectorBackend,
        image_text_scorer: ImageTextScorerBackend,
        vqa_model: VisionLanguageBackend,
        config: MoreVQAConfig,
    ) -> None:
        self.generator = ProgramGenerator(llm)
        self.interpreter = ProgramInterpreter()
        self.detector = detector
        self.image_text_scorer = image_text_scorer
        self.vqa_model = vqa_model
        self.top_k = config.int("grounding", "grounding_top_k", default=6)

    def run(self, memory: ExternalMemory, frames: list[VideoFrame]) -> ExternalMemory:
        prompt = build_grounding_prompt(memory)
        log_prompt("M2 Grounding LLM tool-plan", prompt)
        print("[MoReVQA][M2 Grounding] 正在请求 LLM 生成 grounding 工具计划...", flush=True)
        calls, raw = self.generator.generate(prompt)
        log_model_output("M2 Grounding LLM tool-plan", raw)
        log_api_program("M2 Grounding parsed program", calls)
        print(
            f"[MoReVQA][M2 Grounding] LLM 返回完成，解析到 {len(calls)} 个工具调用",
            flush=True,
        )
        if not calls:
            print("[MoReVQA][M2 Grounding] LLM 未返回可执行工具调用，使用启发式 grounding 计划", flush=True)
            calls = heuristic_grounding_plan(memory)
            raw = "heuristic_grounding_plan"
            log_api_program("M2 Grounding heuristic program", calls)
        memory.set_plan(self.stage_name, calls, raw)
        api = GroundingAPI(
            memory=memory,
            frames=frames,
            detector=self.detector,
            image_text_scorer=self.image_text_scorer,
            vqa_model=self.vqa_model,
            top_k=self.top_k,
        )
        print("[MoReVQA][M2 Grounding] 开始执行 grounding 工具计划", flush=True)
        self.interpreter.execute(self.stage_name, calls, api, memory)
        if not memory.grounded_frame_ids and memory.frame_ids:
            middle = memory.frame_ids[len(memory.frame_ids) // 2]
            memory.set_grounded_frame_ids([middle], self.stage_name, "fallback:middle_frame")
        return memory


def heuristic_grounding_plan(memory: ExternalMemory) -> list[ActionCall]:
    query = memory.event_queue[0] if memory.event_queue else memory.working_question or memory.question
    calls = [ActionCall("verify_action", [query])]
    if memory.conjunction in {"before", "after", "during", "while", "when"}:
        calls.append(ActionCall("truncate", [memory.conjunction]))
    return calls
