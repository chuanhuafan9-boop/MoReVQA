from __future__ import annotations

import re
from collections import Counter

from engine.config import MoreVQAConfig
from engine.memory import ExternalMemory
from engine.models.base import LLMBackend
from engine.schemas import PredictionResult
from prompts.prompt_engineering import build_prediction_prompt


class PredictionStage:
    stage_name = "prediction"

    def __init__(self, llm: LLMBackend, config: MoreVQAConfig) -> None:
        self.llm = llm
        self.answer_max_words = config.int("pipeline", "answer_max_words", default=16)

    def run(self, memory: ExternalMemory) -> PredictionResult:
        prompt = build_prediction_prompt(memory)
        raw = self.llm.generate(prompt)
        answer = self._postprocess(raw, memory)
        result = PredictionResult(answer=answer, raw_response=raw)
        memory.add_trace(self.stage_name, "prediction", result)
        return result

    def _postprocess(self, raw: str, memory: ExternalMemory) -> str:
        raw = (raw or "").strip()
        if memory.options:
            option = _select_option(raw, memory.options)
            if option is not None:
                return option
            return _fallback_option(memory)
        line = raw.splitlines()[0].strip() if raw else "unknown"
        words = line.split()
        if len(words) > self.answer_max_words:
            line = " ".join(words[: self.answer_max_words])
        return line.strip(" .") or "unknown"


def _select_option(raw: str, options: list[str]) -> str | None:
    lower = raw.strip().lower()
    match = re.search(r"\b(?:option|answer)?\s*([1-9][0-9]*)\b", lower)
    if match:
        index = int(match.group(1)) - 1
        if 0 <= index < len(options):
            return options[index]
    for option in options:
        if option.lower() in lower:
            return option
    letters = "abcdefghijklmnopqrstuvwxyz"
    match = re.search(r"\b([a-z])\b", lower)
    if match:
        index = letters.index(match.group(1))
        if 0 <= index < len(options):
            return options[index]
    return None


def _fallback_option(memory: ExternalMemory) -> str:
    if not memory.options:
        return "unknown"
    evidence = memory.text_summary().lower()
    scores = []
    for option in memory.options:
        tokens = [token for token in re.findall(r"[a-z0-9]+", option.lower()) if len(token) > 2]
        counts = Counter(re.findall(r"[a-z0-9]+", evidence))
        score = sum(counts[token] for token in tokens)
        scores.append(score)
    best = max(range(len(memory.options)), key=lambda idx: (scores[idx], -idx))
    return memory.options[best]
