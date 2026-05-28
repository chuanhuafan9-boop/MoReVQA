from __future__ import annotations

from engine.memory import ExternalMemory


EVENT_PARSER_PROMPT = """You are the event parsing stage M1 in MoReVQA.
Your task is language-only parsing for video question answering.

Return JSON only:
{"calls": [{"name": "...", "args": [...], "kwargs": {...}}]}

Allowed APIs:
- trim(hint): reduce active frames using temporal hint. hint is one of
  beginning, middle, end, none.
- parse_event(conjunction, event, rewritten_question): store a high-level event
  and a clearer question for later stages. conjunction can be before, after,
  during, while, when, and, none.
- classify(qa_type): one of what, why, how, where, who, when, yes_no, count,
  description, unknown.
- require_ocr(flag): true if reading visible text may be required.
- noop(): do nothing.

Examples:
Question: "Why was the cat lying on its back near the end?"
{"calls":[
  {"name":"trim","args":["end"]},
  {"name":"parse_event","args":["none","cat lying on its back","why was the cat lying on its back?"]},
  {"name":"classify","args":["why"]},
  {"name":"require_ocr","args":[false]}
]}

Question: "What does the man do before removing his skates?"
{"calls":[
  {"name":"parse_event","args":["before","man removing his skates","what does the man do?"]},
  {"name":"classify","args":["what"]},
  {"name":"require_ocr","args":[false]}
]}

Question: "{question}"
Candidate answers:
{options}
"""


GROUNDING_PROMPT = """You are the grounding stage M2 in MoReVQA.
Use the shared memory to identify relevant temporal regions in the video.

Return JSON only:
{"calls": [{"name": "...", "args": [...], "kwargs": {...}}]}

Allowed APIs:
- localize(query, top_k=null): localize an object/entity phrase over active
  frames using detection and image-text matching.
- verify_action(action, top_k=null): verify which active frames contain the
  event/action.
- truncate(relation): refine active frames around the verified event. relation
  can be before, after, during, while, when, none.
- noop(): do nothing.

Use localize for objects and verify_action for events. If the memory contains
a temporal conjunction such as before/after/during, call truncate after
verification.

Shared memory:
{memory}
"""


REASONING_PROMPT = """You are the reasoning stage M3 in MoReVQA.
Ask targeted VQA sub-questions on grounded frames, then let the final
prediction stage combine your outputs with general video context.

Return JSON only:
{"calls": [{"name": "...", "args": [...], "kwargs": {...}}]}

Allowed APIs:
- vqa(question_or_questions, frame_ids=null): ask a VLM one or more questions
  on grounded frames.
- noop(): do nothing.

Good reasoning decomposes the original question into answer-relevant
sub-questions. For why/how questions, ask about the visible action, cause,
surroundings, and actor state. For where/who/what questions, ask directly and
add one supporting question if helpful.

Shared memory:
{memory}
"""


PREDICTION_PROMPT = """You are the final prediction LLM for a MoReVQA system.
Answer the video question using only the memory below.

If candidate answers are provided, return exactly one candidate answer. If no
candidate answers are provided, return a short direct answer.

Memory:
{memory}

Final answer:
"""


def format_options(options: list[str] | None) -> str:
    if not options:
        return "None"
    return "\n".join(f"{idx}. {option}" for idx, option in enumerate(options, start=1))


def build_event_prompt(memory: ExternalMemory) -> str:
    return _render(
        EVENT_PARSER_PROMPT,
        question=memory.question,
        options=format_options(memory.options),
    )


def build_grounding_prompt(memory: ExternalMemory) -> str:
    return _render(
        GROUNDING_PROMPT,
        memory=memory.text_summary(max_captions=16, max_qas=16),
    )


def build_reasoning_prompt(memory: ExternalMemory) -> str:
    return _render(
        REASONING_PROMPT,
        memory=memory.text_summary(max_captions=24, max_qas=16),
    )


def build_prediction_prompt(memory: ExternalMemory) -> str:
    return _render(
        PREDICTION_PROMPT,
        memory=memory.text_summary(max_captions=96, max_qas=96),
    )


def _render(template: str, **values: str) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    return rendered
