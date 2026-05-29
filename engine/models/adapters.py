from __future__ import annotations

import hashlib
import time
from typing import Any

from engine.models.base import (
    DetectorBackend,
    ImageTextScorerBackend,
    LLMBackend,
    OCRBackend,
    VisionLanguageBackend,
)
from engine.schemas import Detection, VideoFrame


class OpenAICompatibleLLM(LLMBackend):
    def __init__(
        self,
        api_base: str,
        model: str,
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: int = 120,
        extra_body: dict[str, Any] | None = None,
        client: str = "requests",
        proxy_url: str | None = None,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.extra_body = dict(extra_body or {})
        self.client = str(client or "requests").lower()
        self.proxy_url = _normalize_optional_text(proxy_url)

    def generate(self, prompt: str, system: str | None = None) -> str:
        if self.client in {"openai", "openai_sdk", "sdk"}:
            return self._generate_with_openai_sdk(prompt, system=system)
        return self._generate_with_requests(prompt, system=system)

    def _generate_with_openai_sdk(self, prompt: str, system: str | None = None) -> str:
        from openai import OpenAI

        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI SDK transport requires httpx. Install dependencies with "
                "`pip install -r requirements.txt`."
            ) from exc

        print(
            "[LLM] 使用 OpenAI SDK 调用百炼兼容接口: "
            f"model={self.model}, base_url={self.api_base}, timeout={self.timeout}s, "
            f"proxy={'enabled' if self.proxy_url else 'disabled'}",
            flush=True,
        )
        started = time.perf_counter()
        messages = self._build_messages(prompt, system)
        http_client = self._build_httpx_client(httpx)
        client_kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "base_url": self.api_base,
            "timeout": float(self.timeout),
        }
        if http_client is not None:
            client_kwargs["http_client"] = http_client
        client = OpenAI(**client_kwargs)
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                extra_body=self.extra_body or None,
            )
            elapsed = time.perf_counter() - started
            print(f"[LLM] 请求完成，用时 {elapsed:.2f}s", flush=True)
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            raise RuntimeError(f"OpenAI-compatible LLM request failed: {exc}") from exc
        finally:
            client.close()

    def _generate_with_requests(self, prompt: str, system: str | None = None) -> str:
        import requests

        url = f"{self.api_base}/chat/completions"
        print(
            "[LLM] 使用 requests 调用 OpenAI-compatible 接口: "
            f"model={self.model}, url={url}, timeout={self.timeout}s, "
            f"proxy={'enabled' if self.proxy_url else 'disabled'}",
            flush=True,
        )
        started = time.perf_counter()
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        messages = self._build_messages(prompt, system)
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        # DashScope/Bailian accepts OpenAI-compatible payloads and also model-specific
        # fields such as enable_thinking. OpenAI SDK calls this parameter extra_body;
        # because we send raw HTTP here, these fields must be merged into the JSON body.
        payload.update(self.extra_body)
        request_kwargs: dict[str, Any] = {
            "headers": headers,
            "json": payload,
            "timeout": self.timeout,
        }
        if self.proxy_url:
            request_kwargs["proxies"] = {"http": self.proxy_url, "https": self.proxy_url}
        try:
            response = requests.post(url, **request_kwargs)
        except requests.Timeout as exc:
            raise TimeoutError(
                f"LLM request timed out after {self.timeout}s: model={self.model}, url={url}"
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"LLM request failed before receiving a response: {exc}") from exc

        status_code = int(getattr(response, "status_code", 200))
        if status_code >= 400:
            body_preview = response.text[:1000]
            raise RuntimeError(
                "LLM request returned an error: "
                f"status={status_code}, model={self.model}, body={body_preview}"
            )
        data = response.json()
        elapsed = time.perf_counter() - started
        print(f"[LLM] 请求完成，用时 {elapsed:.2f}s", flush=True)
        return data["choices"][0]["message"]["content"].strip()

    def _build_messages(self, prompt: str, system: str | None = None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _build_httpx_client(self, httpx: Any) -> Any | None:
        timeout = httpx.Timeout(float(self.timeout), connect=min(15.0, float(self.timeout)))
        if not self.proxy_url:
            return httpx.Client(timeout=timeout, trust_env=True)
        try:
            return httpx.Client(proxies=self.proxy_url, timeout=timeout, trust_env=False)
        except TypeError:
            return httpx.Client(proxy=self.proxy_url, timeout=timeout, trust_env=False)


class TransformersLLM(LLMBackend):
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=_resolve_torch_dtype(dtype),
            trust_remote_code=True,
        )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, prompt: str, system: str | None = None) -> str:
        import torch

        text = prompt if system is None else f"{system}\n\n{prompt}"
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        do_sample = self.temperature > 0
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=max(self.temperature, 1e-6),
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded[len(text) :].strip() if decoded.startswith(text) else decoded.strip()


class TransformersVisionLanguageModel(VisionLanguageBackend):
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 64,
        caption_prompt: str = "Describe the image in one concise sentence.",
    ) -> None:
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=_resolve_torch_dtype(dtype),
            trust_remote_code=True,
        )
        self.max_new_tokens = max_new_tokens
        self.caption_prompt = caption_prompt

    def caption(self, frame: VideoFrame, prompt: str | None = None) -> str:
        return self._generate(frame, prompt or self.caption_prompt)

    def answer(self, frame: VideoFrame, question: str) -> str:
        prompt = f"Question: {question}\nAnswer:"
        return self._generate(frame, prompt)

    def _generate(self, frame: VideoFrame, prompt: str) -> str:
        import torch

        inputs = self.processor(images=frame.image, text=prompt, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return decoded.strip()


class PaliGemma2VisionLanguageModel(VisionLanguageBackend):
    """Local PaliGemma 2 mix checkpoint for caption, VQA, and OCR-style tasks."""

    def __init__(
        self,
        model_path: str = "google/paligemma2-10b-mix-448",
        device: str = "auto",
        dtype: str = "bfloat16",
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        caption_prompt: str = "caption en",
        vqa_prefix: str = "answer en",
    ) -> None:
        import torch
        from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

        self.processor = PaliGemmaProcessor.from_pretrained(model_path)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=_resolve_torch_dtype(dtype) or torch.bfloat16,
        ).eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.caption_prompt = caption_prompt
        self.vqa_prefix = vqa_prefix

    def caption(self, frame: VideoFrame, prompt: str | None = None) -> str:
        return self._generate(frame, prompt or self.caption_prompt)

    def answer(self, frame: VideoFrame, question: str) -> str:
        return self._generate(frame, f"{self.vqa_prefix} {question}".strip())

    def _generate(self, frame: VideoFrame, prompt: str) -> str:
        import torch

        prompt = _ensure_image_token(prompt)
        inputs = self.processor(text=prompt, images=frame.image, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        for key, value in inputs.items():
            if value.is_floating_point():
                inputs[key] = value.to(dtype=self.model.dtype)
        input_length = inputs["input_ids"].shape[-1]
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature
        with torch.inference_mode():
            generated = self.model.generate(**inputs, **generation_kwargs)
        generated = generated[0][input_length:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()


class OwlVitDetector(DetectorBackend):
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        dtype: str = "auto",
        threshold: float = 0.12,
        top_k: int = 8,
        query_max_length: int = 16,
    ) -> None:
        from transformers import OwlViTForObjectDetection, OwlViTProcessor

        self.processor = OwlViTProcessor.from_pretrained(model_path)
        self.model = OwlViTForObjectDetection.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=_resolve_torch_dtype(dtype),
        )
        self.threshold = threshold
        self.top_k = top_k
        self.query_max_length = query_max_length

    def detect(self, frame: VideoFrame, label: str) -> list[Detection]:
        import torch

        label = self._truncate_query(label)
        texts = [[label]]
        inputs = self.processor(text=texts, images=frame.image, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = torch.tensor([frame.image.size[::-1]], device=self.model.device)
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            threshold=self.threshold,
            target_sizes=target_sizes,
        )[0]
        detections: list[Detection] = []
        for score, box in zip(results["scores"], results["boxes"]):
            detections.append(
                Detection(
                    frame_id=frame.frame_id,
                    label=label,
                    score=float(score.detach().cpu()),
                    box=[float(value) for value in box.detach().cpu().tolist()],
                )
            )
        detections.sort(key=lambda item: item.score, reverse=True)
        return detections[: self.top_k]

    def _truncate_query(self, label: str) -> str:
        tokens = self.processor.tokenizer(
            str(label),
            max_length=self.query_max_length,
            truncation=True,
        )["input_ids"]
        return self.processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()


class CLIPScorer(ImageTextScorerBackend):
    def __init__(self, model_path: str, device: str = "auto", dtype: str = "auto") -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.model = CLIPModel.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=_resolve_torch_dtype(dtype),
        )

    def score(self, frame: VideoFrame, text: str) -> float:
        import torch

        inputs = self.processor(text=[text], images=frame.image, return_tensors="pt", padding=True)
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        logit = float(outputs.logits_per_image[0, 0].detach().cpu())
        return 1.0 / (1.0 + pow(2.718281828, -logit / 10.0))


class OpenAIClipRN50Scorer(ImageTextScorerBackend):
    """CLIP RN50 scorer used by the MoReVQA supplementary configuration."""

    def __init__(
        self,
        model_name: str = "RN50",
        device: str = "auto",
        threshold: float = 0.7,
    ) -> None:
        import clip
        import torch

        self.device = _resolve_device(device, torch)
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        self.clip = clip
        self.torch = torch
        self.threshold = threshold

    def score(self, frame: VideoFrame, text: str) -> float:
        image = self.preprocess(frame.image).unsqueeze(0).to(self.device)
        tokens = self.clip.tokenize([text], truncate=True).to(self.device)
        with self.torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(tokens)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity = float((image_features @ text_features.T)[0, 0].detach().cpu())
        return similarity if similarity >= self.threshold else 0.0


class MockLLM(LLMBackend):
    """Deterministic fallback useful before real models are configured."""

    def generate(self, prompt: str, system: str | None = None) -> str:
        lower = prompt.lower()
        if "event parsing stage" in lower:
            return (
                '{"calls":['
                '{"name":"parse_event","args":["none","main visible event",null]},'
                '{"name":"classify","args":["what"]},'
                '{"name":"require_ocr","args":[false]}]}'
            )
        if "grounding stage" in lower:
            return '{"calls":[{"name":"verify_action","args":["main visible event"],"kwargs":{"top_k":4}}]}'
        if "reasoning stage" in lower:
            return '{"calls":[{"name":"vqa","args":["What is happening in the grounded frames?"]}]}'
        if "candidate answers:" in lower:
            options = _extract_numbered_options(prompt)
            return options[0] if options else "unknown"
        return "unknown"


class MockVisionLanguageModel(VisionLanguageBackend):
    def caption(self, frame: VideoFrame, prompt: str | None = None) -> str:
        return f"a sampled video frame at {frame.timestamp:.2f} seconds"

    def answer(self, frame: VideoFrame, question: str) -> str:
        lowered = question.lower()
        if lowered.startswith("is ") or "answer yes or no" in lowered:
            return "yes"
        return f"visible content around {frame.timestamp:.2f} seconds"


class MockDetector(DetectorBackend):
    def detect(self, frame: VideoFrame, label: str) -> list[Detection]:
        width, height = _image_size(frame.image)
        return [
            Detection(
                frame_id=frame.frame_id,
                label=label,
                score=0.5,
                box=[0.0, 0.0, float(width), float(height)],
            )
        ]


class MockImageTextScorer(ImageTextScorerBackend):
    def score(self, frame: VideoFrame, text: str) -> float:
        key = f"{frame.frame_id}:{text}".encode("utf-8", errors="ignore")
        value = int(hashlib.sha1(key).hexdigest()[:8], 16)
        return 0.25 + (value % 7500) / 10000.0


class PaliGemma2PromptedOCR(OCRBackend):
    """Use the PaliGemma 2 OCR task prompt when text is needed."""

    def __init__(
        self,
        vqa_model: VisionLanguageBackend,
        prompt: str = "ocr",
    ) -> None:
        self.vqa_model = vqa_model
        self.prompt = prompt

    def read_text(self, frame: VideoFrame) -> str:
        return self.vqa_model.caption(frame, prompt=self.prompt).strip()


class MockOCR(OCRBackend):
    def read_text(self, frame: VideoFrame) -> str:
        return ""


class FallbackLLM(LLMBackend):
    def __init__(self, primary: LLMBackend, fallback: LLMBackend | None = None) -> None:
        self.primary = primary
        self.fallback = fallback or MockLLM()

    def generate(self, prompt: str, system: str | None = None) -> str:
        try:
            return self.primary.generate(prompt, system=system)
        except Exception:
            return self.fallback.generate(prompt, system=system)


class FallbackVisionLanguageModel(VisionLanguageBackend):
    def __init__(
        self,
        primary: VisionLanguageBackend,
        fallback: VisionLanguageBackend | None = None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback or MockVisionLanguageModel()

    def caption(self, frame: VideoFrame, prompt: str | None = None) -> str:
        try:
            return self.primary.caption(frame, prompt=prompt)
        except Exception:
            return self.fallback.caption(frame, prompt=prompt)

    def answer(self, frame: VideoFrame, question: str) -> str:
        try:
            return self.primary.answer(frame, question)
        except Exception:
            return self.fallback.answer(frame, question)


class FallbackDetector(DetectorBackend):
    def __init__(self, primary: DetectorBackend, fallback: DetectorBackend | None = None) -> None:
        self.primary = primary
        self.fallback = fallback or MockDetector()

    def detect(self, frame: VideoFrame, label: str) -> list[Detection]:
        try:
            return self.primary.detect(frame, label)
        except Exception:
            return self.fallback.detect(frame, label)


class FallbackImageTextScorer(ImageTextScorerBackend):
    def __init__(
        self,
        primary: ImageTextScorerBackend,
        fallback: ImageTextScorerBackend | None = None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback or MockImageTextScorer()

    def score(self, frame: VideoFrame, text: str) -> float:
        try:
            return self.primary.score(frame, text)
        except Exception:
            return self.fallback.score(frame, text)


class FallbackOCR(OCRBackend):
    def __init__(self, primary: OCRBackend, fallback: OCRBackend | None = None) -> None:
        self.primary = primary
        self.fallback = fallback or MockOCR()

    def read_text(self, frame: VideoFrame) -> str:
        try:
            return self.primary.read_text(frame)
        except Exception:
            return self.fallback.read_text(frame)


def _image_size(image: Any) -> tuple[int, int]:
    size = getattr(image, "size", None)
    if isinstance(size, tuple) and len(size) == 2:
        return int(size[0]), int(size[1])
    return 1, 1


def _resolve_device(device: str, torch: Any) -> str:
    if str(device).lower() in {"auto", "", "none"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device)


def _normalize_optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "~"}:
        return None
    return text


def _ensure_image_token(prompt: str) -> str:
    prompt = str(prompt).strip()
    if prompt.startswith("<image>"):
        return prompt
    return f"<image> {prompt}".strip()


def _resolve_torch_dtype(dtype: str) -> Any:
    if dtype in {"auto", "", "none", "None", None}:
        return None
    import torch

    aliases = {
        "fp16": "float16",
        "float16": "float16",
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp32": "float32",
        "float32": "float32",
    }
    name = aliases.get(str(dtype), str(dtype))
    if not hasattr(torch, name):
        raise ValueError(f"Unsupported torch dtype: {dtype}")
    return getattr(torch, name)


def _extract_numbered_options(prompt: str) -> list[str]:
    options: list[str] = []
    for line in prompt.splitlines():
        stripped = line.strip()
        if len(stripped) > 2 and stripped[0].isdigit() and stripped[1] in {".", ")"}:
            options.append(stripped[2:].strip())
    return options
