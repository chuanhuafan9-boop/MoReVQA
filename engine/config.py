from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping


class ConfigError(RuntimeError):
    """Raised when a configuration file cannot be loaded."""


class MoreVQAConfig:
    """Thin helper around the YAML config used by all modules."""

    def __init__(self, data: Mapping[str, Any] | None = None) -> None:
        self.data: dict[str, Any] = dict(data or {})

    @classmethod
    def from_file(cls, path: str | Path) -> "MoreVQAConfig":
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"Config file does not exist: {path}")
        try:
            import yaml
        except ImportError:
            data = _parse_simple_yaml(path.read_text(encoding="utf-8"))
        else:
            with path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
        if not isinstance(data, MutableMapping):
            raise ConfigError(f"Config root must be a mapping: {path}")
        return cls(data)

    @classmethod
    def defaults(cls) -> "MoreVQAConfig":
        return cls(
            {
                "runtime": {"device": "auto", "dtype": "auto", "mock_on_missing": True},
                "llm": {
                    "provider": "mock",
                    "temperature": 0.0,
                    "max_tokens": 1024,
                    "timeout": 120,
                },
                "vision_language": {
                    "captioner": {"provider": "mock"},
                    "vqa": {"provider": "mock"},
                },
                "grounding": {
                    "detector": {"provider": "mock", "threshold": 0.12, "top_k": 8},
                    "image_text_scorer": {"provider": "mock"},
                    "verify_top_k": 8,
                    "grounding_top_k": 6,
                    "temporal_keep_ratio": 0.4,
                },
                "ocr": {"provider": "mock"},
                "video": {
                    "sample_fps": 1.0,
                    "max_frames": None,
                    "context_frames": 16,
                },
                "pipeline": {"execute_llm_plans": True, "answer_max_words": 16},
            }
        )

    def merged_with_defaults(self) -> "MoreVQAConfig":
        base = MoreVQAConfig.defaults().data
        return MoreVQAConfig(_deep_merge(base, self.data))

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.data
        for key in keys:
            if not isinstance(node, Mapping) or key not in node:
                return default
            node = node[key]
        return node

    def section(self, *keys: str) -> dict[str, Any]:
        value = self.get(*keys, default={})
        return dict(value or {}) if isinstance(value, Mapping) else {}

    def bool(self, *keys: str, default: bool = False) -> bool:
        value = self.get(*keys, default=default)
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "y", "on"}
        return bool(value)

    def int(self, *keys: str, default: int = 0) -> int:
        value = self.get(*keys, default=default)
        return int(default if value is None else value)

    def float(self, *keys: str, default: float = 0.0) -> float:
        value = self.get(*keys, default=default)
        return float(default if value is None else value)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def first_present(mapping: Mapping[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Small fallback parser for the repository's simple mapping-only YAML."""

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        line = _strip_inline_comment(raw_line.rstrip())
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            raise ConfigError(f"Unsupported YAML line: {raw_line}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        if value == "":
            child: dict[str, Any] = {}
            current[key] = child
            stack.append((indent, child))
        else:
            current[key] = _parse_scalar(value)
    return root


def _strip_inline_comment(line: str) -> str:
    quote: str | None = None
    for index, char in enumerate(line):
        if char in {"'", '"'}:
            quote = None if quote == char else char if quote is None else quote
        elif char == "#" and quote is None:
            prefix = line[:index]
            if not prefix or prefix[-1].isspace():
                return prefix.rstrip()
    return line


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
