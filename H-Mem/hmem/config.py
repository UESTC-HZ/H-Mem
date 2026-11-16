"""Configuration for H-MEM.

Every setting has a default mirroring the EACL 2026 paper and can be overridden
through a YAML file, keyword arguments, or ``HMEM_*`` environment variables.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------- #
# Small nested configs
# --------------------------------------------------------------------------- #


@dataclass
class ForgettingConfig:
    """Ebbinghaus forgetting-curve parameters (Section 3.3)."""

    initial_strength: float = 100.0
    min_strength: float = 1.0
    max_strength: float = 10_000.0
    min_weight: float = 0.01
    max_weight: float = 10.0


@dataclass
class FeedbackConfig:
    """LLM-driven feedback regulation multipliers (Section 3.3)."""

    approve_weight: float = 1.5
    rebut_weight: float = 0.5
    neutral_weight: float = 1.0
    use_llm_weight: bool = True


@dataclass
class EncoderConfig:
    """Text-encoder settings (BERT is used in the paper)."""

    backend: str = "auto"  # "sentence" | "transformers" | "hash" | "auto"
    model_name: str = "bert-base-uncased"
    dim: int = 768
    normalize: bool = True
    device: str = "auto"
    batch_size: int = 32


@dataclass
class LLMConfig:
    """A single LLM endpoint (base QA model or the memory-analysis model)."""

    provider: str = "ollama"  # "ollama" | "openai" | "mock"
    base_url: str = "http://localhost:11434"
    api_key: str = ""
    model: str = "qwen2.5:1.5b"
    temperature: float = 0.0
    max_tokens: int = 512
    timeout: float = 300.0


@dataclass
class HMemConfig:
    """Top-level H-MEM configuration."""

    # Hierarchy
    num_layers: int = 4

    # Retrieval
    top_k: int = 10
    final_top_k: int = 10
    backend: str = "auto"  # "faiss" | "numpy" | "auto"
    dedup_threshold: float = 0.6

    # Memory update
    forgetting: ForgettingConfig = field(default_factory=ForgettingConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)

    # Encoder / LLMs
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory_llm: LLMConfig = field(
        default_factory=lambda: LLMConfig(model="deepseek-r1:8b", max_tokens=1024)
    )

    # Persistence / ablation
    persist_dir: str = "./runs/memory"
    storage_mode: str = "hierarchical"  # "hierarchical" | "flat" | "none"
    retrieval_mode: str = "hierarchical"  # "hierarchical" | "flat" | "none"

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """Recursively serialize to a plain dict (useful for logging)."""
        out: dict[str, Any] = {}
        for key, value in asdict(self).items():
            out[key] = value
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HMemConfig:
        """Build a config from a (possibly partial) dict, e.g. parsed YAML."""
        known = {
            "forgetting": ForgettingConfig,
            "feedback": FeedbackConfig,
            "encoder": EncoderConfig,
            "llm": LLMConfig,
            "memory_llm": LLMConfig,
        }
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key in known and isinstance(value, dict):
                kwargs[key] = known[key](**value)
            else:
                kwargs[key] = value
        return cls(**kwargs)


# Well-known environment variables -> config paths (dotted keys). Lower
# precedence than explicit ``load_config`` overrides, higher than YAML.
_ENV_MAP = {
    "HMEM_LLM_PROVIDER": "llm.provider",
    "HMEM_LLM_BASE_URL": "llm.base_url",
    "HMEM_LLM_MODEL": "llm.model",
    "HMEM_LLM_API_KEY": "llm.api_key",
    "HMEM_MEMORY_LLM_PROVIDER": "memory_llm.provider",
    "HMEM_MEMORY_LLM_BASE_URL": "memory_llm.base_url",
    "HMEM_MEMORY_LLM_MODEL": "memory_llm.model",
    "HMEM_MEMORY_LLM_API_KEY": "memory_llm.api_key",
    "HMEM_ENCODER_BACKEND": "encoder.backend",
    "HMEM_ENCODER_MODEL_NAME": "encoder.model_name",
    "HMEM_TOP_K": "top_k",
    "HMEM_BACKEND": "backend",
}


def _apply_env_overrides(config: HMemConfig) -> HMemConfig:
    for env_name, config_path in _ENV_MAP.items():
        value = os.environ.get(env_name)
        if value:
            _set_attr(config, config_path, _coerce(value))
    return config


def _coerce(value: str) -> Any:
    """Best-effort str -> bool/int/float for env-provided values."""
    lowered = value.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def discover_config_path() -> str | None:
    """Locate a config file if one exists, in priority order.

    1. ``$HMEM_CONFIG`` environment variable
    2. ``./configs/default.yaml`` (repo checkout)
    3. ``~/.config/hmem/config.yaml`` (per-user)

    Returns ``None`` when nothing is found (defaults are then used).
    """
    candidates = []
    env = os.environ.get("HMEM_CONFIG")
    if env:
        candidates.append(env)
    candidates.append("configs/default.yaml")
    candidates.append(str(Path.home() / ".config" / "hmem" / "config.yaml"))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    return None


def load_config(path: str | None = None, **overrides: Any) -> HMemConfig:
    """Load configuration: defaults <- YAML file <- explicit overrides.

    When ``path`` is ``None`` the config file is auto-discovered (see
    :func:`discover_config_path`). ``overrides`` may use dotted keys such as
    ``llm__model="qwen2.5:3b"`` or nested dataclass instances
    (e.g. ``llm=LLMConfig(...)``).
    """
    config = HMemConfig()

    resolved = path or discover_config_path()
    if resolved:
        with open(resolved, encoding="utf-8") as fh:
            config = HMemConfig.from_dict(yaml.safe_load(fh) or {})

    _apply_env_overrides(config)

    for key, value in overrides.items():
        _set_attr(config, key, value)

    return config


def _set_attr(config: HMemConfig, key: str, value: Any) -> None:
    """Set a (possibly dotted) attribute on the config dataclass."""
    if "__" in key or "." in key:
        parts = key.replace(".", "__").split("__")
        obj: Any = config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
    else:
        setattr(config, key, value)


def save_config(config: HMemConfig, path: str) -> None:
    """Persist the effective configuration as YAML."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config.to_dict(), fh, sort_keys=False)
