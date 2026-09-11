"""Configuration loading (YAML-driven, attribute access).

Every tunable parameter lives in ``configs/default.yaml``.  A :class:`Config`
wraps a nested dict and exposes dot access with a safe fallback so a missing
key never crashes the pipeline at runtime.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


class Config:
    """Thin dot-access wrapper over a nested dict loaded from YAML."""

    def __init__(self, data: dict[str, Any] | None = None):
        self._data = data or {}

    # -- loading -----------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> "Config":
        p = Path(path) if path else DEFAULT_CONFIG_PATH
        with open(p, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        return cls(_deep_copy(data))

    # -- accessors ---------------------------------------------------------
    def get(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted_key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def section(self, dotted_key: str) -> "Config":
        return Config(self.get(dotted_key, {}) or {})

    def as_dict(self) -> dict[str, Any]:
        return _deep_copy(self._data)

    # convenience helpers
    @property
    def bands(self) -> list[dict[str, float]]:
        return self.get("bands", [])

    def __getattr__(self, name: str) -> Any:
        # Fallback for top-level keys, e.g. cfg.bands / cfg.preprocess
        try:
            return self._data[name]
        except (KeyError, TypeError):
            raise AttributeError(name)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Config({self._data!r})"

    def set(self, dotted_key: str, value: Any) -> None:
        node = self._data
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value


def _deep_copy(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _deep_copy(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_deep_copy(v) for v in node]
    return node


def load_config(path: str | Path | None = None) -> Config:
    """Load the default config (optionally overriding with ``path``)."""
    return Config.from_yaml(path)