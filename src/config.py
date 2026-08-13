from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is incomplete or invalid."""


def _expand(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    return value


def _load_config_mapping(
    config_path: Path, stack: tuple[Path, ...] = ()
) -> tuple[dict[str, Any], list[str]]:
    if config_path in stack:
        cycle = " -> ".join(str(item) for item in (*stack, config_path))
        raise ConfigError(f"Cyclic config inheritance: {cycle}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ConfigError(f"Top-level YAML value must be a mapping: {config_path}")

    parents = config.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list) or not all(
        isinstance(parent, str) for parent in parents
    ):
        raise ConfigError(f"extends must be a path or list of paths: {config_path}")

    merged: dict[str, Any] = {}
    sources: list[str] = []
    for parent in parents:
        expanded_parent = Path(os.path.expandvars(os.path.expanduser(parent)))
        parent_path = (
            expanded_parent
            if expanded_parent.is_absolute()
            else config_path.parent / expanded_parent
        ).resolve()
        parent_config, parent_sources = _load_config_mapping(
            parent_path, (*stack, config_path)
        )
        merged = deep_merge(merged, parent_config)
        sources.extend(parent_sources)

    merged = deep_merge(merged, config)
    sources.append(str(config_path))
    return merged, list(dict.fromkeys(sources))


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    config, sources = _load_config_mapping(config_path)
    config = _expand(config)
    config["_config_path"] = str(config_path)
    config["_config_sources"] = sources
    return config


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def require(config: dict[str, Any], *keys: str) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        raise ConfigError(f"Missing required configuration keys: {', '.join(missing)}")


def method_config(config: dict[str, Any], name: str) -> dict[str, Any]:
    common = config.get("method_defaults", {})
    specific = config.get("method_configs", {}).get(name, {})
    return deep_merge(common, specific)
