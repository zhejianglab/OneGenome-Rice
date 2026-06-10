from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml


class Config:
    """Nested configuration object with dict and attribute access."""

    def __init__(self, data: Dict[str, Any]):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def update(self, data: Dict[str, Any]) -> None:
        for key, value in data.items():
            if isinstance(value, dict) and hasattr(self, key):
                existing = getattr(self, key)
                if isinstance(existing, Config):
                    existing.update(value)
                else:
                    setattr(self, key, Config(value))
            elif isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        if config_dict is None:
            raise ValueError(f"Config file is empty: {path}")
        return config_dict
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file {path}: {e}")


def parse_config_from_path(config_path: str | Path) -> Config:
    return Config(load_yaml_config(config_path))


def merge_cli_overrides(config: Config, cli_args: List[str]) -> Config:
    i = 0
    while i < len(cli_args):
        arg = cli_args[i]
        if arg.startswith("--"):
            arg_name = arg[2:]
            if "=" in arg_name:
                key_path, value = arg_name.split("=", 1)
            elif i + 1 < len(cli_args) and not cli_args[i + 1].startswith("--"):
                key_path = arg_name
                value = cli_args[i + 1]
                i += 1
            else:
                i += 1
                continue

            keys = key_path.split(".")
            current = config
            for key in keys[:-1]:
                if not hasattr(current, key):
                    setattr(current, key, Config({}))
                current = getattr(current, key)

            final_key = keys[-1]
            try:
                if value.isdigit():
                    value = int(value)
                elif value.replace(".", "", 1).isdigit():
                    value = float(value)
                elif value.lower() in ("true", "false"):
                    value = value.lower() == "true"
            except (AttributeError, ValueError):
                pass
            setattr(current, final_key, value)
        i += 1
    return config


def load_config(config_path: str | Path, cli_args: list[str] | None = None) -> Config:
    """Load project config and apply optional CLI overrides."""
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    config = Config(raw)
    if cli_args:
        config = merge_cli_overrides(config, cli_args)
    return config


def require_keys(config: Config, required: list[str]) -> None:
    """Require top-level config keys."""
    for key in required:
        if not hasattr(config, key):
            raise ValueError(f"Missing required config key: {key}")


def get_nested(config: Config, path: str, default: Any = None) -> Any:
    """Get nested value from dot path."""
    current: Any = config
    for key in path.split("."):
        if hasattr(current, key):
            current = getattr(current, key)
        else:
            return default
    return current

