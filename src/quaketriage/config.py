"""Config loader — reads configs/default.yaml, expanding ${VAR} / ${VAR:-default} from the env.

Values for secrets and provider choices live in `.env`; the YAML only references them.
Set QUAKETRIAGE_CONFIG to point at another YAML file (e.g. for experiments).
"""
import copy
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

DEFAULT_CONFIG_PATH = Path(__file__).parents[2] / "configs" / "default.yaml"

_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(ValueError):
    """Raised for unresolved placeholders (strict mode) or missing keys."""


def _expand(value: Any, strict: bool) -> Any:
    if isinstance(value, str):

        def repl(match: re.Match) -> str:
            name, default = match.group(1), match.group(2)
            found = os.environ.get(name)
            if found is not None:
                return found
            if default is not None:
                return default
            if strict:
                raise ConfigError(f"Environment variable {name} is not set")
            return ""

        return _PLACEHOLDER.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand(v, strict) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v, strict) for v in value]
    return value


@lru_cache(maxsize=8)
def _load(path: str, strict: bool) -> dict:
    load_dotenv()
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _expand(raw, strict)


def load_config(path: str | Path | None = None, strict: bool = False) -> dict:
    """Return the config as a nested dict (a fresh copy; the parsed file is cached).

    Unset `${VAR}` placeholders become "" unless `strict=True`, which raises ConfigError.
    """
    resolved = path or os.environ.get("QUAKETRIAGE_CONFIG") or DEFAULT_CONFIG_PATH
    return copy.deepcopy(_load(str(resolved), strict))


def get(key: str, default: Any = ..., path: str | Path | None = None) -> Any:
    """Look up a dotted key such as 'geo.crs'. Raises ConfigError if missing without a default."""
    node: Any = load_config(path)
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            if default is ...:
                raise ConfigError(f"Config key not found: {key}")
            return default
        node = node[part]
    return node
