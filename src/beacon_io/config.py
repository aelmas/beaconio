"""Centralised configuration loader."""

from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = _ROOT / "config" / "config.yaml"


def load_config(path: Path | None = None) -> dict:
    path = path or DEFAULT_CONFIG
    with open(path) as fh:
        return yaml.safe_load(fh)


CFG = load_config()
