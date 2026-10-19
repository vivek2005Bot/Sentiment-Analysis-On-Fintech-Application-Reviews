"""Configuration loader. All modules import paths/settings from here.

Reads configs/config.yaml and configs/model_config.yaml relative to the
project root (the parent of src/). Provides a cached singleton so YAML is
parsed once per process.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model_config.yaml"

_cfg_cache: dict | None = None
_model_cfg_cache: dict | None = None


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required config not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def get_config(reload: bool = False) -> dict:
    """Return the merged project config (cached)."""
    global _cfg_cache
    if _cfg_cache is None or reload:
        _cfg_cache = _load_yaml(CONFIG_PATH)
    return copy.deepcopy(_cfg_cache)


def get_model_config(reload: bool = False) -> dict:
    """Return the transformer benchmark config (cached)."""
    global _model_cfg_cache
    if _model_cfg_cache is None or reload:
        _model_cfg_cache = _load_yaml(MODEL_CONFIG_PATH)
    return copy.deepcopy(_model_cfg_cache)


def get_paths() -> dict[str, Path]:
    """Resolve every configured path into an absolute Path object."""
    cfg = get_config()
    paths = {
        key: (PROJECT_ROOT / value)
        for key, value in cfg["paths"].items()
        if key not in ("raw_data_dirs",)
    }
    for key in paths:
        if key.endswith("_dir"):
            paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def resolve_raw_dirs() -> list[tuple[Path, str]]:
    """Return [(absolute_raw_dir, bank_type), ...] for immutable raw folders."""
    cfg = get_config()
    out = []
    for folder, bank_type in cfg["paths"]["raw_data_dirs"].items():
        p = PROJECT_ROOT / folder
        out.append((p, bank_type))
    return out


def get_seed() -> int:
    return int(get_config()["project"]["seed"])


if __name__ == "__main__":
    cfg = get_config()
    print(f"Project: {cfg['project']['name']} v{cfg['project']['version']}")
    for k, v in get_paths().items():
        print(f"  {k}: {v}")
