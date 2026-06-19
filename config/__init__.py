"""モデル定義(YAML)のローダ。

``config/models.yaml`` を読み込み、エンジン選択やUI生成に使う構造を提供する。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional

import yaml

_YAML_PATH = os.path.join(os.path.dirname(__file__), "models.yaml")

# UIで並べるカテゴリの推奨順
_CATEGORY_ORDER = ["Whisper", "Kotoba", "ReazonSpeech", "Qwen3-ASR"]


@dataclass
class ModelConfig:
    id: str
    engine: str
    category: str
    label: str
    description: str = ""
    vram: str = ""
    languages: Optional[List[str]] = None
    options: Dict = field(default_factory=dict)


@lru_cache(maxsize=1)
def _raw() -> dict:
    with open(_YAML_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not data.get("models"):
        raise ValueError("models.yaml に models が定義されていません")
    return data


@lru_cache(maxsize=1)
def list_models() -> List[ModelConfig]:
    models = []
    for entry in _raw()["models"]:
        models.append(
            ModelConfig(
                id=entry["id"],
                engine=entry["engine"],
                category=entry.get("category", "Other"),
                label=entry.get("label", entry["id"]),
                description=entry.get("description", ""),
                vram=entry.get("vram", ""),
                languages=entry.get("languages"),
                options=entry.get("options", {}) or {},
            )
        )
    return models


def _by_id() -> Dict[str, ModelConfig]:
    return {m.id: m for m in list_models()}


def get_model(model_id: str) -> ModelConfig:
    """モデルIDから設定を取得。未知IDは KeyError。"""
    try:
        return _by_id()[model_id]
    except KeyError:
        raise KeyError(f"未知のモデルIDです: {model_id}")


def get_default() -> str:
    """UI初期選択のモデルID。未指定なら先頭モデル。"""
    default = _raw().get("default")
    if default and default in _by_id():
        return default
    return list_models()[0].id


def grouped_for_ui() -> dict:
    """カテゴリ別にまとめたUI用構造を返す。

    {
      "default": "...",
      "groups": [
        {"category": "Whisper", "models": [{"id","label","description"}, ...]},
        ...
      ]
    }
    """
    groups: Dict[str, List[dict]] = {}
    for m in list_models():
        groups.setdefault(m.category, []).append(
            {"id": m.id, "label": m.label, "description": m.description}
        )

    # 推奨順 → それ以外は登場順
    ordered_categories = [c for c in _CATEGORY_ORDER if c in groups]
    ordered_categories += [c for c in groups if c not in ordered_categories]

    return {
        "default": get_default(),
        "groups": [
            {"category": c, "models": groups[c]} for c in ordered_categories
        ],
    }
