"""STTエンジンのファクトリ。

YAML(config/models.yaml)で宣言された ``engine`` 値でエンジンクラスを生成する。
生成済みインスタンスはキャッシュし、モデルの再ロードを避ける（Celery worker
プロセス内で常駐）。エンジン未指定時はモデルIDの接頭辞から推定する。
"""

from __future__ import annotations

import logging
from typing import Optional

from engines.base import STTEngine
from engines.faster_whisper_engine import FasterWhisperEngine
from engines.kotoba_engine import KotobaEngine
from engines.qwen3_asr_engine import Qwen3ASREngine
from engines.reazon_engine import ReazonEngine

logger = logging.getLogger(__name__)

# engine 名 -> クラス
_REGISTRY = {
    "faster-whisper": FasterWhisperEngine,
    "whisper": FasterWhisperEngine,  # 旧 whisper は faster-whisper に集約
    "kotoba": KotobaEngine,
    "reazon": ReazonEngine,
    "qwen3": Qwen3ASREngine,
}


def _infer_engine(model_id: str) -> str:
    """engine 未指定時にモデルIDから推定する（計画のprefix判定）。"""
    mid = (model_id or "").lower()
    if mid.startswith("qwen"):
        return "qwen3"
    if mid.startswith("reazon"):
        return "reazon"
    if mid.startswith("kotoba"):
        return "kotoba"
    return "faster-whisper"


class EngineFactory:
    # (engine, model_id, options_key) -> STTEngine
    _cache: dict = {}

    @staticmethod
    def _key(engine: str, model_id: str, options: Optional[dict]) -> tuple:
        opts_key = tuple(sorted((options or {}).items()))
        return (engine, model_id, opts_key)

    @classmethod
    def create(
        cls,
        engine: Optional[str],
        model_id: str,
        options: Optional[dict] = None,
    ) -> STTEngine:
        engine = (engine or "").strip().lower() or _infer_engine(model_id)
        engine_cls = _REGISTRY.get(engine)
        if engine_cls is None:
            logger.warning("未知のエンジン '%s' → モデルIDから推定します", engine)
            engine = _infer_engine(model_id)
            engine_cls = _REGISTRY[engine]

        key = cls._key(engine, model_id, options)
        if key not in cls._cache:
            logger.info("エンジン生成: %s (%s)", engine, model_id)
            cls._cache[key] = engine_cls(model_id, options)
        return cls._cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._cache.clear()
