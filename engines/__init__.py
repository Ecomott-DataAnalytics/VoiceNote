"""STTエンジンのプラグインパッケージ。

各エンジンは :class:`engines.base.STTEngine` を継承し、共通の
``load()`` / ``transcribe()`` インターフェースを実装する。
重いML依存（torch, transformers など）は各エンジンの ``load()`` 内で
遅延importするため、このパッケージのim-port自体は軽量。
"""

from engines.base import STTEngine, TranscriptionResult

__all__ = ["STTEngine", "TranscriptionResult"]
