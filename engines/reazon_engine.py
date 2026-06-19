"""ReazonSpeech k2-v2 エンジン（公式SDK / sherpa-onnx）。

日本語専用の軽量モデル（159M）。CPU でも動作する。
``from reazonspeech.k2.asr import load_model, transcribe, audio_from_path`` を利用。
"""

from __future__ import annotations

import logging
from typing import Optional

from engines.base import ProgressCallback, Segment, STTEngine, TranscriptionResult

logger = logging.getLogger(__name__)


class ReazonEngine(STTEngine):
    name = "reazon"
    # k2 は入力全体を1パス処理し長尺は非推奨（大量メモリ）。タスク側で分割する。
    chunks_internally = False

    def load(self) -> None:
        if self._loaded:
            return
        from reazonspeech.k2.asr import load_model

        device, _ = self.resolve_device()
        logger.info("ReazonSpeech k2 モデルをロード中 (device=%s)", device)
        try:
            self._model = load_model(device=device)
        except TypeError:
            # 古い/新しいシグネチャ差異への保険
            self._model = load_model()
        self._loaded = True

    def transcribe(
        self,
        file_path: str,
        language: str = "auto",
        progress_cb: Optional[ProgressCallback] = None,
    ) -> TranscriptionResult:
        if language not in ("auto", "ja"):
            logger.warning(
                "ReazonSpeech は日本語専用です。指定言語 '%s' は無視されます。", language
            )
        self.load()
        self._report(progress_cb, 5)

        from reazonspeech.k2.asr import audio_from_path, transcribe

        audio = audio_from_path(file_path)
        ret = transcribe(self._model, audio)
        self._report(progress_cb, 95)

        segments = []
        for seg in getattr(ret, "segments", []) or []:
            segments.append(
                Segment(
                    text=getattr(seg, "text", ""),
                    start=getattr(seg, "start_seconds", None),
                    end=getattr(seg, "end_seconds", None),
                )
            )

        self._report(progress_cb, 100)
        return TranscriptionResult(
            text=(getattr(ret, "text", "") or "").strip(),
            language="ja",
            segments=segments,
        )
