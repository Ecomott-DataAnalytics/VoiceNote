"""Faster-Whisper (CTranslate2) エンジン。

旧 voicenote.py の faster-whisper 経路を移植したもの。Whisperカテゴリ
（Large-v3 など）はすべてこのエンジンで処理する。V100 では fp16 で動作する。
"""

from __future__ import annotations

import logging
from typing import Optional

from engines.base import ProgressCallback, Segment, STTEngine, TranscriptionResult

logger = logging.getLogger(__name__)


class FasterWhisperEngine(STTEngine):
    name = "faster-whisper"

    def load(self) -> None:
        if self._loaded:
            return
        from faster_whisper import WhisperModel

        device, is_cuda = self.resolve_device()
        compute_type = "float16" if is_cuda else "int8"
        logger.info(
            "Faster-Whisper モデルをロード中: %s (device=%s, compute_type=%s)",
            self.model_id,
            device,
            compute_type,
        )
        self._model = WhisperModel(self.model_id, device=device, compute_type=compute_type)
        self._loaded = True

    def transcribe(
        self,
        file_path: str,
        language: str = "auto",
        progress_cb: Optional[ProgressCallback] = None,
    ) -> TranscriptionResult:
        self.load()
        self._report(progress_cb, 1)

        lang = None if language == "auto" else language
        segments, info = self._model.transcribe(file_path, language=lang)

        # faster-whisper のセグメントはジェネレータ。進捗算出のため一旦展開する。
        segments_list = list(segments)
        total = len(segments_list) or 1

        out_segments = []
        texts = []
        for i, seg in enumerate(segments_list):
            progress = min(100, int((i + 1) / total * 100))
            if i % 10 == 0 or progress == 100:
                self._report(progress_cb, progress)
            texts.append(seg.text)
            out_segments.append(Segment(text=seg.text, start=seg.start, end=seg.end))

        self._report(progress_cb, 100)
        return TranscriptionResult(
            text="".join(texts).strip(),
            language=getattr(info, "language", None),
            segments=out_segments,
        )
