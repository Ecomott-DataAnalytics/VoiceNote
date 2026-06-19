"""Qwen3-ASR エンジン（独自パッケージ qwen-asr）。

``from qwen_asr import Qwen3ASRModel`` を利用。0.6B / 1.7B に対応。

V100(Volta, sm_70) は bfloat16 / FlashAttention-2 非対応のため、dtype は
``resolve_dtype()`` で自動判定し（V100→fp16）、attention は sdpa を使う。
既定は transformers バックエンド。``QWEN_ASR_BACKEND=vllm`` で vLLM に切替。

簡易APIはテキスト1ブロックを返す（タイムスタンプ/セグメントなし）。進捗は粗い。
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from engines.base import ProgressCallback, STTEngine, TranscriptionResult

logger = logging.getLogger(__name__)

# VoiceNote の言語コード -> Qwen が期待する言語名
_LANG_MAP = {
    "auto": None,
    "ja": "Japanese",
    "en": "English",
}


class Qwen3ASREngine(STTEngine):
    name = "qwen3"

    def load(self) -> None:
        if self._loaded:
            return
        from qwen_asr import Qwen3ASRModel

        _, is_cuda = self.resolve_device()
        backend = os.environ.get("QWEN_ASR_BACKEND", "transformers").lower()

        if backend == "vllm":
            logger.info("Qwen3-ASR を vLLM バックエンドでロード中: %s", self.model_id)
            self._model = Qwen3ASRModel.LLM(
                model=self.model_id,
                gpu_memory_utilization=0.7,
                max_new_tokens=4096,
            )
        else:
            dtype = self.resolve_dtype()  # V100 では fp16
            device_map = "cuda:0" if is_cuda else "cpu"
            logger.info(
                "Qwen3-ASR を transformers バックエンドでロード中: %s (dtype=%s, device_map=%s)",
                self.model_id,
                dtype,
                device_map,
            )
            kwargs = dict(
                dtype=dtype,
                device_map=device_map,
                attn_implementation="sdpa",  # V100 は FA2 非対応
                max_inference_batch_size=8,
                max_new_tokens=448,
            )
            try:
                self._model = Qwen3ASRModel.from_pretrained(self.model_id, **kwargs)
            except TypeError as e:
                # qwen-asr のバージョン差で受け付けないkwargがあれば外して再試行
                logger.warning("from_pretrained kwarg非対応のため縮退します: %s", e)
                self._model = Qwen3ASRModel.from_pretrained(
                    self.model_id, dtype=dtype, device_map=device_map
                )
        self._loaded = True

    def transcribe(
        self,
        file_path: str,
        language: str = "auto",
        progress_cb: Optional[ProgressCallback] = None,
    ) -> TranscriptionResult:
        self.load()
        self._report(progress_cb, 5)

        qwen_lang = _LANG_MAP.get(language, None)
        results = self._model.transcribe(audio=file_path, language=qwen_lang)
        self._report(progress_cb, 95)

        first = results[0] if results else None
        text = (getattr(first, "text", "") or "").strip() if first else ""
        detected = getattr(first, "language", None) if first else None

        self._report(progress_cb, 100)
        return TranscriptionResult(text=text, language=detected)
