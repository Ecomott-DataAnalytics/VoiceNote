"""Kotoba-Whisper エンジン（transformers pipeline）。

``options.diarization`` で2モードに分岐する:
- True  : v2.2 相当。話者分離(pyannote)付き。HFトークンと pyannote gated
          モデルの利用規約承認が必要。出力は話者ラベル付き。
- False : v2.0/v2.1 相当。素のASR。トークン不要。

進捗はチャンク非同期処理のため粗い（開始 → 完了の2点）。

注: kotoba-whisper-v2.2 の出力dictのキー構造はバージョンで揺れるため、
``_parse_diarized`` は複数の形を許容して防御的にパースする。実機（GPUホスト）
で実際の出力形を一度確認すること。
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from engines.base import ProgressCallback, Segment, STTEngine, TranscriptionResult

logger = logging.getLogger(__name__)


class KotobaEngine(STTEngine):
    name = "kotoba"

    @property
    def diarization(self) -> bool:
        return bool(self.options.get("diarization", False))

    def load(self) -> None:
        if self._loaded:
            return
        import torch
        from transformers import pipeline

        device, is_cuda = self.resolve_device()
        device_str = "cuda:0" if is_cuda else "cpu"
        torch_dtype = torch.float16 if is_cuda else torch.float32
        model_kwargs = {"attn_implementation": "sdpa"} if is_cuda else {}

        # 話者分離(v2.2)は pyannote gated モデルへアクセスするため HF トークンが要る。
        if self.diarization:
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            if token:
                try:
                    from huggingface_hub import login

                    login(token=token, add_to_git_credential=False)
                except Exception as e:  # ログだけ残して継続（pipeline側でも解決され得る）
                    logger.warning("HuggingFace login に失敗: %s", e)
            else:
                logger.warning(
                    "話者分離(v2.2)には HF_TOKEN が必要です。未設定のためロードに失敗する可能性があります。"
                )

        logger.info(
            "Kotoba モデルをロード中: %s (device=%s, diarization=%s)",
            self.model_id,
            device_str,
            self.diarization,
        )
        def _build_pipeline(*args, **kwargs):
            # dtype は model_kwargs に入れる。新しい transformers では top-level の
            # dtype=/torch_dtype= を model_kwargs と併用すると（model_kwargs 側に dtype が
            # 無くても）dtype 競合の ValueError になるため。新版は "dtype"、旧版は "torch_dtype"。
            mk = dict(kwargs.pop("model_kwargs", {}) or {})
            try:
                return pipeline(*args, model_kwargs={**mk, "dtype": torch_dtype}, **kwargs)
            except TypeError:
                return pipeline(*args, model_kwargs={**mk, "torch_dtype": torch_dtype}, **kwargs)

        if self.diarization:
            # v2.2: 話者分離・句読点付与はモデル同梱のカスタムパイプラインが担う。
            # task を "automatic-speech-recognition" に固定すると標準 Whisper パイプラインに
            # なり add_punctuation が拒否されるため、task は指定せず trust_remote_code で
            # カスタムパイプラインをロードする。
            # また PyTorch 2.6 で torch.load の既定が weights_only=True になり、pyannote の
            # チェックポイント(TorchVersion 等のグローバルを含む)が読めなくなるため、信頼できる
            # ローカルモデルに限り読み込み中だけ weights_only=False を強制する。
            _orig_torch_load = torch.load

            def _trusting_torch_load(*a, **k):
                k["weights_only"] = False  # lightning が True を明示するため setdefault では不可
                return _orig_torch_load(*a, **k)

            torch.load = _trusting_torch_load
            try:
                self._pipe = _build_pipeline(
                    model=self.model_id,
                    device=device_str,
                    model_kwargs=model_kwargs,
                    batch_size=8,
                    trust_remote_code=True,
                )
            finally:
                torch.load = _orig_torch_load
        else:
            self._pipe = _build_pipeline(
                "automatic-speech-recognition",
                model=self.model_id,
                device=device_str,
                model_kwargs=model_kwargs,
                batch_size=8,
                trust_remote_code=True,
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

        if self.diarization:
            result = self._pipe(file_path, chunk_length_s=15, add_punctuation=True)
            out = self._parse_diarized(result)
        else:
            result = self._pipe(file_path, chunk_length_s=15, return_timestamps=True)
            out = self._parse_plain(result)

        self._report(progress_cb, 100)
        return out

    # --- パーサ -------------------------------------------------------------

    @staticmethod
    def _parse_plain(result) -> TranscriptionResult:
        if isinstance(result, str):
            return TranscriptionResult(text=result.strip())
        text = (result.get("text") or "").strip()
        segments = []
        for ch in result.get("chunks", []) or []:
            ts = ch.get("timestamp") or (None, None)
            segments.append(
                Segment(text=ch.get("text", ""), start=ts[0], end=ts[1])
            )
        return TranscriptionResult(text=text, language="ja", segments=segments)

    @staticmethod
    def _parse_diarized(result) -> TranscriptionResult:
        """話者分離結果を防御的にパースする。

        想定する形のいずれかに対応:
        (a) result["chunks"] = [{"speaker"/"speaker_id", "timestamp", "text"}, ...]
        (b) result のキーが "chunks/SPEAKER_XX" / "text/SPEAKER_XX" 形式
        (c) いずれも無ければ result["text"] にフォールバック
        """
        if isinstance(result, str):
            return TranscriptionResult(text=result.strip())

        segments = []

        # (a) フラットな chunks に speaker が含まれる形
        chunks = result.get("chunks")
        if isinstance(chunks, list) and chunks and any(
            ("speaker" in c or "speaker_id" in c) for c in chunks if isinstance(c, dict)
        ):
            for c in chunks:
                if not isinstance(c, dict):
                    continue
                ts = c.get("timestamp") or (None, None)
                segments.append(
                    Segment(
                        text=c.get("text", ""),
                        start=ts[0],
                        end=ts[1],
                        speaker=str(c.get("speaker") or c.get("speaker_id") or "SPEAKER"),
                    )
                )

        # (b) "chunks/SPEAKER_XX" 形式のキー
        if not segments:
            for key, value in result.items():
                if isinstance(key, str) and key.startswith("chunks/") and isinstance(value, list):
                    speaker = key.split("/", 1)[1]
                    for c in value:
                        if not isinstance(c, dict):
                            continue
                        ts = c.get("timestamp") or (None, None)
                        segments.append(
                            Segment(text=c.get("text", ""), start=ts[0], end=ts[1], speaker=speaker)
                        )
            segments.sort(key=lambda s: (s.start if s.start is not None else 0.0))

        if segments:
            text = "\n".join(
                f"{s.speaker}: {s.text.strip()}" for s in segments if s.text.strip()
            )
            return TranscriptionResult(text=text, language="ja", segments=segments)

        # (c) フォールバック
        text = result.get("text")
        if not text:
            # "text/SPEAKER_XX" を結合
            parts = [v for k, v in result.items() if isinstance(k, str) and k.startswith("text/")]
            text = "\n".join(str(p) for p in parts)
        return TranscriptionResult(text=(text or "").strip(), language="ja")
