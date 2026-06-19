"""STTエンジンの抽象基底と正規化された結果型。

各エンジンの出力形（セグメント列 / 話者ごとのdict / テキスト1ブロック）の
違いを :class:`TranscriptionResult` に正規化して吸収する。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, Optional


# 進捗コールバック: 0-100 の整数を受け取る
ProgressCallback = Callable[[int], None]


@dataclass
class Segment:
    """文字起こしの1区間。エンジンによっては start/end/speaker を持たない。"""

    text: str
    start: Optional[float] = None
    end: Optional[float] = None
    speaker: Optional[str] = None


@dataclass
class TranscriptionResult:
    """エンジン横断で正規化した文字起こし結果。"""

    text: str
    language: Optional[str] = None
    segments: List[Segment] = field(default_factory=list)

    @property
    def has_speakers(self) -> bool:
        return any(seg.speaker for seg in self.segments)

    def to_text(self) -> str:
        """ファイル書き出し用のテキストを返す。

        話者情報があれば ``話者: 本文`` を改行区切りで、無ければ ``text`` を返す。
        """
        if self.has_speakers:
            lines = []
            for seg in self.segments:
                speaker = seg.speaker or "SPEAKER"
                lines.append(f"{speaker}: {seg.text.strip()}")
            return "\n".join(lines)
        return self.text


class STTEngine(ABC):
    """全STTエンジンの基底クラス。

    実装の約束:
    - ``load()`` は冪等で、モデルを ``self`` に保持する（2回目以降は即return）。
    - ``transcribe()`` は :class:`TranscriptionResult` を返す。
    - torch などの重い依存は ``load()`` 内で遅延importする。
    """

    name: str = "base"

    def __init__(self, model_id: str, options: Optional[dict] = None):
        self.model_id = model_id
        self.options = options or {}
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """モデルをメモリへロードする（冪等）。"""

    @abstractmethod
    def transcribe(
        self,
        file_path: str,
        language: str = "auto",
        progress_cb: Optional[ProgressCallback] = None,
    ) -> TranscriptionResult:
        """音声/動画ファイルを文字起こしする。"""

    # --- 共通ヘルパ ---------------------------------------------------------

    @staticmethod
    def _report(progress_cb: Optional[ProgressCallback], percent: int) -> None:
        if progress_cb is not None:
            try:
                progress_cb(max(0, min(100, int(percent))))
            except Exception:  # 進捗通知の失敗は本処理を止めない
                pass

    @staticmethod
    def resolve_device():
        """(device, is_cuda) を返す。torch を遅延importする。"""
        import torch

        if torch.cuda.is_available():
            return "cuda", True
        return "cpu", False

    @staticmethod
    def resolve_dtype():
        """GPUに応じた推論dtypeを返す。

        V100(Volta, sm_70) は bfloat16 非対応のため fp16 を使う。
        bf16 が使える環境（Ampere以降）では bf16 を返す。
        """
        import torch

        if not torch.cuda.is_available():
            return torch.float32
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
