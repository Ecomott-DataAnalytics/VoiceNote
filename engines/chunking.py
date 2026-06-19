"""長尺音声の分割処理。

入力全体を1パスで処理するエンジン（Reazon k2 / Qwen3-ASR 簡易API）向けに、
ffmpeg で音声を一定長のWAVへ分割し、チャンクごとに逐次文字起こしする。
これにより長尺ファイルでもメモリ爆発を避け、実際の進捗を報告できる。

ffmpeg はコンテナ(voicenote.def)に同梱。WAV(16k/mono)へ正規化するため
mp4等の動画コンテナのデコードもここで吸収される。
"""

from __future__ import annotations

import glob
import logging
import os
import subprocess
import tempfile
from typing import Optional

from engines.base import ProgressCallback, Segment, STTEngine, TranscriptionResult

logger = logging.getLogger(__name__)

# 1チャンクの長さ(秒)。環境変数 VN_CHUNK_SECONDS で上書き可。
def _chunk_seconds() -> int:
    try:
        return max(5, int(os.environ.get("VN_CHUNK_SECONDS", "30")))
    except ValueError:
        return 30


def to_wav(file_path: str) -> str:
    """任意のメディアを 16k/mono WAV に正規化して一時ファイルパスを返す。

    ffmpeg に**ファイルパス**を渡す（シーク可能）ため、transformers pipeline の
    stdin パイプ方式では失敗する mp4 等のコンテナも確実にデコードできる。
    呼び出し側で生成ファイルを削除すること。
    """
    import uuid

    out = os.path.join(tempfile.gettempdir(), f"vn_norm_{uuid.uuid4().hex}.wav")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", file_path,
        "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        out,
    ]
    logger.info("入力を 16k/mono WAV に正規化します: %s", file_path)
    subprocess.run(cmd, check=True)
    return out


def split_audio(file_path: str, out_dir: str, chunk_seconds: int) -> list:
    """ffmpeg で音声を 16k/mono WAV の連番チャンクへ分割し、パス一覧を返す。"""
    pattern = os.path.join(out_dir, "chunk_%05d.wav")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", file_path,
        "-vn",                       # 映像を捨てる
        "-ar", "16000", "-ac", "1",  # 16kHz / モノラルに正規化
        "-f", "segment",
        "-segment_time", str(chunk_seconds),
        "-c:a", "pcm_s16le",
        pattern,
    ]
    logger.info("ffmpeg で %ds 単位に分割します: %s", chunk_seconds, file_path)
    subprocess.run(cmd, check=True)
    return sorted(glob.glob(os.path.join(out_dir, "chunk_*.wav")))


def transcribe_chunked(
    engine: STTEngine,
    file_path: str,
    language: str = "auto",
    progress_cb: Optional[ProgressCallback] = None,
    chunk_seconds: Optional[int] = None,
) -> TranscriptionResult:
    """`engine` を使って、分割した各チャンクを逐次文字起こしして結合する。"""
    chunk_seconds = chunk_seconds or _chunk_seconds()
    tmp_dir = tempfile.mkdtemp(prefix="vn_chunks_")
    try:
        chunks = split_audio(file_path, tmp_dir, chunk_seconds)
        if not chunks:
            logger.warning("分割結果が空のため、分割なしで処理します")
            return engine.transcribe(file_path, language=language, progress_cb=progress_cb)

        total = len(chunks)
        logger.info("%d 個のチャンクに分割しました", total)

        texts = []
        all_segments = []
        language_detected = None

        for i, chunk in enumerate(chunks):
            res = engine.transcribe(chunk, language=language, progress_cb=None)
            offset = i * chunk_seconds

            if res.text and res.text.strip():
                texts.append(res.text.strip())
            for seg in res.segments:
                all_segments.append(
                    Segment(
                        text=seg.text,
                        start=(seg.start + offset) if seg.start is not None else None,
                        end=(seg.end + offset) if seg.end is not None else None,
                        speaker=seg.speaker,
                    )
                )
            language_detected = language_detected or res.language

            pct = min(99, int((i + 1) / total * 100))
            if progress_cb is not None:
                progress_cb(pct)
            logger.info("チャンク %d/%d 完了 (%d%%)", i + 1, total, pct)

        if progress_cb is not None:
            progress_cb(100)

        return TranscriptionResult(
            text="\n".join(texts),
            language=language_detected,
            segments=all_segments,
        )
    finally:
        try:
            for f in glob.glob(os.path.join(tmp_dir, "*")):
                os.remove(f)
            os.rmdir(tmp_dir)
        except Exception as e:
            logger.warning("チャンク一時ファイルの削除に失敗: %s", e)
