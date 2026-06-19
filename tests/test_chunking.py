import unittest
from unittest.mock import patch

from engines.base import Segment, TranscriptionResult
from engines import chunking


class FakeEngine:
    """transcribe(file_path, language, progress_cb) を持つだけのダミーエンジン。"""

    def __init__(self):
        self.calls = []

    def transcribe(self, file_path, language="auto", progress_cb=None):
        idx = len(self.calls)
        self.calls.append(file_path)
        return TranscriptionResult(
            text=f"text{idx}",
            language="ja",
            segments=[Segment(text=f"text{idx}", start=0.0, end=5.0)],
        )


class ChunkingTestCase(unittest.TestCase):
    def test_concatenates_text_and_offsets_segments(self):
        fake_chunks = ["c0.wav", "c1.wav", "c2.wav"]
        engine = FakeEngine()
        progress = []

        with patch.object(chunking, "split_audio", return_value=fake_chunks):
            result = chunking.transcribe_chunked(
                engine, "input.mp4", language="ja",
                progress_cb=progress.append, chunk_seconds=30,
            )

        # 各チャンクが順番に処理される
        self.assertEqual(engine.calls, fake_chunks)
        # テキストは改行で連結
        self.assertEqual(result.text, "text0\ntext1\ntext2")
        # セグメントのタイムスタンプは chunk_seconds 分オフセットされる
        starts = [s.start for s in result.segments]
        ends = [s.end for s in result.segments]
        self.assertEqual(starts, [0.0, 30.0, 60.0])
        self.assertEqual(ends, [5.0, 35.0, 65.0])
        self.assertEqual(result.language, "ja")
        # 進捗は最後に100で締める
        self.assertEqual(progress[-1], 100)
        self.assertTrue(all(0 <= p <= 100 for p in progress))

    def test_empty_split_falls_back_to_direct(self):
        engine = FakeEngine()
        with patch.object(chunking, "split_audio", return_value=[]):
            result = chunking.transcribe_chunked(engine, "input.mp4", chunk_seconds=30)
        # 分割できなければ元ファイルをそのまま1回処理
        self.assertEqual(engine.calls, ["input.mp4"])
        self.assertEqual(result.text, "text0")


if __name__ == "__main__":
    unittest.main()
