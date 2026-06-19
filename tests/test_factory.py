import unittest

# engines パッケージは重いML依存をload()内で遅延importするため、
# Factory経由のインスタンス生成はGPU/torch等が無くても実行できる。
from engines.factory import EngineFactory
from engines.faster_whisper_engine import FasterWhisperEngine
from engines.kotoba_engine import KotobaEngine
from engines.reazon_engine import ReazonEngine
from engines.qwen3_asr_engine import Qwen3ASREngine


class FactoryTestCase(unittest.TestCase):
    def setUp(self):
        EngineFactory.clear_cache()

    def test_create_by_engine_name(self):
        self.assertIsInstance(
            EngineFactory.create("faster-whisper", "large-v3"), FasterWhisperEngine
        )
        self.assertIsInstance(
            EngineFactory.create("kotoba", "kotoba-tech/kotoba-whisper-v2.2"), KotobaEngine
        )
        self.assertIsInstance(
            EngineFactory.create("reazon", "reazon-research/reazonspeech-k2-v2"),
            ReazonEngine,
        )
        self.assertIsInstance(
            EngineFactory.create("qwen3", "Qwen/Qwen3-ASR-0.6B"), Qwen3ASREngine
        )

    def test_whisper_alias_maps_to_faster_whisper(self):
        self.assertIsInstance(
            EngineFactory.create("whisper", "large-v3"), FasterWhisperEngine
        )

    def test_engine_inferred_from_model_id(self):
        # engine 未指定 → モデルIDの接頭辞から推定
        self.assertIsInstance(
            EngineFactory.create(None, "Qwen/Qwen3-ASR-1.7B"), Qwen3ASREngine
        )
        self.assertIsInstance(
            EngineFactory.create("", "reazon-research/reazonspeech-k2-v2"), ReazonEngine
        )
        self.assertIsInstance(
            EngineFactory.create(None, "kotoba-tech/kotoba-whisper-v2.0"), KotobaEngine
        )
        # 不明 → faster-whisper にフォールバック
        self.assertIsInstance(
            EngineFactory.create(None, "some-unknown-model"), FasterWhisperEngine
        )

    def test_instances_are_cached(self):
        a = EngineFactory.create("reazon", "reazon-research/reazonspeech-k2-v2")
        b = EngineFactory.create("reazon", "reazon-research/reazonspeech-k2-v2")
        self.assertIs(a, b)

    def test_options_are_passed(self):
        engine = EngineFactory.create(
            "kotoba", "kotoba-tech/kotoba-whisper-v2.2", {"diarization": True}
        )
        self.assertTrue(engine.diarization)
        engine2 = EngineFactory.create(
            "kotoba", "kotoba-tech/kotoba-whisper-v2.0", {"diarization": False}
        )
        self.assertFalse(engine2.diarization)


if __name__ == "__main__":
    unittest.main()
