import importlib
import unittest

yaml_available = importlib.util.find_spec("yaml") is not None

if yaml_available:
    import config

VALID_ENGINES = {"faster-whisper", "kotoba", "reazon", "qwen3"}


@unittest.skipUnless(yaml_available, "PyYAML is required")
class ConfigTestCase(unittest.TestCase):
    def test_models_load(self):
        models = config.list_models()
        self.assertTrue(models, "models.yaml にモデルが定義されていること")

    def test_required_fields(self):
        for m in config.list_models():
            self.assertTrue(m.id, "id は必須")
            self.assertIn(m.engine, VALID_ENGINES, f"未知のengine: {m.engine}")
            self.assertTrue(m.category, "category は必須")
            self.assertTrue(m.label, "label は必須")
            # UIに表示するユーザー向けの特徴説明は全モデルに必須
            self.assertTrue(m.description, f"description は必須: {m.id}")
            self.assertIsInstance(m.options, dict)

    def test_ui_payload_has_description(self):
        ui = config.grouped_for_ui()
        for group in ui["groups"]:
            for m in group["models"]:
                self.assertIn("description", m)
                self.assertTrue(m["description"])

    def test_default_is_valid(self):
        ids = {m.id for m in config.list_models()}
        self.assertIn(config.get_default(), ids)

    def test_get_model_unknown_raises(self):
        with self.assertRaises(KeyError):
            config.get_model("no-such-model")

    def test_grouped_for_ui_shape(self):
        ui = config.grouped_for_ui()
        self.assertIn("default", ui)
        self.assertIn("groups", ui)
        self.assertTrue(ui["groups"])
        for group in ui["groups"]:
            self.assertIn("category", group)
            self.assertIn("models", group)
            for m in group["models"]:
                self.assertIn("id", m)
                self.assertIn("label", m)


if __name__ == "__main__":
    unittest.main()
