import io
import unittest
import importlib
from unittest.mock import patch, MagicMock

flask_spec = importlib.util.find_spec("flask")
celery_spec = importlib.util.find_spec("celery")
yaml_spec = importlib.util.find_spec("yaml")
psutil_spec = importlib.util.find_spec("psutil")
gputil_spec = importlib.util.find_spec("GPUtil")
app_available = all(
    spec is not None
    for spec in (flask_spec, celery_spec, yaml_spec, psutil_spec, gputil_spec)
)

if app_available:
    from voicenote import app
    import config
else:
    app = None


@unittest.skipUnless(app_available, "Flask, Celery, PyYAML, psutil, GPUtil are required")
class EndpointsTestCase(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_home_status_code(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

    def test_models_endpoint(self):
        response = self.client.get('/models')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('default', data)
        self.assertIn('groups', data)
        self.assertTrue(len(data['groups']) > 0)
        # 各グループに category と models がある
        for group in data['groups']:
            self.assertIn('category', group)
            self.assertIn('models', group)

    @patch('voicenote.transcribe_audio.delay')
    def test_transcribe_returns_202(self, mock_delay):
        mock_task = MagicMock()
        mock_task.id = 'dummy-task-id'
        mock_delay.return_value = mock_task

        data = {
            'file': (io.BytesIO(b'dummy'), 'dummy.wav'),
            'model': config.get_default(),
        }
        response = self.client.post('/transcribe', data=data,
                                    content_type='multipart/form-data')
        self.assertEqual(response.status_code, 202)

    @patch('voicenote.transcribe_audio.delay')
    def test_transcribe_rejects_unknown_model(self, mock_delay):
        data = {
            'file': (io.BytesIO(b'dummy'), 'dummy.wav'),
            'model': 'does-not-exist',
        }
        response = self.client.post('/transcribe', data=data,
                                    content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        mock_delay.assert_not_called()


if __name__ == '__main__':
    unittest.main()
