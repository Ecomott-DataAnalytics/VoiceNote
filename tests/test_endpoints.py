import io
import unittest
import importlib
from unittest.mock import patch, MagicMock

flask_spec = importlib.util.find_spec("flask")
celery_spec = importlib.util.find_spec("celery")
flask_available = flask_spec is not None and celery_spec is not None

if flask_available:
    from voicenote import app
else:
    app = None


@unittest.skipUnless(flask_available, "Flask and Celery are required")
class EndpointsTestCase(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_home_status_code(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

    @patch('voicenote.transcribe_audio.delay')
    def test_transcribe_returns_202(self, mock_delay):
        mock_task = MagicMock()
        mock_task.id = 'dummy-task-id'
        mock_delay.return_value = mock_task

        data = {
            'file': (io.BytesIO(b'dummy'), 'dummy.wav')
        }
        response = self.client.post('/transcribe', data=data,
                                    content_type='multipart/form-data')
        self.assertEqual(response.status_code, 202)


if __name__ == '__main__':
    unittest.main()
