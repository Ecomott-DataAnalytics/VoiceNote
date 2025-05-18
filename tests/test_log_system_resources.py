import importlib.util
import os
import sys
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))
VOICENOTE_PATH = os.path.join(ROOT, 'voicenote.py')

def load_voicenote(get_gpus_func):
    # remove previously loaded module
    if 'voicenote' in sys.modules:
        del sys.modules['voicenote']
    stubs = {}

    def register(name, module):
        stubs[name] = sys.modules.get(name)
        sys.modules[name] = module

    # flask stub
    flask_mod = types.ModuleType('flask')
    class Flask:
        def __init__(self, *a, **k):
            self.name = 'app'
            self.config = {}
        def route(self, *ra, **rk):
            def decorator(func):
                return func
            return decorator
        def errorhandler(self, *ea, **ek):
            def decorator(func):
                return func
            return decorator
        def run(self, *a, **k):
            pass
    flask_mod.Flask = Flask
    flask_mod.request = None
    flask_mod.send_file = lambda *a, **k: None
    flask_mod.render_template = lambda *a, **k: None
    flask_mod.jsonify = lambda *a, **k: None
    register('flask', flask_mod)

    # celery stub
    celery_mod = types.ModuleType('celery')
    class Celery:
        def __init__(self, *a, **k):
            self.conf = types.SimpleNamespace(update=lambda *a, **k: None)
        def task(self, *ta, **tkw):
            def decorator(func):
                return func
            return decorator
    celery_mod.Celery = Celery
    register('celery', celery_mod)

    celery_exc = types.ModuleType('celery.exceptions')
    class SoftTimeLimitExceeded(Exception):
        pass
    celery_exc.SoftTimeLimitExceeded = SoftTimeLimitExceeded
    register('celery.exceptions', celery_exc)
    celery_mod.exceptions = celery_exc

    # werkzeug stub
    werkzeug_mod = types.ModuleType('werkzeug')
    werkzeug_exc = types.ModuleType('werkzeug.exceptions')
    class RequestEntityTooLarge(Exception):
        pass
    werkzeug_exc.RequestEntityTooLarge = RequestEntityTooLarge
    werkzeug_mod.exceptions = werkzeug_exc
    register('werkzeug', werkzeug_mod)
    register('werkzeug.exceptions', werkzeug_exc)

    # whisper stub
    whisper_mod = types.ModuleType('whisper')
    whisper_mod.load_model = lambda *a, **k: None
    register('whisper', whisper_mod)

    # faster_whisper stub
    fw_mod = types.ModuleType('faster_whisper')
    class WhisperModel:
        pass
    fw_mod.WhisperModel = WhisperModel
    register('faster_whisper', fw_mod)

    # torch stub
    torch_mod = types.ModuleType('torch')
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
    register('torch', torch_mod)

    # psutil stub
    psutil_mod = types.ModuleType('psutil')
    psutil_mod.cpu_percent = lambda: 0
    psutil_mod.virtual_memory = lambda: types.SimpleNamespace(percent=0, used=0, available=0)
    psutil_mod.disk_usage = lambda path: types.SimpleNamespace(percent=0, used=0, free=0)
    register('psutil', psutil_mod)

    # GPUtil stub
    gputil_mod = types.ModuleType('GPUtil')
    gputil_mod.getGPUs = get_gpus_func
    register('GPUtil', gputil_mod)

    spec = importlib.util.spec_from_file_location('voicenote', VOICENOTE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules['voicenote'] = module
    spec.loader.exec_module(module)

    def cleanup():
        for name, original in stubs.items():
            if original is None:
                del sys.modules[name]
            else:
                sys.modules[name] = original
        if 'voicenote' in sys.modules:
            del sys.modules['voicenote']
    return module, cleanup

class LogSystemResourcesTests(unittest.TestCase):
    def test_gpu_exception(self):
        def raise_exc():
            raise RuntimeError('no gpu')
        module, cleanup = load_voicenote(raise_exc)
        self.addCleanup(cleanup)
        with self.assertLogs(module.logger, level='WARNING') as cm:
            module.log_system_resources()
        self.assertTrue(any('GPU' in m for m in cm.output))

    def test_gpu_empty(self):
        module, cleanup = load_voicenote(lambda: [])
        self.addCleanup(cleanup)
        with self.assertLogs(module.logger, level='WARNING') as cm:
            module.log_system_resources()
        self.assertTrue(any('GPU' in m for m in cm.output))

if __name__ == '__main__':
    unittest.main()
