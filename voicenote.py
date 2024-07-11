import multiprocessing
from flask import Flask, request, send_file, render_template_string, jsonify
from faster_whisper import WhisperModel
import os
import torch
from celery import Celery
import uuid
import logging
from werkzeug.exceptions import RequestEntityTooLarge
from celery.exceptions import SoftTimeLimitExceeded
import time
import psutil
import GPUtil

# Set multiprocessing start method to 'spawn'
multiprocessing.set_start_method('spawn', force=True)

# Flask app configuration
app = Flask(__name__)
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024 * 1024  # 4GB

# Celery configuration
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.result_backend = 'redis://localhost:6379/0'
celery.conf.broker_url = 'redis://localhost:6379/0'
celery.conf.broker_connection_retry_on_startup = True
celery.conf.update(app.config)

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CUDA and device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
logger.info(f"Using device: {device}, Compute type: {compute_type}")

# Model storage
models = {}

# File storage configuration
UPLOAD_FOLDER = '/tmp/voicenote_uploads'
RESULT_FOLDER = '/tmp/voicenote_results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# HTML template (unchanged)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>VoiceNote</title>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <meta name="viewport" content="width=device-width,initial-scale=1">
</head>
<body>
    <h1>Faster WhisperによるAudio/Videoの文字起こしサービスです</h1>
    <form id="transcription-form" enctype="multipart/form-data">
        <input type="file" name="file" accept="audio/*,video/*"><br><br>
        <label for="language">文字起こししたい言語を指定（不明な場合はAuto）:</label>
        <select name="language" id="language">
            <option value="ja">Japanese</option>
            <option value="auto">Auto-detect</option>
            <option value="en">English</option>
        </select><br><br>
        <label for="model">モデル選択:</label>
        <select name="model" id="model">
            <option value="large-v3">Large-v3</option>
            <option value="medium">Medium</option>
        </select><br><br>
        <input type="submit" value="Transcribe">
    </form>
    <div id="status"></div>
    <div id="progress"></div>
    <div id="result"></div>
    <script>
    $(document).ready(function() {
        $('#transcription-form').submit(function(event) {
            event.preventDefault();
            var formData = new FormData(this);
            $.ajax({
                url: '/transcribe',
                type: 'POST',
                data: formData,
                processData: false,
                contentType: false,
                success: function(data) {
                    $('#status').text('Transcription in progress...');
                    checkStatus(data.task_id);
                },
                error: function() {
                    $('#status').text('Error occurred while submitting the task.');
                }
            });
        });
        function checkStatus(taskId) {
            $.get('/status/' + taskId, function(data) {
                $('#status').text('Status: ' + data.status);
                if (data.state === 'PROGRESS') {
                    $('#progress').text('Progress: ' + data.progress + '%');
                }
                if (data.state === 'SUCCESS') {
                    $('#result').html('<a href="/result/' + taskId + '" target="_blank">完成ファイルのダウンロード</a>');
                } else if (data.state === 'FAILURE') {
                    $('#status').text('Transcription failed: ' + data.error);
                } else {
                    setTimeout(function() {
                        checkStatus(taskId);
                    }, 5000);  // 5秒ごとにステータスをチェック
                }
            });
        }
    });
    </script>
</body>
</html>
"""

def log_system_resources():
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    gpu_info = GPUtil.getGPUs()[0] if GPUtil.getGPUs() else None
    
    logger.info(f"CPU Usage: {cpu_percent}%")
    logger.info(f"Memory Usage: {memory.percent}% (Used: {memory.used / 1024 / 1024:.2f} MB, Available: {memory.available / 1024 / 1024:.2f} MB)")
    logger.info(f"Disk Usage: {disk.percent}% (Used: {disk.used / 1024 / 1024 / 1024:.2f} GB, Free: {disk.free / 1024 / 1024 / 1024:.2f} GB)")
    if gpu_info:
        logger.info(f"GPU Usage: {gpu_info.memoryUsed}MB / {gpu_info.memoryTotal}MB")

def load_model(model_name):
    if model_name not in models:
        logger.info(f"Loading model: {model_name}")
        try:
            models[model_name] = WhisperModel(model_name, device=device, compute_type=compute_type)
            logger.info(f"Model {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise
    return models[model_name]

@celery.task(bind=True, soft_time_limit=7200, time_limit=7500)
def transcribe_audio(self, file_path, language, model_name, original_filename):
    try:
        start_time = time.time()
        logger.info(f"Starting transcription: file={file_path}, language={language}, model={model_name}")
        log_system_resources()

        model = load_model(model_name)
        
        logger.info("Starting transcription")
        segments, info = model.transcribe(file_path, language=language if language != 'auto' else None)
        
        logger.info("Processing segments")
        segments_list = list(segments)
        total_segments = len(segments_list)
        
        output_filename = f"{uuid.uuid4()}.txt"
        output_file = os.path.join(RESULT_FOLDER, output_filename)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            #f.write(f"Detected language: {info.language}\n\n")
            for i, segment in enumerate(segments_list):
                progress = min(100, int((i + 1) / total_segments * 100))
                if i % 10 == 0 or progress == 100:
                    self.update_state(state='PROGRESS', meta={'progress': progress})
                    logger.info(f"Transcription progress: {progress}%")
                    #log_system_resources()
                
                # Write segment text
                f.write(f"{segment.text} ")

        logger.info(f"Transcription completed: {total_segments} segments")

        # 音声ファイルの削除
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Removed input file: {file_path}")
            else:
                logger.warning(f"Input file not found for deletion: {file_path}")
        except Exception as e:
            logger.error(f"Failed to remove input file {file_path}: {str(e)}")

        total_time = time.time() - start_time
        logger.info(f"Total task time: {total_time:.2f} seconds")
        log_system_resources()

        return output_file, original_filename

    except SoftTimeLimitExceeded:
        logger.error("Soft time limit exceeded in transcribe_audio")
        raise
    except Exception as e:
        logger.error(f"Error in transcribe_audio: {str(e)}", exc_info=True)
        # エラーが発生した場合でも、可能であれば入力ファイルを削除
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Removed input file after error: {file_path}")
        except Exception as del_e:
            logger.error(f"Failed to remove input file after error {file_path}: {str(del_e)}")
        raise

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    language = request.form.get('language', 'auto')
    model_name = request.form.get('model', 'medium')
    
    original_filename = file.filename
    file_extension = os.path.splitext(original_filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(file_path)
    logger.info(f"File saved: {file_path}")
    
    task = transcribe_audio.delay(file_path, language, model_name, original_filename)
    logger.info(f"Transcription task started: {task.id}")
    
    return jsonify({"task_id": task.id}), 202

@app.route('/status/<task_id>')
def task_status(task_id):
    task = transcribe_audio.AsyncResult(task_id)
    logger.info(f"Checking status for task: {task_id}, State: {task.state}")
    
    if task.state == 'PENDING':
        response = {'state': task.state, 'status': 'Task is waiting for execution.'}
    elif task.state == 'PROGRESS':
        response = {
            'state': task.state,
            'status': 'Task is in progress.',
            'progress': task.info.get('progress', 0)
        }
    elif task.state == 'SUCCESS':
        response = {'state': task.state, 'status': 'Task completed successfully.'}
    elif task.state == 'FAILURE':
        response = {'state': task.state, 'status': 'Task failed.', 'error': str(task.result)}
    else:
        response = {'state': task.state, 'status': 'Task is in an unknown state.'}
    
    return jsonify(response)

@app.route('/result/<task_id>')
def get_result(task_id):
    task = transcribe_audio.AsyncResult(task_id)
    logger.info(f"Retrieving result for task: {task_id}, State: {task.state}")
    
    if task.state == 'SUCCESS':
        result_file, original_filename = task.result
        if os.path.exists(result_file):
            try:
                # 元のファイル名の拡張子を.txtに変更
                download_name = os.path.splitext(original_filename)[0] + '.txt'
                return_value = send_file(result_file, as_attachment=True, download_name=download_name)
                
                # ファイルを送信した後、非同期でファイルを削除
                @return_value.call_on_close
                def delete_file_after_send():
                    try:
                        os.remove(result_file)
                        logger.info(f"Removed result file after sending: {result_file}")
                    except Exception as e:
                        logger.error(f"Failed to remove result file {result_file}: {str(e)}")
                
                return return_value
            except Exception as e:
                logger.error(f"Error while sending file {result_file}: {str(e)}")
                return jsonify({"error": "Error occurred while sending file"}), 500
        else:
            logger.warning(f"Result file not found: {result_file}")
            return jsonify({"error": "Result file not found"}), 404
    elif task.state == 'FAILURE':
        logger.error(f"Task failed: {task_id}")
        return jsonify({"error": "Task failed", "details": str(task.result)}), 500
    else:
        logger.info(f"Result not ready for task: {task_id}")
        return jsonify({"error": "Result not ready"}), 404

@app.errorhandler(RequestEntityTooLarge)
def handle_request_entity_too_large(error):
    return jsonify({"error": "File is too large"}), 413

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
