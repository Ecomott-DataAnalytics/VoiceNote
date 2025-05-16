import multiprocessing
from flask import Flask, request, send_file, render_template_string, jsonify
# 追加: whisperのインポート
import whisper
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

# マルチプロセッシングの開始方法を'spawn'に設定
multiprocessing.set_start_method('spawn', force=True)

# Flaskアプリケーション設定
app = Flask(__name__)
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024 * 1024  # 4GB

# Celery設定
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.result_backend = 'redis://localhost:6379/0'
celery.conf.broker_url = 'redis://localhost:6379/0'
celery.conf.broker_connection_retry_on_startup = True
celery.conf.update(app.config)

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CUDAとデバイス設定
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
logger.info(f"使用デバイス: {device}, 計算タイプ: {compute_type}")

# モデルストレージ
# (engine, model_name) の組み合わせでキャッシュ
models = {}

# ファイルストレージ設定
UPLOAD_FOLDER = '/tmp/voicenote_uploads'
RESULT_FOLDER = '/tmp/voicenote_results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# モダンなHTMLテンプレート
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <title>VoiceNote - 音声・動画文字起こしサービス</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700&display=swap" rel="stylesheet">
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <style>
        :root {
            --primary: #2A54A2;
            --primary-light: #3A64B2;
            --secondary: #5F8DE8;
            --gray-light: #f5f7fa;
            --gray: #e2e8f0;
            --text: #333333;
            --text-light: #666666;
            --success: #10b981;
            --warning: #f59e0b;
            --error: #ef4444;
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: 'Noto Sans JP', sans-serif;
            line-height: 1.6;
            color: var(--text);
            background-color: var(--gray-light);
            padding: 0;
            margin: 0;
        }
        
        .container {
            max-width: 1100px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        .header {
            text-align: center;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--gray);
        }
        
        .title {
            font-size: 2rem;
            font-weight: 700;
            color: var(--primary);
            margin-bottom: 0.5rem;
        }
        
        .subtitle {
            font-size: 1rem;
            color: var(--text-light);
            margin-bottom: 1rem;
        }
        
        .card {
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            padding: 2rem;
            margin-bottom: 2rem;
        }
        
        .panel {
            display: flex;
            flex-wrap: wrap;
            gap: 2rem;
        }
        
        .panel-left {
            flex: 1;
            min-width: 300px;
        }
        
        .panel-right {
            flex: 2;
            min-width: 300px;
        }
        
        .form-group {
            margin-bottom: 1.5rem;
        }
        
        label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 500;
            color: var(--text);
        }
        
        input[type="file"] {
            width: 100%;
            padding: 0.5rem;
            border: 1px solid var(--gray);
            border-radius: 6px;
            background-color: white;
        }
        
        select {
            width: 100%;
            padding: 0.75rem;
            border: 1px solid var(--gray);
            border-radius: 6px;
            background-color: white;
            font-family: 'Noto Sans JP', sans-serif;
            font-size: 1rem;
        }
        
        button {
            display: inline-block;
            background-color: var(--primary);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            font-size: 1rem;
            font-weight: 500;
            border-radius: 6px;
            cursor: pointer;
            transition: background-color 0.2s;
            width: 100%;
        }
        
        button:hover {
            background-color: var(--primary-light);
        }
        
        .result-area {
            background-color: white;
            border: 1px solid var(--gray);
            border-radius: 6px;
            padding: 1rem;
            min-height: 200px;
            margin-bottom: 1rem;
        }
        
        .status {
            margin-bottom: 1rem;
            padding: 1rem;
            border-radius: 6px;
            background-color: var(--gray-light);
        }
        
        .progress-container {
            width: 100%;
            background-color: var(--gray);
            border-radius: 8px;
            margin: 1rem 0;
            overflow: hidden;
        }
        
        .progress-bar {
            height: 12px;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            border-radius: 8px;
            transition: width 0.5s;
            width: 0%;
        }
        
        .result-link {
            display: inline-block;
            background-color: var(--success);
            color: white;
            text-decoration: none;
            padding: 0.75rem 1.5rem;
            font-weight: 500;
            border-radius: 6px;
            margin-top: 1rem;
        }
        
        .accordion {
            border: 1px solid var(--gray);
            border-radius: 6px;
            margin-bottom: 1rem;
            overflow: hidden;
        }
        
        .accordion-header {
            padding: 1rem;
            background-color: var(--gray-light);
            font-weight: 500;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .accordion-content {
            padding: 1rem;
            display: none;
        }
        
        .accordion-content.active {
            display: block;
        }
        
        .accordion-header::after {
            content: '+';
            font-size: 1.2rem;
        }
        
        .accordion-header.active::after {
            content: '-';
        }
        
        .alert {
            padding: 1rem;
            border-radius: 6px;
            margin-bottom: 1rem;
        }
        
        .alert-error {
            background-color: #fee2e2;
            color: var(--error);
        }
        
        .alert-success {
            background-color: #d1fae5;
            color: var(--success);
        }
        
        .alert-warning {
            background-color: #fef3c7;
            color: var(--warning);
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 1rem;
            }
            
            .panel {
                flex-direction: column;
                gap: 1rem;
            }
            
            .panel-left, .panel-right {
                width: 100%;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 class="title">VoiceNote</h1>
            <p class="subtitle">音声・動画ファイルからテキストを作成する文字起こしサービス</p>
        </div>
        
        <div class="card">
            <div class="panel">
                <div class="panel-left">
                    <form id="transcription-form" enctype="multipart/form-data">
                        <div class="form-group">
                            <label for="file">音声または動画ファイルをアップロード</label>
                            <input type="file" name="file" id="file" accept="audio/*,video/*" required>
                        </div>
                        
                        <div class="form-group">
                            <label for="language">言語選択</label>
                            <select name="language" id="language">
                                <option value="ja">日本語</option>
                                <option value="auto">自動検出</option>
                                <option value="en">英語</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label for="engine">エンジン選択</label>
                            <select name="engine" id="engine">
                                <option value="faster-whisper">Faster-Whisper（推奨）</option>
                                <option value="whisper">Whisper</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label for="model">モデル選択</label>
                            <select name="model" id="model">
                                <option value="large-v3">Large-v3（高精度）</option>
                                <option value="turbo">Turbo（高速）</option>
                                <option value="medium">Medium（バランス型）</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <button type="submit">文字起こし開始</button>
                        </div>
                    </form>
                    
                    <div class="accordion">
                        <div class="accordion-header">使い方</div>
                        <div class="accordion-content">
                            <ol>
                                <li>音声または動画ファイルをアップロードします</li>
                                <li>言語を選択します（自動検出、日本語、英語）</li>
                                <li>使用するエンジンとモデルを選択します</li>
                                <li>「文字起こし開始」ボタンをクリックします</li>
                                <li>処理が完了するまでお待ちください</li>
                                <li>完了したらダウンロードリンクが表示されます</li>
                            </ol>
                            <p>注意：ファイルサイズや長さによって処理時間が異なります。大きなファイルは時間がかかる場合があります。</p>
                        </div>
                    </div>
                </div>
                
                <div class="panel-right">
                    <div id="alerts"></div>
                    
                    <h3>処理状態</h3>
                    <div id="status" class="status">ファイルをアップロードして文字起こしを開始してください</div>
                    
                    <div class="progress-container">
                        <div id="progress-bar" class="progress-bar"></div>
                    </div>
                    <div id="progress-text"></div>
                    
                    <h3>結果</h3>
                    <div id="result" class="result-area">
                        <p>文字起こし結果がここに表示されます</p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
    $(document).ready(function() {
        // アコーディオン機能
        $('.accordion-header').click(function() {
            $(this).toggleClass('active');
            $(this).next('.accordion-content').toggleClass('active');
        });
        
        // フォーム送信
        $('#transcription-form').submit(function(event) {
            event.preventDefault();
            
            // 送信前の検証
            const fileInput = document.getElementById('file');
            if (!fileInput.files[0]) {
                showAlert('ファイルを選択してください', 'error');
                return;
            }
            
            // アラート表示クリア
            $('#alerts').empty();
            
            // フォームデータ取得
            const formData = new FormData(this);
            
            // ファイルサイズ表示
            const fileSize = (fileInput.files[0].size / (1024 * 1024)).toFixed(2);
            
            // 処理状態表示
            $('#status').text(`ファイルをアップロード中...(${fileSize}MB)`);
            $('#progress-bar').width('10%');
            $('#progress-text').text('10%');
            
            // Ajax送信
            $.ajax({
                url: '/transcribe',
                type: 'POST',
                data: formData,
                processData: false,
                contentType: false,
                success: function(data) {
                    $('#status').text('文字起こし処理を開始しました');
                    showAlert('処理を開始しました。完了までお待ちください。', 'success');
                    checkStatus(data.task_id);
                },
                error: function(xhr) {
                    let errorMessage = '処理開始中にエラーが発生しました';
                    if (xhr.responseJSON && xhr.responseJSON.error) {
                        errorMessage += ': ' + xhr.responseJSON.error;
                    }
                    $('#status').text('エラーが発生しました');
                    $('#progress-bar').width('0%');
                    $('#progress-text').text('');
                    showAlert(errorMessage, 'error');
                }
            });
        });
        
        // 処理状態確認
        function checkStatus(taskId) {
            $.get('/status/' + taskId, function(data) {
                // 状態表示更新
                $('#status').text('状態: ' + getStatusText(data.status));
                
                // 進捗表示更新
                if (data.state === 'PROGRESS') {
                    $('#progress-bar').width(data.progress + '%');
                    $('#progress-text').text(data.progress + '%');
                }
                
                // 処理完了
                if (data.state === 'SUCCESS') {
                    $('#progress-bar').width('100%');
                    $('#progress-text').text('100%');
                    $('#result').html('<p>文字起こしが完了しました！</p><a href="/result/' + 
                        taskId + '" target="_blank" class="result-link">テキストファイルをダウンロード</a>');
                    showAlert('文字起こしが完了しました！ダウンロードリンクをクリックしてファイルを保存してください。', 'success');
                } 
                // 処理失敗
                else if (data.state === 'FAILURE') {
                    $('#status').text('文字起こしに失敗しました');
                    $('#progress-bar').width('0%');
                    $('#progress-text').text('');
                    showAlert('文字起こし処理に失敗しました: ' + data.error, 'error');
                } 
                // 処理継続中
                else {
                    setTimeout(function() {
                        checkStatus(taskId);
                    }, 3000);  // 3秒ごとに状態確認
                }
            }).fail(function() {
                $('#status').text('状態確認中にエラーが発生しました');
                showAlert('処理状態の取得に失敗しました。更新してやり直してください。', 'error');
            });
        }
        
        // 状態テキスト変換
        function getStatusText(status) {
            switch (status) {
                case 'Task is waiting for execution.':
                    return '処理待機中...';
                case 'Task is in progress.':
                    return '文字起こし処理中...';
                case 'Task completed successfully.':
                    return '文字起こし完了！';
                case 'Task failed.':
                    return '処理に失敗しました';
                default:
                    return status;
            }
        }
        
        // アラート表示
        function showAlert(message, type) {
            const alertClass = type === 'error' ? 'alert-error' : 
                              type === 'warning' ? 'alert-warning' : 'alert-success';
            
            const alertHtml = `<div class="alert ${alertClass}">${message}</div>`;
            $('#alerts').html(alertHtml);
            
            // 成功メッセージは5秒後に消える
            if (type === 'success') {
                setTimeout(function() {
                    $('#alerts .alert-success').fadeOut(500, function() {
                        $(this).remove();
                    });
                }, 5000);
            }
        }
    });
    </script>
</body>
</html>
"""

def log_system_resources():
    """システムリソース情報をログに記録する"""
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    gpu_info = GPUtil.getGPUs()[0] if GPUtil.getGPUs() else None
    
    logger.info(f"CPU使用率: {cpu_percent}%")
    logger.info(f"メモリ使用率: {memory.percent}% (使用中: {memory.used / 1024 / 1024:.2f} MB, 利用可能: {memory.available / 1024 / 1024:.2f} MB)")
    logger.info(f"ディスク使用率: {disk.percent}% (使用中: {disk.used / 1024 / 1024 / 1024:.2f} GB, 空き: {disk.free / 1024 / 1024 / 1024:.2f} GB)")
    
    if gpu_info:
        logger.info(f"GPU使用状況: {gpu_info.memoryUsed}MB / {gpu_info.memoryTotal}MB")

def load_model(engine, model_name):
    """モデルをロードする関数"""
    # キャッシュされたモデルがあればそれを使う
    if (engine, model_name) not in models:
        logger.info(f"モデルをロード中: {model_name}, エンジン: {engine}")
        try:
            if engine == "faster-whisper":
                # Faster-Whisperモデルのロード
                models[(engine, model_name)] = WhisperModel(model_name, device=device, compute_type=compute_type)
            elif engine == "whisper":
                # Whisper公式モデルのロード
                models[(engine, model_name)] = whisper.load_model(model_name, device=device)
            else:
                raise ValueError("不明なエンジンが指定されました")
            
            logger.info(f"{engine}エンジンで{model_name}モデルのロードに成功しました")
        except Exception as e:
            logger.error(f"{engine}エンジンで{model_name}モデルのロード中にエラーが発生しました: {str(e)}")
            raise
    
    return models[(engine, model_name)]

@celery.task(bind=True, soft_time_limit=7200, time_limit=7500)
def transcribe_audio(self, file_path, language, engine, model_name, original_filename):
    """音声/動画ファイルの文字起こしを行うCeleryタスク"""
    try:
        start_time = time.time()
        logger.info(f"文字起こし開始: ファイル={file_path}, 言語={language}, エンジン={engine}, モデル={model_name}")
        log_system_resources()
        
        # モデルをロード
        model = load_model(engine, model_name)
        logger.info("文字起こし処理を開始します")
        
        if engine == "faster-whisper":
            # Faster-Whisperの場合
            segments, info = model.transcribe(file_path, language=language if language != 'auto' else None)
            segments_list = list(segments)
            detected_language = info.language
        else:
            # whisper公式モデルの場合
            kwargs = {}
            if language != 'auto':
                kwargs['language'] = language
            result = model.transcribe(file_path, **kwargs)
            segments_list = result["segments"]
            detected_language = result["language"]
        
        logger.info("セグメント処理中")
        total_segments = len(segments_list)
        output_filename = f"{uuid.uuid4()}.txt"
        output_file = os.path.join(RESULT_FOLDER, output_filename)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # 検出された言語を記録（オプション）
            # f.write(f"検出された言語: {detected_language}\n\n")
            
            for i, segment in enumerate(segments_list):
                progress = min(100, int((i + 1) / total_segments * 100))
                if i % 10 == 0 or progress == 100:
                    self.update_state(state='PROGRESS', meta={'progress': progress})
                    logger.info(f"文字起こし進捗: {progress}%")
                
                # エンジンに応じたテキスト抽出
                text = segment.text if engine == "faster-whisper" else segment["text"]
                f.write(f"{text} ")
        
        logger.info(f"文字起こし完了: {total_segments}セグメント")
        
        # 入力ファイルの削除
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"入力ファイルを削除しました: {file_path}")
            else:
                logger.warning(f"削除対象の入力ファイルが見つかりません: {file_path}")
        except Exception as e:
            logger.error(f"入力ファイル{file_path}の削除に失敗しました: {str(e)}")
        
        total_time = time.time() - start_time
        logger.info(f"処理時間: {total_time:.2f}秒")
        log_system_resources()
        
        return output_file, original_filename
    
    except SoftTimeLimitExceeded:
        logger.error("文字起こし処理が制限時間を超過しました")
        raise
    except Exception as e:
        logger.error(f"文字起こし処理中にエラーが発生しました: {str(e)}", exc_info=True)
        
        # エラー発生時も入力ファイルを削除
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"エラー後に入力ファイルを削除しました: {file_path}")
        except Exception as del_e:
            logger.error(f"エラー後の入力ファイル{file_path}の削除に失敗しました: {str(del_e)}")
        
        raise

@app.route('/')
def home():
    """トップページを表示"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """文字起こしリクエストを処理"""
    if 'file' not in request.files:
        return jsonify({"error": "ファイルがアップロードされていません"}), 400
    
    file = request.files['file']
    language = request.form.get('language', 'auto')
    model_name = request.form.get('model', 'medium')
    engine = request.form.get('engine', 'faster-whisper')
    original_filename = file.filename
    
    # ファイル保存
    file_extension = os.path.splitext(original_filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(file_path)
    logger.info(f"ファイルを保存しました: {file_path}")
    
    # 文字起こしタスク開始
    task = transcribe_audio.delay(file_path, language, engine, model_name, original_filename)
    logger.info(f"文字起こしタスクを開始しました: {task.id}")
    
    return jsonify({"task_id": task.id}), 202

@app.route('/status/<task_id>')
def task_status(task_id):
    """タスクの状態を確認"""
    task = transcribe_audio.AsyncResult(task_id)
    logger.info(f"タスク状態確認: {task_id}, 状態: {task.state}")
    
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
    """文字起こし結果を取得"""
    task = transcribe_audio.AsyncResult(task_id)
    logger.info(f"結果取得: {task_id}, 状態: {task.state}")
    
    if task.state == 'SUCCESS':
        result_file, original_filename = task.result
        if os.path.exists(result_file):
            try:
                download_name = os.path.splitext(original_filename)[0] + '.txt'
                return_value = send_file(result_file, as_attachment=True, download_name=download_name)
                
                @return_value.call_on_close
                def delete_file_after_send():
                    try:
                        os.remove(result_file)
                        logger.info(f"結果ファイルを送信後に削除しました: {result_file}")
                    except Exception as e:
                        logger.error(f"結果ファイル{result_file}の削除に失敗しました: {str(e)}")
                
                return return_value
            except Exception as e:
                logger.error(f"ファイル{result_file}の送信中にエラーが発生しました: {str(e)}")
                return jsonify({"error": "ファイル送信中にエラーが発生しました"}), 500
        else:
            logger.warning(f"結果ファイルが見つかりません: {result_file}")
            return jsonify({"error": "結果ファイルが見つかりません"}), 404
    elif task.state == 'FAILURE':
        logger.error(f"タスクが失敗しました: {task_id}")
        return jsonify({"error": "タスクが失敗しました", "details": str(task.result)}), 500
    else:
        logger.info(f"結果がまだ準備できていません: {task_id}")
        return jsonify({"error": "結果がまだ準備できていません"}), 404

@app.errorhandler(RequestEntityTooLarge)
def handle_request_entity_too_large(error):
    """ファイルサイズ超過エラーを処理"""
    return jsonify({"error": "ファイルサイズが大きすぎます"}), 413

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)