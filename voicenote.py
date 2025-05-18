import multiprocessing
from flask import Flask, request, send_file, render_template, jsonify
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
app = Flask(__name__, template_folder='templates', static_folder='static')
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

def log_system_resources():
    """システムリソース情報をログに記録する"""
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    try:
        gpus = GPUtil.getGPUs()
        gpu_info = gpus[0] if gpus else None
        if not gpus:
            logger.warning("GPU情報が取得できません")
    except Exception as e:
        gpu_info = None
        logger.warning(f"GPU情報の取得中に例外が発生しました: {str(e)}")
    
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
    return render_template('index.html')

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
        progress = task.info.get('progress', 0) if task.info else 0
        response = {
            'state': task.state,
            'status': 'Task is in progress.',
            'progress': progress
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