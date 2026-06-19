import multiprocessing
from flask import Flask, request, send_file, render_template, jsonify
import os
import uuid
import logging
import time
from celery import Celery
from werkzeug.exceptions import RequestEntityTooLarge
from celery.exceptions import SoftTimeLimitExceeded
import psutil
import GPUtil

# STTプラグイン
import config
from engines.factory import EngineFactory

# マルチプロセッシングの開始方法を'spawn'に設定（CUDA利用のため）
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


@celery.task(bind=True, soft_time_limit=7200, time_limit=7500)
def transcribe_audio(self, file_path, language, model_name, original_filename):
    """音声/動画ファイルの文字起こしを行うCeleryタスク。

    使用エンジンはモデル定義(config/models.yaml)から決定する。
    """
    try:
        start_time = time.time()
        logger.info(f"文字起こし開始: ファイル={file_path}, 言語={language}, モデル={model_name}")
        log_system_resources()

        # モデル定義からエンジン/オプションを決定
        model_cfg = config.get_model(model_name)
        engine = EngineFactory.create(model_cfg.engine, model_cfg.id, model_cfg.options)
        logger.info(f"エンジン={engine.name} で文字起こし処理を開始します")

        # 進捗をCeleryへ橋渡しするコールバック
        def progress_cb(percent):
            self.update_state(state='PROGRESS', meta={'progress': percent})
            logger.info(f"文字起こし進捗: {percent}%")

        result = engine.transcribe(file_path, language=language, progress_cb=progress_cb)
        logger.info(f"文字起こし完了: 検出言語={result.language}, セグメント数={len(result.segments)}")

        # 結果の書き出し（話者情報があれば話者ラベル付き）
        output_filename = f"{uuid.uuid4()}.txt"
        output_file = os.path.join(RESULT_FOLDER, output_filename)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result.to_text())

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


@app.route('/models')
def models():
    """利用可能なモデルをカテゴリ別に返す（UIが動的にプルダウンを生成）"""
    return jsonify(config.grouped_for_ui())


@app.route('/transcribe', methods=['POST'])
def transcribe():
    """文字起こしリクエストを処理"""
    if 'file' not in request.files:
        return jsonify({"error": "ファイルがアップロードされていません"}), 400

    file = request.files['file']
    language = request.form.get('language', 'auto')
    model_name = request.form.get('model') or config.get_default()
    original_filename = file.filename

    # モデルIDを検証（エンジンはサーバ側でモデル定義から決定する）
    try:
        config.get_model(model_name)
    except KeyError:
        return jsonify({"error": f"不明なモデルが指定されました: {model_name}"}), 400

    # ファイル保存
    file_extension = os.path.splitext(original_filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(file_path)
    logger.info(f"ファイルを保存しました: {file_path}")

    # 文字起こしタスク開始
    task = transcribe_audio.delay(file_path, language, model_name, original_filename)
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
