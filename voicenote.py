import multiprocessing
from flask import Flask, request, send_file, render_template, jsonify
# 追加: 各種モデルのインポート
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
import json
from pathlib import Path
import transformers
from transformers import pipeline

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
CONFIG_FOLDER = '/tmp/voicenote_config'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(CONFIG_FOLDER, exist_ok=True)

# HuggingFace APIトークン管理
TOKEN_FILE = os.path.join(CONFIG_FOLDER, 'huggingface_token.json')

def save_huggingface_token(token):
    """HuggingFace APIトークンを保存する"""
    try:
        with open(TOKEN_FILE, 'w', encoding='utf-8') as f:
            json.dump({"token": token}, f)
        logger.info("HuggingFace APIトークンを保存しました")
        return True
    except Exception as e:
        logger.error(f"HuggingFace APIトークンの保存に失敗しました: {str(e)}")
        return False

def load_huggingface_token():
    """保存されたHuggingFace APIトークンを読み込む"""
    if not os.path.exists(TOKEN_FILE):
        # 環境変数からトークンを取得
        token = os.environ.get('HUGGINGFACE_TOKEN')
        if token:
            save_huggingface_token(token)
            return token
        return None
    
    try:
        with open(TOKEN_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("token")
    except Exception as e:
        logger.error(f"HuggingFace APIトークンの読み込みに失敗しました: {str(e)}")
        return None

def check_huggingface_token_validity(token):
    """HuggingFace APIトークンの有効性を確認する"""
    if not token:
        return False
    
    try:
        # Transformersライブラリを使ってトークンが有効かチェック
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.whoami()
        return True
    except Exception as e:
        logger.error(f"HuggingFace APIトークンの検証に失敗しました: {str(e)}")
        return False

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

def load_model(engine, model_name, use_diarization=False, use_punctuation=False):
    """モデルをロードする関数"""
    # キャッシュされたモデルがあればそれを使う
    # 話者分離と句読点のオプションも含めてキャッシュする
    cache_key = (engine, model_name, use_diarization, use_punctuation)
    if cache_key not in models:
        logger.info(f"モデルをロード中: {model_name}, エンジン: {engine}")
        logger.info(f"話者分離: {use_diarization}, 句読点: {use_punctuation}")
        
        try:
            if engine == "faster-whisper":
                # Faster-Whisperモデルのロード
                models[cache_key] = WhisperModel(model_name, device=device, compute_type=compute_type)
            elif engine == "whisper":
                # Whisper公式モデルのロード
                models[cache_key] = whisper.load_model(model_name, device=device)
            elif engine == "kotoba-whisper-transformers":
                # Kotoba-Whisper Transformers版モデルのロード
                # HuggingFaceトークンの確認
                token = load_huggingface_token()
                if not token or not check_huggingface_token_validity(token):
                    raise ValueError("有効なHuggingFace APIトークンが必要です。環境変数HUGGINGFACE_TOKENを設定してください。")
                
                # Kotoba-Whisper v2.2のロード
                if model_name == "kotoba-whisper-v2.2":
                    logger.info("Kotoba-Whisper v2.2モデルをロード中")
                    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
                    
                    # Diarization（話者分離）
                    diarization_pipeline = None
                    if use_diarization:
                        logger.info("話者分離パイプラインをロード中")
                        try:
                            # まず標準の方法を試す
                            from diarizers import Diarizer
                            diarization_pipeline = Diarizer(
                                model_name="pyannote/speaker-diarization-3.1",
                                token=token,
                                device=device
                            )
                            logger.info("話者分離パイプラインのロードに成功しました")
                        except ImportError as e:
                            # ImportErrorが発生した場合はエラーをログに記録し、話者分離をスキップ
                            logger.error(f"話者分離モジュールのインポートに失敗しました: {str(e)}")
                            logger.warning("話者分離機能は無効化されます")
                            use_diarization = False
                    
                    # 句読点
                    punctuation = None
                    if use_punctuation:
                        logger.info("句読点パイプラインをロード中")
                        from punctuators import PunctuationModel
                        punctuation = PunctuationModel(
                            model="kotoba-tech/punctuation-ja", 
                            token=token,
                            device=device
                        )
                    
                    # メインモデルのロード
                    processor = AutoProcessor.from_pretrained(
                        "kotoba-tech/kotoba-whisper-v2.2", 
                        token=token
                    )
                    model = AutoModelForSpeechSeq2Seq.from_pretrained(
                        "kotoba-tech/kotoba-whisper-v2.2",
                        token=token,
                        device_map="auto"
                    )
                    pipe = pipeline(
                        "automatic-speech-recognition",
                        model=model,
                        tokenizer=processor.tokenizer,
                        feature_extractor=processor.feature_extractor,
                        chunk_length_s=30,
                        device=device
                    )
                    
                    # モデルとパイプラインをセットで保存
                    models[cache_key] = {
                        "main_pipeline": pipe,
                        "diarization": diarization_pipeline,
                        "punctuation": punctuation
                    }
                else:
                    # その他のKotoba-Whisperモデル
                    models[cache_key] = pipeline(
                        "automatic-speech-recognition",
                        model=model_name,
                        chunk_length_s=30,
                        device=device,
                        token=token
                    )
            else:
                raise ValueError("不明なエンジンが指定されました")
            
            logger.info(f"{engine}エンジンで{model_name}モデルのロードに成功しました")
        except Exception as e:
            logger.error(f"{engine}エンジンで{model_name}モデルのロード中にエラーが発生しました: {str(e)}")
            raise
    
    return models[cache_key]

@celery.task(bind=True, soft_time_limit=7200, time_limit=7500)
def transcribe_audio(self, file_path, language, engine, model_name, original_filename, use_diarization=False, use_punctuation=False):
    """音声/動画ファイルの文字起こしを行うCeleryタスク"""
    try:
        start_time = time.time()
        logger.info(f"文字起こし開始: ファイル={file_path}, 言語={language}, エンジン={engine}, モデル={model_name}")
        logger.info(f"オプション: 話者分離={use_diarization}, 句読点={use_punctuation}")
        log_system_resources()
        
        # モデルをロード
        model = load_model(engine, model_name, use_diarization, use_punctuation)
        logger.info("文字起こし処理を開始します")
        
        if engine == "faster-whisper":
            # Faster-Whisperの場合
            logger.info("Faster-Whisperモデルで文字起こしを実行中")
            # 中間進捗報告のための処理
            self.update_state(state='PROGRESS', meta={'progress': 25})
            logger.info(f"文字起こし進捗: 25%")
            
            # 全体の文字起こしを一度に取得
            segments, info = model.transcribe(file_path, language=language if language != 'auto' else None)
            
            # 中間進捗報告
            self.update_state(state='PROGRESS', meta={'progress': 75})
            logger.info(f"文字起こし進捗: 75%")
            
            # 全テキストの結合（セグメント化せずに連結）
            full_text = " ".join([segment.text for segment in segments])
            detected_language = info.language
            
        elif engine == "whisper":
            # whisper公式モデルの場合
            logger.info("Whisper公式モデルで文字起こしを実行中")
            # 中間進捗報告のための処理
            self.update_state(state='PROGRESS', meta={'progress': 25})
            logger.info(f"文字起こし進捗: 25%")
            
            kwargs = {}
            if language != 'auto':
                kwargs['language'] = language
            
            # 全体の文字起こし
            result = model.transcribe(file_path, **kwargs)
            
            # 中間進捗報告
            self.update_state(state='PROGRESS', meta={'progress': 75})
            logger.info(f"文字起こし進捗: 75%")
            
            # 全テキストの取得（セグメント化せずに使用）
            full_text = result["text"]
            detected_language = result["language"]
            
        elif engine == "kotoba-whisper-transformers":
            # Kotoba-Whisper Transformers版モデルの場合
            logger.info("Kotoba-Whisper Transformersモデルで文字起こしを実行中")
            
            if model_name == "kotoba-whisper-v2.2":
                # Kotoba-Whisper v2.2専用の処理
                logger.info("Kotoba-Whisper v2.2専用の処理を実行中")
                # 中間進捗報告のための処理
                self.update_state(state='PROGRESS', meta={'progress': 10})
                
                # 音声の読み込み
                logger.info("音声ファイルを読み込み中")
                import librosa
                audio, sr = librosa.load(file_path, sr=16000)
                
                # 話者分離が有効な場合
                segments = None
                if use_diarization and model["diarization"]:
                    try:
                        logger.info("話者分離を実行中")
                        self.update_state(state='PROGRESS', meta={'progress': 25})
                        
                        # 話者分離を実行
                        # 注意: 入力形式はライブラリのバージョンによって異なる可能性があります
                        try:
                            # 新しいバージョンの場合
                            segments = model["diarization"]({"waveform": audio, "sample_rate": 16000})
                        except Exception as e:
                            logger.warning(f"新形式での話者分離に失敗しました。古い形式を試します: {str(e)}")
                            try:
                                # 古いバージョンの場合
                                segments = model["diarization"](file_path)
                            except Exception as e2:
                                logger.error(f"話者分離に失敗しました: {str(e2)}")
                                raise ValueError("話者分離に失敗しました。他のオプションを試してください。")
                        
                        logger.info(f"話者分離結果: {len(segments)}個のセグメントを検出")
                    except Exception as e:
                        # 話者分離に失敗した場合は、話者分離なしで処理を続行
                        logger.error(f"話者分離処理中にエラーが発生しました: {str(e)}")
                        logger.warning("話者分離なしで処理を続行します")
                        use_diarization = False
                        segments = None

                # 話者分離が有効かつ成功した場合は、各セグメントごとに音声認識を実行
                if use_diarization and segments:
                    # 音声認識実行
                    self.update_state(state='PROGRESS', meta={'progress': 50})
                    logger.info("音声認識を実行中（話者分離あり）")
                    
                    # 各セグメントごとに認識
                    texts = []
                    for i, segment in enumerate(segments):
                        # セグメントのオーディオを抽出
                        try:
                            # 新しいバージョンの場合
                            start_sample = int(segment["start"] * sr)
                            end_sample = int(segment["end"] * sr)
                            speaker_id = segment['speaker']
                        except (KeyError, TypeError):
                            # 古いバージョンの場合
                            try:
                                # 別の形式かもしれない
                                start_sample = int(segment.start * sr)
                                end_sample = int(segment.end * sr)
                                speaker_id = segment.speaker
                            except AttributeError:
                                # さらに別の形式かもしれない
                                start_sample = int(float(segment[0]) * sr)
                                end_sample = int(float(segment[1]) * sr)
                                speaker_id = segment[2] if len(segment) > 2 else "不明"
                        
                        segment_audio = audio[start_sample:end_sample]
                        
                        # 音声認識
                        recognition_result = model["main_pipeline"](
                            {"array": segment_audio, "sampling_rate": sr}, 
                            generate_kwargs={"language": language if language != 'auto' else "ja"}
                        )
                        
                        # 結果の保存
                        try:
                            speaker = f"【話者{speaker_id}】 "
                        except NameError:
                            # speaker_idが取得できない場合
                            try:
                                speaker = f"【話者{segment['speaker']}】 "
                            except (KeyError, TypeError):
                                try:
                                    speaker = f"【話者{segment.speaker}】 "
                                except AttributeError:
                                    speaker = f"【話者{i+1}】 "
                                    
                        text = recognition_result["text"]
                        
                        # 句読点があればテキストを整形
                        if use_punctuation and model["punctuation"]:
                            text = model["punctuation"](text)
                        
                        texts.append(f"{speaker}{text}")
                        
                        # 進捗更新
                        progress = 50 + int((i + 1) / len(segments) * 40)
                        self.update_state(state='PROGRESS', meta={'progress': progress})
                    
                    # 全テキストの結合
                    full_text = "\n".join(texts)
                    detected_language = language if language != 'auto' else "ja"
                
                # 話者分離が無効か、話者分離が失敗した場合
                else:
                    # 話者分離なしの場合
                    self.update_state(state='PROGRESS', meta={'progress': 25})
                    logger.info("音声認識を実行中（話者分離なし）")
                    
                    # 音声認識実行
                    recognition_result = model["main_pipeline"](
                        {"array": audio, "sampling_rate": sr}, 
                        generate_kwargs={"language": language if language != 'auto' else "ja"}
                    )
                    
                    self.update_state(state='PROGRESS', meta={'progress': 75})
                    text = recognition_result["text"]
                    
                    # 句読点処理
                    if use_punctuation and model["punctuation"]:
                        logger.info("句読点を追加中")
                        text = model["punctuation"](text)
                    
                    full_text = text
                    detected_language = language if language != 'auto' else "ja"
            
            else:
                # その他のKotoba-Whisperモデル
                logger.info(f"通常のTransformers ASRパイプラインを使用中: {model_name}")
                self.update_state(state='PROGRESS', meta={'progress': 25})
                
                # 音声の読み込み
                import librosa
                audio, sr = librosa.load(file_path, sr=16000)
                
                # 音声認識実行
                recognition_result = model(
                    {"array": audio, "sampling_rate": sr}, 
                    generate_kwargs={"language": language if language != 'auto' else "ja"}
                )
                
                self.update_state(state='PROGRESS', meta={'progress': 75})
                full_text = recognition_result["text"]
                detected_language = language if language != 'auto' else "ja"
                
        else:
            raise ValueError(f"不明なエンジン: {engine}")
        
        logger.info("文字起こし完了、テキストファイル作成中")
        output_filename = f"{uuid.uuid4()}.txt"
        output_file = os.path.join(RESULT_FOLDER, output_filename)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # 完全なテキストを一度に書き込み
            f.write(full_text)
            
        # 最終進捗報告
        self.update_state(state='PROGRESS', meta={'progress': 100})
        logger.info(f"文字起こし進捗: 100%")
        
        logger.info("文字起こし完了、テキストファイルの作成が完了しました")
        
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
    
    # 追加設定を取得
    use_diarization = request.form.get('use_diarization', 'false').lower() == 'true'
    use_punctuation = request.form.get('use_punctuation', 'false').lower() == 'true'
    
    # Kotoba-Whisperモデルが選択されていて、トークンが必要な場合にチェック
    if engine == 'kotoba-whisper-transformers':
        token = load_huggingface_token()
        if not token:
            # 環境変数からトークンを取得
            token = os.environ.get('HUGGINGFACE_TOKEN')
            if token:
                save_huggingface_token(token)
            else:
                return jsonify({"error": "Kotoba-Whisperモデルの使用にはHuggingFace APIトークンが必要です。環境変数HUGGINGFACE_TOKENを設定してください。"}), 400
        
        # トークンの有効性確認
        if not check_huggingface_token_validity(token):
            return jsonify({"error": "設定されたHuggingFace APIトークンが無効です。有効なトークンを設定してください。"}), 400
    
    # ファイル保存
    file_extension = os.path.splitext(original_filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(file_path)
    logger.info(f"ファイルを保存しました: {file_path}")
    
    # 文字起こしタスク開始
    task = transcribe_audio.delay(file_path, language, engine, model_name, original_filename, use_diarization, use_punctuation)
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