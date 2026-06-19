# VoiceNote

VoiceNoteは、複数のSTT（音声認識）エンジンをプラグイン方式で切り替えられる、音声・動画文字起こしサービスのFlaskベースWebアプリケーションです。効率的な処理のためにGPUアクセラレーションを活用し、非同期タスク管理にCeleryを使用しています。

## 特徴

- **STTプラグイン構造**：Faster-Whisper / Kotoba-Whisper / ReazonSpeech / Qwen3-ASR を共通インターフェースで切り替え
- モデル定義は `config/models.yaml` で外部管理（コード変更なしでモデル追加・変更が可能）
- UIはカテゴリ別（Whisper / Kotoba / ReazonSpeech / Qwen3-ASR）にモデルを表示
- 複数言語のサポート（自動検出を含む。エンジンにより対応言語は異なる）
- リアルタイムの進捗更新を伴う非同期処理
- より高速な文字起こしのためのGPUアクセラレーション
- 簡単なファイルアップロードと結果取得のためのWebインターフェース

## 対応エンジン / モデル

| カテゴリ | モデルID | エンジン | 対応言語 | VRAM目安 |
| --- | --- | --- | --- | --- |
| Whisper | `large-v3` | faster-whisper | 多言語 | 4〜6GB |
| Kotoba | `kotoba-tech/kotoba-whisper-v2.2` | kotoba（話者分離） | 日本語 | 5〜8GB |
| Kotoba | `kotoba-tech/kotoba-whisper-v2.0` | kotoba（シンプル） | 日本語 | 5〜8GB |
| ReazonSpeech | `reazon-research/reazonspeech-k2-v2` | reazon | 日本語 | 2〜4GB |
| Qwen3-ASR | `Qwen/Qwen3-ASR-1.7B` | qwen3 | 多言語 | 12〜20GB |

**用途の目安**：既定=Whisper Large-v3（多言語汎用） / 日本語・軽量=ReazonSpeech k2-v2 / 高精度=Kotoba v2.2（話者分離・議事録向け） / 最高精度=Qwen3-ASR 1.7B（多言語・重い）。

> **Kotoba v2.2 の話者分離について**：pyannote の gated モデル（`pyannote/segmentation-3.0`, `pyannote/speaker-diarization-3.1`）の利用規約に同意し、HuggingFaceトークンを `HF_TOKEN`（または `SINGULARITYENV_HF_TOKEN`）で渡す必要があります。トークン不要で使いたい場合は Kotoba v2.0（シンプル）を選択してください。

> **GPUについて**：V100(32GB)×4 を含む一般的なCUDA環境で全モデルが動作します。V100(Volta)は bfloat16 / FlashAttention-2 非対応のため、Qwen3-ASR は自動的に fp16 + sdpa で実行されます。

> **長尺ファイルについて**：ReazonSpeech k2 と Qwen3-ASR は入力全体を1パス処理するため、長尺はサーバ側で自動的に ffmpeg で分割（既定30秒、`VN_CHUNK_SECONDS` で変更可）して逐次処理します。Whisper / Kotoba はエンジン内部で長尺を扱うため分割しません。

## 要件

- CUDAサポート付きのNVIDIA GPU（推奨ドライバ: CUDA 12.4 利用時は R550+）
- [Singularity](https://sylabs.io/singularity/)コンテナランタイム

## インストール

1. このリポジトリをクローンします：
   ```
   git clone https://github.com/Ecomott-DataAnalytics/VoiceNote.git
   cd VoiceNote
   ```

2. Singularityコンテナをビルドします：
   ```
   singularity build --fakeroot voicenote.sif voicenote.def
   ```

## 使用方法

1. VoiceNoteサービスを起動します：
   ```
   singularity run --nv --nvccli voicenote.sif
   ```

2. Webブラウザを開き、`http://localhost:5000`にアクセスします

3. 音声または動画ファイルをアップロードし、希望の言語とモデルを選択して「文字起こし開始」をクリックします

4. 文字起こしが完了するまで待ち、結果をダウンロードします

Kotoba v2.2（話者分離）を使う場合は、起動前にHuggingFaceトークンを渡します：

```
SINGULARITYENV_HF_TOKEN=hf_xxx singularity run --nv --nvccli voicenote.sif
```

## テストの実行

```bash
python -m pytest tests/ -v
# または
python -m unittest discover -s tests
```

## プロジェクト構成

```
VoiceNote/
├── voicenote.py              # メインFlaskアプリ（Flask + Celery + エンドポイント）
├── voicenote.def             # Singularityコンテナ定義（CUDA 12系）
├── requirements.txt          # Python依存（ローカル/CI参照用）
├── engines/                  # STTエンジン・プラグイン
│   ├── base.py               # 抽象基底 STTEngine と正規化結果 TranscriptionResult
│   ├── factory.py            # EngineFactory（engine値で生成・キャッシュ）
│   ├── faster_whisper_engine.py
│   ├── kotoba_engine.py      # v2.2(話者分離)/v2.0(シンプル) 両対応
│   ├── reazon_engine.py
│   └── qwen3_asr_engine.py   # fp16自動判定（V100対応）
├── config/
│   ├── __init__.py           # YAMLローダ（load_models / get_model / grouped_for_ui）
│   └── models.yaml           # モデル定義（id/engine/category/label/vram/options）
├── static/
│   ├── css/style.css
│   └── js/main.js            # /models から動的にプルダウン生成
├── templates/
│   └── index.html
└── tests/                    # ユニットテスト
```

## 新しいモデル / エンジンの追加

1. `config/models.yaml` にエントリを追加（`id` / `engine` / `category` / `label` / `vram`、必要なら `options`）。
2. 新エンジンが必要なら `engines/` にクラス（`STTEngine` 継承）を追加し、`engines/factory.py` の `_REGISTRY` に登録。
3. UI（プルダウン）は `/models` から自動生成されるため変更不要。

## 設定

- モデル定義：`config/models.yaml`
- アップロード/結果フォルダ・最大ファイルサイズ・CeleryブローカーURL：`voicenote.py`
- Qwen3-ASRバックエンド：環境変数 `QWEN_ASR_BACKEND`（`transformers`（既定）/ `vllm`）
- HuggingFaceトークン：環境変数 `HF_TOKEN`（Kotoba v2.2用）
- 使用GPU：環境変数 `CUDA_VISIBLE_DEVICES`（マルチGPUのワーカー固定例は `voicenote.def` のコメント参照）

## ライセンス

LICENSEファイルを参照してください

## 謝辞

VoiceNoteプロジェクトは、以下のオープンソースプロジェクトの上に構築されています。これらのプロジェクトの開発者とコントリビューターに心から感謝いたします：

- [Faster Whisper](https://github.com/guillaumekln/faster-whisper): 高速で効率的な音声認識モデル
- [Kotoba-Whisper](https://huggingface.co/kotoba-tech): 日本語に強いWhisper派生モデル（v2.2は話者分離対応）
- [ReazonSpeech](https://github.com/reazon-research/ReazonSpeech): 日本語音声認識モデル（k2-v2）
- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR): Alibaba Cloud Qwenチームによる多言語ASRモデル（Apache-2.0）
- [Flask](https://flask.palletsprojects.com/): 軽量で拡張可能なPythonウェブフレームワーク
- [Gunicorn](https://gunicorn.org/): A Python WSGI HTTP サーバ
- [Celery](https://docs.celeryproject.org/): 分散タスクキューシステム
- [Redis](https://redis.io/): 高性能なインメモリデータストア
- [PyTorch](https://pytorch.org/): オープンソースの機械学習フレームワーク
- [NVIDIA CUDA](https://developer.nvidia.com/cuda-zone): GPUコンピューティングプラットフォーム

また、[OpenAI Whisper](https://github.com/openai/whisper)チームにも感謝いたします。彼らの革新的な研究と公開モデルがなければ、このプロジェクトは実現しませんでした。

これらのプロジェクトとその貢献者たちの努力と献身なくして、VoiceNoteは存在しませんでした。オープンソースコミュニティに深い感謝の意を表します。

## 備考

このプロジェクトは、エコモットテックブログの記事「[社内文字起こしサービス（仮）をサクっと立てた話](https://www.ecomottblog.com/?p=13901)」で公開されていた内容を改名し、拡張したものです。