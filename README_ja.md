# VoiceNote

VoiceNoteは、Faster Whisperモデルを使用して音声および動画の文字起こしサービスを提供するFlaskベースのWebアプリケーションです。効率的な処理のためにGPUアクセラレーションを活用し、非同期タスク管理にCeleryを使用しています。

## 特徴

- Faster Whisperを使用した音声および動画ファイルの文字起こし
- 複数言語のサポート（自動検出を含む）
- モデルサイズの選択（Large-v3およびMedium）
- リアルタイムの進捗更新を伴う非同期処理
- より高速な文字起こしのためのGPUアクセラレーション
- 簡単なファイルアップロードと結果取得のためのWebインターフェース

## 要件

- CUDAサポート付きのNVIDIA GPU
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

## プロジェクト構成

アプリケーションは標準的なFlaskプロジェクト構造に従っています：

```
VoiceNote/
├── voicenote.py         # メインFlaskアプリケーション
├── voicenote.def        # Singularityコンテナ定義
├── static/              # 静的ファイル
│   ├── css/
│   │   └── style.css    # CSSスタイル
│   └── js/
│       └── main.js      # UI対話用JavaScript
├── templates/
│   └── index.html       # Webインターフェース用HTMLテンプレート
└── README.md            # プロジェクトドキュメント
```

## 設定

`voicenote.py`ファイルを変更して、以下のような様々な設定を調整できます：

- アップロードおよび結果フォルダのパス
- 最大ファイルサイズ
- CeleryブローカーのURL
- GPUデバイスの選択

## ライセンス

LICENSEファイルを参照してください

## 謝辞

VoiceNoteプロジェクトは、以下のオープンソースプロジェクトの上に構築されています。これらのプロジェクトの開発者とコントリビューターに心から感謝いたします：

- [Faster Whisper](https://github.com/guillaumekln/faster-whisper): 高速で効率的な音声認識モデル
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