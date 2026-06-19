# VoiceNote

VoiceNote is a Flask-based web application for audio/video transcription with a **pluggable STT engine architecture**. You can switch between Faster-Whisper, Kotoba-Whisper, ReazonSpeech, and Qwen3-ASR through a common interface. It leverages GPU acceleration and uses Celery for asynchronous task management.

## Features

- **STT plugin architecture**: switch between Faster-Whisper / Kotoba-Whisper / ReazonSpeech / Qwen3-ASR
- Model definitions are externalized in `config/models.yaml` (add/change models without touching code)
- UI lists models grouped by category (Whisper / Kotoba / ReazonSpeech / Qwen3-ASR)
- Support for multiple languages (auto-detection; supported languages vary by engine)
- Asynchronous processing with real-time progress updates
- GPU acceleration for faster transcription
- Web interface for easy file upload and result retrieval

## Supported Engines / Models

| Category | Model ID | Engine | Languages | VRAM |
| --- | --- | --- | --- | --- |
| Whisper | `large-v3` | faster-whisper | multilingual | 4–6GB |
| Kotoba | `kotoba-tech/kotoba-whisper-v2.2` | kotoba (diarization) | Japanese | 5–8GB |
| Kotoba | `kotoba-tech/kotoba-whisper-v2.0` | kotoba (simple) | Japanese | 5–8GB |
| ReazonSpeech | `reazon-research/reazonspeech-k2-v2` | reazon | Japanese | 2–4GB |
| Qwen3-ASR | `Qwen/Qwen3-ASR-1.7B` | qwen3 | multilingual | 12–20GB |

**Recommended use**: default = Whisper Large-v3 (general multilingual) / Japanese & light = ReazonSpeech k2-v2 / high accuracy = Kotoba v2.2 (speaker diarization, meeting minutes) / best accuracy = Qwen3-ASR 1.7B (multilingual, heavy).

> **Kotoba v2.2 diarization**: requires accepting the terms of the gated pyannote models (`pyannote/segmentation-3.0`, `pyannote/speaker-diarization-3.1`) and passing a HuggingFace token via `HF_TOKEN` (or `SINGULARITYENV_HF_TOKEN`). If you don't want to set up a token, use Kotoba v2.0 (simple).

> **GPUs**: all models run on common CUDA setups including V100 (32GB) × 4. Since V100 (Volta) lacks bfloat16 / FlashAttention-2, Qwen3-ASR automatically runs in fp16 + sdpa.

> **Long files**: ReazonSpeech k2 and Qwen3-ASR process the whole input in a single pass, so the server automatically splits long inputs with ffmpeg (30s by default, configurable via `VN_CHUNK_SECONDS`) and transcribes chunk by chunk. Whisper / Kotoba handle long audio internally and are not split.

## Requirements

- NVIDIA GPU with CUDA support (driver R550+ when using CUDA 12.4)
- [Singularity](https://sylabs.io/singularity/) container runtime

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Ecomott-DataAnalytics/VoiceNote.git
   cd VoiceNote
   ```

2. Build the Singularity container:
   ```
   singularity build --fakeroot voicenote.sif voicenote.def
   ```

## Usage

1. Start the VoiceNote service:
   ```
   singularity run --nv --nvccli voicenote.sif
   ```

2. Open a web browser and navigate to `http://localhost:5000`

3. Upload an audio or video file, select the desired language and model, and click "Transcribe"

4. Wait for the transcription to complete and download the result

To use Kotoba v2.2 (diarization), pass a HuggingFace token at launch:

```
SINGULARITYENV_HF_TOKEN=hf_xxx singularity run --nv --nvccli voicenote.sif
```

## Running Tests

```bash
python -m pytest tests/ -v
# or
python -m unittest discover -s tests
```

## Project Structure

```
VoiceNote/
├── voicenote.py              # Main Flask app (Flask + Celery + endpoints)
├── voicenote.def             # Singularity container definition (CUDA 12.x)
├── requirements.txt          # Python dependencies (for local/CI reference)
├── engines/                  # STT engine plugins
│   ├── base.py               # STTEngine ABC + normalized TranscriptionResult
│   ├── factory.py            # EngineFactory (creates/caches by engine value)
│   ├── faster_whisper_engine.py
│   ├── kotoba_engine.py      # v2.2 (diarization) / v2.0 (simple)
│   ├── reazon_engine.py
│   └── qwen3_asr_engine.py   # auto fp16 (V100-compatible)
├── config/
│   ├── __init__.py           # YAML loader (load_models / get_model / grouped_for_ui)
│   └── models.yaml           # Model definitions (id/engine/category/label/vram/options)
├── static/
│   ├── css/style.css
│   └── js/main.js            # builds the dropdown dynamically from /models
├── templates/
│   └── index.html
└── tests/                    # Unit tests
```

## Adding a new model / engine

1. Add an entry to `config/models.yaml` (`id` / `engine` / `category` / `label` / `vram`, plus `options` if needed).
2. If a new engine is needed, add a class (subclass of `STTEngine`) under `engines/` and register it in `_REGISTRY` in `engines/factory.py`.
3. The UI dropdown is generated from `/models`, so no UI change is needed.

## Configuration

- Model definitions: `config/models.yaml`
- Upload/result folders, max file size, Celery broker URL: `voicenote.py`
- Qwen3-ASR backend: env var `QWEN_ASR_BACKEND` (`transformers` (default) / `vllm`)
- HuggingFace token: env var `HF_TOKEN` (for Kotoba v2.2)
- GPU selection: env var `CUDA_VISIBLE_DEVICES` (see `voicenote.def` comments for the multi-GPU worker example)

## License

See LICENSE

## Acknowledgments

The VoiceNote project is built upon the following open-source projects. We extend our heartfelt gratitude to the developers and contributors of these projects:

- [Faster Whisper](https://github.com/guillaumekln/faster-whisper): A fast and efficient speech recognition model
- [Kotoba-Whisper](https://huggingface.co/kotoba-tech): Japanese-focused Whisper derivatives (v2.2 supports diarization)
- [ReazonSpeech](https://github.com/reazon-research/ReazonSpeech): Japanese speech recognition models (k2-v2)
- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR): Multilingual ASR models by the Alibaba Cloud Qwen team (Apache-2.0)
- [Flask](https://flask.palletsprojects.com/): A lightweight and extensible Python web framework
- [Gunicorn](https://gunicorn.org/): A Python WSGI HTTP Server for UNIX
- [Celery](https://docs.celeryproject.org/): A distributed task queue system
- [Redis](https://redis.io/): A high-performance in-memory data store
- [PyTorch](https://pytorch.org/): An open-source machine learning framework
- [NVIDIA CUDA](https://developer.nvidia.com/cuda-zone): A GPU computing platform

We would also like to express our appreciation to the [OpenAI Whisper](https://github.com/openai/whisper) team. Without their groundbreaking research and publicly released models, this project would not have been possible.

VoiceNote would not exist without the efforts and dedication of these projects and their contributors. We extend our deepest appreciation to the open-source community.

## Note

This project is a renamed and extended version of the content originally published in the Ecomott Tech Blog article "[社内文字起こしサービス（仮）をサクっと立てた話](https://www.ecomottblog.com/?p=13901)" (How we quickly set up an in-house transcription service (tentative)).