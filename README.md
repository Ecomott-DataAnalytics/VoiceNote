# VoiceNote

VoiceNote is a Flask-based web application that provides audio and video transcription services using the Faster Whisper model. It leverages GPU acceleration for efficient processing and uses Celery for asynchronous task management.

## Features

- Transcribe audio and video files using Faster Whisper
- Support for multiple languages (including auto-detection)
- Choice of model sizes (Large-v3 and Medium)
- Asynchronous processing with real-time progress updates
- GPU acceleration for faster transcription
- Web interface for easy file upload and result retrieval

## Requirements

- NVIDIA GPU with CUDA support
- [Singularity](https://sylabs.io/singularity/) container runtime

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/VoiceNote.git
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

## Configuration

You can modify the `voicenote.py` file to adjust various settings such as:

- Upload and result folder paths
- Maximum file size
- Celery broker URL
- GPU device selection

## License

See LICENSE

## Acknowledgments

The VoiceNote project is built upon the following open-source projects. We extend our heartfelt gratitude to the developers and contributors of these projects:

- [Faster Whisper](https://github.com/guillaumekln/faster-whisper): A fast and efficient speech recognition model
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
