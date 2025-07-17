https://github.com/user-attachments/assets/e92505d0-6a2d-480b-bc3e-cbce3c0868fb

# Call-Audio Transcription & LLM Analysis

This project is a FastAPI web application for uploading call audio files, performing speaker diarization, transcribing speech to text, and analyzing the conversation using LLMs (AWS Bedrock). It supports outputting the transcript in Original Hindi, English (translated), or Hinglish (Hindi in Roman script).

## Features

- **Speaker Diarization** using pyannote-audio.
- **Speech-to-Text** using ai4bharat/indic-seamless (SeamlessM4T).
- **Transcript Output Options**:  
  - Original Hindi  
  - English (translated via LLM)  
  - Hinglish (transliterated via LLM)
- **Conversation Analysis** using AWS Bedrock LLM.
- **Web UI** with file upload, dropdown for output format, and loading spinner.
- **Environment Variable Support** for all sensitive tokens.

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/kxshagramathur/AI-Call-Analytics-Tool.git
cd your-repo
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

You may also need to install `ffmpeg` for audio processing.

### 3. Environment Variables

Create a `.env` file in the project root:

```
DIARI_TOKEN=your_huggingface_diari_token
SEAMLESS_ACCESS_TOKEN=your_huggingface_seamless_token
```

### 4. AWS Credentials

Ensure your AWS credentials are set up for Bedrock access (e.g., via `~/.aws/credentials` or environment variables).

### 5. Run the App

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Visit [http://localhost:8000](http://localhost:8000) in your browser.

## Usage

1. Upload an audio file (e.g., `.mp3`, `.wav`).
2. Select the desired transcript output format from the dropdown:
    - Original Hindi
    - English (translated)
    - Hinglish (Hindi in Roman script)
3. Click **Process**.
4. Wait for processing (spinner will show).
5. View the transcript and LLM analysis on the result page.

## Project Structure

```
.
├── main.py
├── requirements.txt
├── .env
├── templates/
│   └── index.html
└── ...
```

## Notes

- All tokens and secrets are loaded from `.env` and **should not** be hardcoded.
- The app uses GPU for inference (CUDA required).
- For production, remove `--reload` and consider using a production server.
