from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import shutil
import os
import tempfile
import torch
import torchaudio
from pydub.utils import mediainfo
from pydub import AudioSegment
import io
from pyannote.audio import Pipeline
from transformers import SeamlessM4Tv2ForSpeechToText, SeamlessM4TTokenizer, SeamlessM4TFeatureExtractor
import boto3
from botocore.exceptions import ClientError
import subprocess

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load models and pipeline at startup
diari_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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"
)
diari_pipeline.to(torch.device("cuda"))

seamless_access_token = "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"
speech_model = SeamlessM4Tv2ForSpeechToText.from_pretrained("ai4bharat/indic-seamless", token=seamless_access_token).to("cuda")
processor = SeamlessM4TFeatureExtractor.from_pretrained("ai4bharat/indic-seamless", token=seamless_access_token)
tokenizer = SeamlessM4TTokenizer.from_pretrained("ai4bharat/indic-seamless", token=seamless_access_token)

bedrock_client = boto3.client("bedrock-runtime", region_name="ap-south-1")
bedrock_model_id = "apac.amazon.nova-micro-v1:0"

system_prompt = """
You are an assistant designed to analyze conversations between a customer and a customer service agent. You will receive a raw transcript of a conversation, often informal, fragmented, and potentially in a mixture of Hindi and English. Your task is to analyze the conversation carefully and generate a detailed, structured report with specific insights.

Your response MUST contain all four of the following sections, in this exact format:

Conversation Summary:
[Provide a detailed and businesslike summary of the entire conversation in English. Include specific item names, references to timestamps or repeated attempts, relevant actions taken, and any follow-up instructions or confusion discussed.]

Identified Issues:
[List each issue the customer faced, using specific details from the conversation. Include exact item names, invoice details, missing data points, software/system issues, etc.]
[Avoid generic phrases. Be precise and descriptive.]

Resolution Status:
[Select only one of the following options:
Resolved
Partially Resolved – Follow-up Required
Unresolved
Base your judgment on the conversation. Do not add any extra commentary. If multiple issues are discussed, base your choice on the overall status.]

Customer Sentiment:
[Briefly describe the customer's overall emotional tone during the conversation — e.g., calm, frustrated, confused, impatient, cooperative, etc. This should be 1-2 lines and reflect the customer's behavior, urgency, or satisfaction level.]

Important Instructions
Always include all four sections.
Use bullet points in Identified Issues.
Keep the tone professional and businesslike.
Be as specific and detailed as possible, especially with product names, time references, or transactional data.
Do not speculate — only summarize based on what is explicitly stated.
"""

def ensure_wav_format(input_path: str, sample_rate: int = 16000, channels: int = 1) -> str:
    try:
        info = mediainfo(input_path)
        if info.get('format_name') == 'wav' and info.get('codec_name') == 'pcm_s16le':
            return input_path
    except Exception:
        pass
    fd, output_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-c:a", "pcm_s16le",
        output_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{result.stderr.decode()}")
    return output_path

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/upload", response_class=HTMLResponse)
def upload_audio(request: Request, file: UploadFile = File(...)):
    # Save uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        # Step 1: Ensure WAV
        cleaned_audio_path = ensure_wav_format(tmp_path)
        audio, sample_rate = torchaudio.load(cleaned_audio_path)
        # Step 2: Diarization
        diarization = diari_pipeline({"waveform": audio, "sample_rate": sample_rate}, num_speakers=2)
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})
        # Step 3: Segment extraction
        full_audio = AudioSegment.from_wav(cleaned_audio_path)
        chunk_tensors = []
        for i, seg in enumerate(segments):
            start_ms = int(seg["start"] * 1000)
            end_ms = int(seg["end"] * 1000)
            speaker = seg["speaker"]
            audio_chunk = full_audio[start_ms:end_ms]
            buffer = io.BytesIO()
            audio_chunk.export(buffer, format="wav")
            buffer.seek(0)
            waveform, sr = torchaudio.load(buffer)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            chunk_tensors.append({
                "waveform": waveform,
                "sampling_rate": 16000,
                "speaker": speaker,
                "index": i
            })
        # Step 4: Transcription
        transcriptions = []
        for chunk in chunk_tensors:
            waveform = chunk["waveform"].squeeze(0).cpu()
            speaker = chunk["speaker"]
            index = chunk["index"]
            if waveform.numel() < 3200:
                continue
            inputs = processor(waveform, sampling_rate=16000, return_tensors="pt").to("cuda")
            with torch.no_grad():
                generated_tokens = speech_model.generate(**inputs, tgt_lang="hin")[0].cpu().numpy().squeeze()
            text = tokenizer.decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            transcriptions.append({"speaker": speaker, "index": index, "text": text})
        transcriptions = sorted(transcriptions, key=lambda x: x["index"])
        transcript_text = "\n".join([f"[{t['speaker']}]: {t['text']}" for t in transcriptions])
        # Step 5: LLM Analysis
        user_message = system_prompt.strip() + "\n\n" + transcript_text.strip()
        messages = [
            {
                "role": "user",
                "content": [{"text": user_message}],
            }
        ]
        try:
            response = bedrock_client.converse(
                modelId=bedrock_model_id,
                messages=messages,
                inferenceConfig={"maxTokens": 1024, "temperature": 0.2, "topP": 0.9},
            )
            llm_output = response["output"]["message"]["content"][0]["text"]
        except (ClientError, Exception) as e:
            llm_output = f"ERROR: Can't invoke '{bedrock_model_id}'. Reason: {e}"
        finally:
            if cleaned_audio_path != tmp_path:
                os.remove(cleaned_audio_path)
            os.remove(tmp_path)
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": True,
                "llm_output": llm_output,
                "transcript": transcript_text
            }
        )
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "result": False, "error": str(e)}
        )
