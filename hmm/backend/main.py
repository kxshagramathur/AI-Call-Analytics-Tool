import os
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware  # Add this import
import shutil
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
from dotenv import load_dotenv
import json  # Add this import

# Load environment variables from .env
load_dotenv()

app = FastAPI()

# Add CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

diari_token = os.getenv("DIARI_TOKEN")
seamless_access_token = os.getenv("SEAMLESS_ACCESS_TOKEN")

# Load models and pipeline at startup
diari_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=diari_token)
diari_pipeline.to(torch.device("cuda"))

speech_model = SeamlessM4Tv2ForSpeechToText.from_pretrained("ai4bharat/indic-seamless", token=seamless_access_token).to("cuda")
processor = SeamlessM4TFeatureExtractor.from_pretrained("ai4bharat/indic-seamless", token=seamless_access_token)
tokenizer = SeamlessM4TTokenizer.from_pretrained("ai4bharat/indic-seamless", token=seamless_access_token)

bedrock_client = boto3.client("bedrock-runtime", region_name="ap-south-1")
bedrock_model_id = "apac.amazon.nova-micro-v1:0"

system_prompt = """You are an assistant designed to analyze conversations between a customer and a customer service agent. You will receive a raw transcript of a conversation, often informal, fragmented, and potentially in a mixture of Hindi and English.

Your task is to analyze the conversation carefully and generate a detailed, structured report with specific insights. Your output must be returned as a valid JSON object with exactly the following 7 fields, using the same field names and structure.

Here is the required JSON format:
{
  "conversation_summary": "[Provide a detailed and businesslike summary of the entire conversation in English. Include specific item names, references to repeated actions, issues raised, and any relevant follow-ups or misunderstandings.]",
  "identified_issues": [
    "[List each specific issue the customer faced. Be detailed — include product names, monetary amounts, invoice dates, account numbers, or software problems as explicitly mentioned.]",
    "[Avoid vague phrases — each point should describe a concrete, specific issue.]"
  ],
  "resolution_status": "[Choose one of the following values only: Resolved, Partially Resolved, Unresolved. Make your decision based strictly on the conversation content. If multiple issues are discussed, choose the option that best reflects the overall situation.]",
  "customer_sentiment": "[Describe the customer's overall emotional tone in 1–2 lines — e.g., calm, frustrated, confused, impatient, cooperative. Base this on their language, tone, and urgency level.]",
  "sentiment_flow": [3, 3, 2, 2, 1, 1, 2, 3, 4, 4],
  "agent_rating": 4,
  "agent_suggestions": [
    "[Provide 1–2 case-specific, professional suggestions for how the agent could have handled the situation better.]",
    "[Suggestions must be constructive and directly related to the interaction.]"
  ]
}

Important Instructions:
- Always return all seven fields in this exact JSON format.
- Ensure the JSON is valid and machine-readable (no trailing commas or extra text).
- Keep the tone professional and analytical.
- Be specific — cite product names, monetary amounts, dates, or actions wherever available.
- Do not speculate — summarize only what is explicitly said in the transcript."""

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

def hindi_to_english(original_transcript):
    system_prompt = """You are an expert transcription editor.

You will receive a conversation transcript in broken Hindi or Hindi-English. The conversation is informal, often fragmented, and may contain repetition or unclear expressions. Each line is labeled with a speaker tag like [SPEAKER_00] or [SPEAKER_01].

Your task is to:
1. Translate all Hindi and Hindi-English lines into fluent, grammatically correct English.
2. Preserve the speaker labels exactly as given — do not remove or change them.
3. Fix any sentence fragments, grammatical errors, or disjointed phrases to make the conversation clear and professional.
4. Remove unnecessary, meaningless, or weird repetitions — especially repeated phrases repeated without context.
5. Do **not** hallucinate or invent content. Keep the original meaning intact.
6. Maintain the structure and order of the conversation.

Your output must be a clean, well-structured English transcript with preserved speaker tags and no extra commentary.
Ensure the final output is a clean, readable English transcript with the same speaker sequence as the original.
"""
    user_message = system_prompt.strip() + "\n\n" + original_transcript.strip()
    messages = [
        {
            "role": "user",
            "content": [{"text": user_message}],
        }
    ]

    try:
        response = bedrock_client.converse(
            modelId="apac.amazon.nova-micro-v1:0",
            messages=messages,
            inferenceConfig={"maxTokens": 5000, "temperature": 0.2, "topP": 0.9},
        )
        return response["output"]["message"]["content"][0]["text"]
    except (ClientError, Exception) as e:
        return f"ERROR: Can't translate to English. Reason: {e}"

def hindi_to_hinglish(original_transcript):
    system_prompt = """You are a transliteration assistant. Your job is to convert Hindi text written in Devanagari script into Hinglish — Hindi written using the English (Roman) alphabet. The result should sound natural and familiar to native Hindi speakers who use messaging apps like WhatsApp.

Guidelines:
- Do NOT translate into English. Preserve the meaning and tone.
- Use casual Hinglish spelling commonly seen in texts and chats.
- Avoid over-formal transliteration. Prioritize readability and familiarity over phonetic accuracy.
- In the Hindi Input text their might be some wierd repitions like सदतत्त्रानियानियानियानियानियानियानियानियानियानियानिया these are error in the transcriptions models, please remove any meaningless repition words that dont make any sense, and dont transliterate them please 
- Do not change sentence structure or meaning.

Examples:
Hindi: मैं अभी घर जा रहा हूँ।
Hinglish: Main abhi ghar ja raha hoon.

Hindi: क्या तुमने खाना खा लिया?
Hinglish: Kya tumne khana kha liya?

Hindi: बहुत अच्छा किया तुमने!
Hinglish: Bahut accha kiya tumne!
"""
    user_message = system_prompt.strip() + "\n\n" + original_transcript.strip()
    messages = [
        {
            "role": "user",
            "content": [{"text": user_message}],
        }
    ]

    try:
        response = bedrock_client.converse(
            modelId="apac.amazon.nova-pro-v1:0",
            messages=messages,
            inferenceConfig={"maxTokens": 10000, "temperature": 0.2, "topP": 0.9},
        )
        return response["output"]["message"]["content"][0]["text"]
    except (ClientError, Exception) as e:
        return f"ERROR: Can't transliterate to Hinglish. Reason: {e}"

def transcription_output(original_transcript, output_lang):
    if output_lang == "hindi":
        return original_transcript
    elif output_lang == "english":
        return hindi_to_english(original_transcript)
    elif output_lang == "hinglish":
        return hindi_to_hinglish(original_transcript)
    else:
        return "ERROR: Invalid output language selected."

def process_audio_pipeline(tmp_path, output_lang):
    """
    Shared audio processing pipeline for both HTML and API endpoints
    """
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

    # Step 4.5: Apply output format selection
    selected_transcript = transcription_output(transcript_text, output_lang)

    # Step 5: LLM Analysis (always use original Hindi transcript)
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

    # Cleanup
    if cleaned_audio_path != tmp_path:
        os.remove(cleaned_audio_path)

    return selected_transcript, llm_output

# Keep your existing HTML route
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

# Add new API endpoint for React frontend
@app.post("/api/analyze", response_class=JSONResponse)
async def analyze_audio_api(file: UploadFile = File(...), output_lang: str = Form(...)):
    """
    API endpoint for React frontend to analyze audio files
    Returns JSON response with analysis results
    """
    # Save uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        selected_transcript, llm_output = process_audio_pipeline(tmp_path, output_lang)
        
        # Parse the LLM output as JSON
        try:
            analysis_data = json.loads(llm_output)
        except json.JSONDecodeError:
            # If JSON parsing fails, return error
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to parse LLM analysis output", "raw_output": llm_output}
            )

        # Format transcript for frontend
        transcript_lines = []
        for line in selected_transcript.split('\n'):
            if line.strip():
                if line.startswith('[SPEAKER_00]'):
                    speaker = "Customer"
                    text = line.replace('[SPEAKER_00]:', '').strip()
                elif line.startswith('[SPEAKER_01]'):
                    speaker = "Agent"
                    text = line.replace('[SPEAKER_01]:', '').strip()
                else:
                    # Handle other speaker formats
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        speaker = parts[0].strip('[]')
                        text = parts[1].strip()
                    else:
                        speaker = "Unknown"
                        text = line.strip()
                
                transcript_lines.append({"speaker": speaker, "text": text})

        # Combine analysis data with transcript
        result = {
            **analysis_data,  # Spread the LLM analysis data
            "transcript": transcript_lines
        }

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
    
    finally:
        # Cleanup temp files
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# Keep your existing HTML upload route
@app.post("/upload", response_class=HTMLResponse)
def upload_audio(request: Request, file: UploadFile = File(...), output_lang: str = Form(...)):
    # Save uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        selected_transcript, llm_output = process_audio_pipeline(tmp_path, output_lang)

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": True,
                "llm_output": llm_output,
                "transcript": selected_transcript
            }
        )

    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "result": False, "error": str(e)}
        )
    
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
