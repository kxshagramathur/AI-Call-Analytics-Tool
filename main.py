import os
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
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

# Load environment variables from .env
load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

diari_token = os.getenv("DIARI_TOKEN")
seamless_access_token = os.getenv("SEAMLESS_ACCESS_TOKEN")

# Load models and pipeline at startup
diari_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=diari_token
)
diari_pipeline.to(torch.device("cuda"))

speech_model = SeamlessM4Tv2ForSpeechToText.from_pretrained("ai4bharat/indic-seamless", token=seamless_access_token).to("cuda")
processor = SeamlessM4TFeatureExtractor.from_pretrained("ai4bharat/indic-seamless", token=seamless_access_token)
tokenizer = SeamlessM4TTokenizer.from_pretrained("ai4bharat/indic-seamless", token=seamless_access_token)

bedrock_client = boto3.client("bedrock-runtime", region_name="ap-south-1")
bedrock_model_id = "apac.amazon.nova-micro-v1:0"

system_prompt = """
You are an assistant designed to analyze conversations between a customer and a customer service agent. You will receive a raw transcript of a conversation, often informal, fragmented, and potentially in a mixture of Hindi and English.

Your task is to analyze the conversation carefully and generate a detailed, structured report with specific insights. Your output must be returned as a valid JSON object with exactly the following 7 fields, using the same field names and structure.

Here is the required JSON format:

{
  "conversation_summary": "[Provide a detailed and businesslike summary of the entire conversation in English. Include specific item names, references to repeated actions, issues raised, and any relevant follow-ups or misunderstandings.]",
  
  "identified_issues": [
    "[List each specific issue the customer faced. Be detailed — include product names, monetary amounts, invoice dates, account numbers, or software problems as explicitly mentioned.]",
    "[Avoid vague phrases — each point should describe a concrete, specific issue.]"
  ],
  
  "resolution_status": "[Choose one of the following values only: Resolved, Partially Resolved – Follow-up Required, Unresolved. Make your decision based strictly on the conversation content. If multiple issues are discussed, choose the option that best reflects the overall situation.]",
  
  "customer_sentiment": "[Describe the customer's overall emotional tone in 1–2 lines — e.g., calm, frustrated, confused, impatient, cooperative. Base this on their language, tone, and urgency level.]",
  
  "sentiment_flow": [3, 3, 2, 2, 1, 1, 2, 3, 4, 4],
  // A list of 10 numeric values representing how the customer's sentiment changes throughout the conversation.
  // Use a scale of 1 (very negative/frustrated) to 5 (very positive/satisfied).
  // Segment the conversation evenly into 10 logical parts and infer the emotional tone in each part.

  "agent_rating": 4,
  // A single integer score from 1 to 10 evaluating the agent’s overall performance based on clarity, professionalism, helpfulness, and resolution ability. Do not include any explanation or text.

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
- Do not speculate — summarize only what is explicitly said in the transcript.
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

def hindi_to_english(original_transcript):
    system_prompt = """You are an expert transcription editor.\n\nYou will receive a conversation transcript in broken Hindi or Hindi-English. The conversation is informal, often fragmented, and may contain repetition or unclear expressions. Each line is labeled with a speaker tag like [SPEAKER_00] or [SPEAKER_01].\n\nYour task is to:\n1. Translate all Hindi and Hindi-English lines into fluent, grammatically correct English.\n2. Preserve the speaker labels exactly as given — do not remove or change them.\n3. Fix any sentence fragments, grammatical errors, or disjointed phrases to make the conversation clear and professional.\n4. Remove unnecessary, meaningless, or weird repetitions — especially repeated phrases repeated without context.\n5. Do **not** hallucinate or invent content. Keep the original meaning intact.\n6. Maintain the structure and order of the conversation.\n\nYour output must be a clean, well-structured English transcript with preserved speaker tags and no extra commentary.\nEnsure the final output is a clean, readable English transcript with the same speaker sequence as the original.\n"""
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
    system_prompt = """You are a transliteration assistant. Your job is to convert Hindi text written in Devanagari script into Hinglish — Hindi written using the English (Roman) alphabet. The result should sound natural and familiar to native Hindi speakers who use messaging apps like WhatsApp.\n\nGuidelines:\n- Do NOT translate into English. Preserve the meaning and tone.\n- Use casual Hinglish spelling commonly seen in texts and chats.\n- Avoid over-formal transliteration. Prioritize readability and familiarity over phonetic accuracy.\n- In the Hindi Input text their might be some wierd repitions like सदतत्त्रानियानियानियानियानियानियानियानियानियानियानिया these are error in the transcriptions models, please remove any meaningless repition words that dont make any sense, and dont transliterate them please \n- Do not change sentence structure or meaning.\n\nExamples:\nHindi: मैं अभी घर जा रहा हूँ।\nHinglish: Main abhi ghar ja raha hoon.\n\nHindi: क्या तुमने खाना खा लिया?\nHinglish: Kya tumne khana kha liya?\n\nHindi: बहुत अच्छा किया तुमने!\nHinglish: Bahut accha kiya tumne!\n"""
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

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/upload", response_class=HTMLResponse)
def upload_audio(request: Request, file: UploadFile = File(...), output_lang: str = Form(...)):
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
                "transcript": selected_transcript
            }
        )
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "result": False, "error": str(e)}
        )
