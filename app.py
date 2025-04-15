from flask import Flask, request, jsonify, send_from_directory
import os
import boto3
import time
import json
import requests
from datetime import datetime
from moviepy import VideoFileClip

# --- CONFIG ---
REGION = 'us-east-1'
BUCKET_NAME = 'my-meeting-audio-bucket-svaid01'
UPLOAD_DIR = "tmp"
TMP_AUDIO = os.path.join(UPLOAD_DIR, "temp_audio.mp3")

# Ensure tmp folder exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)

def extract_audio(video_file, output_audio):
    print(f"🔊 Extracting audio from: {video_file}")
    video = VideoFileClip(video_file)
    try:
        video.audio.write_audiofile(output_audio)
    finally:
        video.reader.close()
        if video.audio:
            video.audio.reader.close()
        del video  # extra safety

def upload_to_s3(filename, bucket):
    print(f"☁️ Uploading to S3: {filename}")
    s3 = boto3.client('s3', region_name=REGION)
    s3.upload_file(filename, bucket, os.path.basename(filename))
    return f"s3://{bucket}/{os.path.basename(filename)}"

def transcribe_audio(s3_uri, job_name):
    transcribe = boto3.client('transcribe', region_name=REGION)
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': s3_uri},
        MediaFormat='mp3',
        LanguageCode='en-US'
    )

    # First wait for job to complete
    while True:
        result = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        status = result['TranscriptionJob']['TranscriptionJobStatus']
        print(f"🕒 Transcription status: {status}")
        if status in ['COMPLETED', 'FAILED']:
            break
        time.sleep(5)

    if status == 'FAILED':
        raise Exception("Transcription failed")

    # Now ensure Transcript URI is actually available
    attempts = 5
    for i in range(attempts):
        result = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        transcript_info = result['TranscriptionJob'].get('Transcript', {})
        transcript_url = transcript_info.get('TranscriptFileUri')
        if transcript_url:
            return transcript_url
        print(f"⏳ Waiting for transcript URL... attempt {i+1}")
        time.sleep(3)

    raise Exception("Transcript URL not available even after job completed")


def download_transcript_text(url):
    print(f"📥 Downloading transcript from: {url}")
    response = requests.get(url)
    data = response.json()
    return data['results']['transcripts'][0]['transcript']

def summarize_with_titan(text):
    print("🧠 Summarizing with Titan...")
    bedrock = boto3.client('bedrock-runtime', region_name=REGION)
    body = {
        "inputText": f"Summarize this meeting transcript into clear bullet points:\n\n{text}",
        "textGenerationConfig": {
            "maxTokenCount": 4096,
            "stopSequences": [],
            "temperature": 0.3,
            "topP": 1
        }
    }
    response = bedrock.invoke_model(
        modelId="amazon.titan-text-lite-v1",
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body)
    )
    return json.loads(response['body'].read())['results'][0]['outputText']

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/upload', methods=['POST'])
def upload():
    saved_filename = None
    try:
        uploaded_file = request.files['file']
        if not uploaded_file or uploaded_file.filename == '':
            return jsonify({'error': 'No file uploaded'}), 400

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_filename = os.path.join(UPLOAD_DIR, f"uploaded_{timestamp}.mp4")
        uploaded_file.save(saved_filename)
        uploaded_file.stream.close()

        print(f"📂 File saved: {saved_filename}")

        # Extract audio
        extract_audio(saved_filename, TMP_AUDIO)

        # Upload to S3
        s3_uri = upload_to_s3(TMP_AUDIO, BUCKET_NAME)

        # Transcribe
        job_name = f"transcription_{timestamp}"
        transcript_url = transcribe_audio(s3_uri, job_name)

        # Download and summarize
        transcript = download_transcript_text(transcript_url)
        summary = summarize_with_titan(transcript)

        return jsonify({'summary': summary})

    except Exception as e:
        print(f"❌ Exception: {e}")
        return jsonify({'error': str(e)}), 500

    finally:
        try:
            time.sleep(1)
            if saved_filename and os.path.exists(saved_filename):
                os.remove(saved_filename)
            if os.path.exists(TMP_AUDIO):
                os.remove(TMP_AUDIO)
        except Exception as cleanup_error:
            print("⚠️ Cleanup error:", cleanup_error)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
