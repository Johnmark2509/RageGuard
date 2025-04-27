from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import UploadFile, File
import threading
import cv2
import numpy as np
import sounddevice as sd
import librosa
import joblib
import time
from deepface import DeepFace

app = FastAPI()

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load models
audio_model, audio_le = joblib.load("models/audio_emotion_model.pkl")

# Shared states
audio_emotion = "Not Angry"
face_emotion = "Not Angry"
final_emotion = "Not Angry"

audio_rage_counter = 0
face_rage_counter = 0

lock = threading.Lock()

# Audio processing
def extract_audio_features(audio, sr=44100):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

def predict_audio_emotion(audio):
    features = extract_audio_features(audio)
    pred = audio_model.predict([features])[0]
    return audio_le.inverse_transform([pred])[0]

def audio_emotion_loop():
    global audio_emotion, audio_rage_counter
    fs = 44100
    duration = 1
    while True:
        try:
            audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
            sd.wait()
            audio_emotion = predict_audio_emotion(audio.flatten())
        except Exception as e:
            print("Audio error:", e)
            audio_emotion = "Unknown"
        time.sleep(1)

# Start audio thread
threading.Thread(target=audio_emotion_loop, daemon=True).start()


# Routes
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/emotion_live")
async def emotion_live():
    global final_emotion

    # Determine if either face or audio is angry
    angry_detected = (face_emotion in ["Slightly Angry", "Angry", "Very Angry"]) or (audio_emotion in ["Slightly Angry", "Angry", "Very Angry"])

    if angry_detected:
        final_emotion = face_emotion if face_emotion in ["Slightly Angry", "Angry", "Very Angry"] else audio_emotion
    else:
        final_emotion = "Not Angry"

    return JSONResponse(content={
        "emotion": final_emotion,
        "is_angry": angry_detected
    })

@app.post("/analyze_frame")
async def analyze_frame(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotions = result[0]['emotion']
        angry_score = emotions.get('angry', 0)

        if angry_score > 70:
            detected_emotion = "Very Angry"
        elif angry_score > 40:
            detected_emotion = "Angry"
        elif angry_score > 10:
            detected_emotion = "Slightly Angry"
        else:
            detected_emotion = "Not Angry"

    except Exception as e:
        print("Face detection error:", e)
        detected_emotion = "Unknown"

    return JSONResponse(content={"emotion": detected_emotion})

import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)