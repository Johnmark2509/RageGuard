from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
import librosa
import joblib
import cv2
from deepface import DeepFace

app = FastAPI()

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load audio model
audio_model, audio_le = joblib.load("models/audio_emotion_model.pkl")

# Shared states
audio_emotion = "Not Angry"
face_emotion = "Not Angry"
final_emotion = "Not Angry"

# Audio processing
def extract_audio_features(audio, sr=44100):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

def predict_audio_emotion(audio):
    features = extract_audio_features(audio)
    pred = audio_model.predict([features])[0]
    return audio_le.inverse_transform([pred])[0]

# Routes
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze_frame")
async def analyze_frame(file: UploadFile = File(...)):
    global face_emotion
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], detector_backend='retinaface', enforce_detection=True)
        emotions = result[0]['emotion']
        angry_score = emotions.get('angry', 0)

        if angry_score > 70:
            face_emotion = "Very Angry"
        elif angry_score > 40:
            face_emotion = "Angry"
        elif angry_score > 10:
            face_emotion = "Slightly Angry"
        else:
            face_emotion = "Not Angry"

    except Exception as e:
        print("Face detection error:", e)
        face_emotion = "Unknown"

    return JSONResponse(content={"emotion": face_emotion})

@app.post("/analyze_audio")
async def analyze_audio(file: UploadFile = File(...)):
    global audio_emotion
    try:
        contents = await file.read()
        audio_np = np.frombuffer(contents, dtype=np.float32)

        if len(audio_np) > 0:
            audio_emotion = predict_audio_emotion(audio_np)
        else:
            audio_emotion = "Unknown"
    except Exception as e:
        print("Audio processing error:", e)
        audio_emotion = "Unknown"

    return JSONResponse(content={"audio_emotion": audio_emotion})

@app.get("/emotion_live")
async def emotion_live():
    global final_emotion

    angry_detected = (face_emotion in ["Slightly Angry", "Angry", "Very Angry"]) or (audio_emotion in ["Slightly Angry", "Angry", "Very Angry"])

    if angry_detected:
        final_emotion = face_emotion if face_emotion in ["Slightly Angry", "Angry", "Very Angry"] else audio_emotion
    else:
        final_emotion = "Not Angry"

    return JSONResponse(content={
        "emotion": final_emotion,
        "is_angry": angry_detected
    })

import os

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)