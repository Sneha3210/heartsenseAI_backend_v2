import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import traceback
import requests
import numpy as np

# -------------------------------------------------
# Model Path (FIXED)
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "heartsense_model")  # ✅ FIXED

print("Model path:", MODEL_PATH)

ecg_model = None
model_error = None

try:
    print("Loading TensorFlow SavedModel...")

    ecg_model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("Model loaded successfully")

except Exception as e:
    print("Model loading failed")
    traceback.print_exc()
    model_error = str(e)
    ecg_model = None


# -------------------------------------------------
# ThingSpeak Config
# -------------------------------------------------

THINGSPEAK_CHANNEL_ID = "3143087"
THINGSPEAK_READ_API_KEY = "0BUW0FE0IF72M1VH"


# -------------------------------------------------
# FastAPI App
# -------------------------------------------------

app = FastAPI(title="HeartSense AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"status": "Backend Running"}


@app.get("/health")
def health():
    return {
        "backend": "running",
        "model_loaded": ecg_model is not None,
        "model_error": model_error
    }


def normalize_ecg(signal):
    signal = np.array(signal, dtype=np.float32)
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-6)


@app.get("/thingspeak-final-risk")
def thingspeak_final_risk():

    if ecg_model is None:
        return {
            "error": "Model not loaded",
            "details": model_error
        }

    try:
        url = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds.json?api_key={THINGSPEAK_READ_API_KEY}&results=180"
        response = requests.get(url, timeout=5).json()

        feeds = response.get("feeds", [])

        ecg = [
            float(f["field4"])
            for f in feeds
            if f.get("field4") and float(f["field4"]) != 0
        ]

        if not ecg:
            return {"status": "WAITING", "message": "No ECG data"}

        if len(ecg) < 180:
            ecg += [ecg[-1]] * (180 - len(ecg))

        ecg = ecg[-180:]

        ecg_tensor = tf.reshape(
            normalize_ecg(ecg),
            (1, 180, 1)
        )

        prediction = ecg_model.predict(ecg_tensor, verbose=0)

        predicted_class = int(tf.argmax(prediction, axis=1)[0])
        confidence = float(tf.reduce_max(prediction))

        return {
            "status": "success",
            "prediction": predicted_class,
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
