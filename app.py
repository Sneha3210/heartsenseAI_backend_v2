```python
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import traceback
import requests

# -------------------------------------------------
# Model Path
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "heartsense_model")

print("Model path:", MODEL_PATH)

ecg_model = None
model_error = None

try:
    print("Loading TensorFlow SavedModel...")
    ecg_model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully")

except Exception as e:
    print("Model loading failed")
    traceback.print_exc()
    model_error = str(e)
    ecg_model = None


# -------------------------------------------------
# ThingSpeak Configuration
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


# -------------------------------------------------
# Root Endpoint
# -------------------------------------------------

@app.get("/")
def home():
    return {"status": "HeartSense AI Backend Running"}


# -------------------------------------------------
# Health Endpoint
# -------------------------------------------------

@app.get("/health")
def health():
    return {
        "backend": "running",
        "model_loaded": ecg_model is not None,
        "model_error": model_error
    }


# -------------------------------------------------
# ThingSpeak Data Endpoint
# -------------------------------------------------

@app.get("/thingspeak-final-risk")
def thingspeak_final_risk():
    try:
        url = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds/last.json?api_key={THINGSPEAK_READ_API_KEY}"
        data = requests.get(url, timeout=5).json()

        return {
            "status": "success",
            "message": "ThingSpeak data fetched",
            "thingspeak_data": data
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
```
