import numpy as np
import requests
from PIL import Image
import io
from fastapi import FastAPI, File, UploadFile
import uvicorn

app = FastAPI()

endpoint = "http://localhost:8501/v1/models/potatoes:predict"
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(io.BytesIO(data)).convert("RGB").resize((256, 256))
    return np.array(image)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    response = requests.post(endpoint, json={"instances": img_batch.tolist()})
    prediction = np.array(response.json()["predictions"][0])

    return {
        "class": CLASS_NAMES[np.argmax(prediction)],
        "confidence": float(np.max(prediction)),
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)