import os
import io
import json
import base64
import datetime
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

# Import our model
from model import PestDetector

app = FastAPI(title="AI Pest Detection API", description="FastAPI Backend for Plant Disease & Pest Detection")

# Enable CORS for frontend integration
cors_origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Detector
detector = PestDetector()

# History database path
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "history.json")

def load_history() -> List[dict]:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading history: {e}")
        return []

def save_history(history: List[dict]):
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"Error saving history: {e}")

def add_to_history(pest: str, confidence: float, solution: str, image_base64: str, boxes: List[dict]):
    history = load_history()
    # Format date-time
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    new_entry = {
        "id": len(history) + 1,
        "pest": pest,
        "confidence": confidence,
        "solution": solution,
        "timestamp": now,
        "image": f"data:image/jpeg;base64,{image_base64}" if not image_base64.startswith("data:") else image_base64,
        "boxes": boxes
    }
    
    # Prepend to history (newest first)
    history.insert(0, new_entry)
    
    # Limit to last 50 entries to avoid bloating
    save_history(history[:50])
    return new_entry

# Request body model for frame prediction
class FrameRequest(BaseModel):
    image: str # Base64 encoded image

@app.get("/")
def read_root():
    return {
        "status": "online",
        "model_loaded": detector.model_name,
        "supported_classes": detector.classes
    }

@app.get("/health")
def health_check():
    return {"status": "ok", "model": detector.model_name}

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded is not an image.")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Run prediction
        result = detector.predict(image)
        
        # Convert uploaded image to base64 for saving to history
        buffered = io.BytesIO()
        # Resize slightly to keep history file size manageable
        image_copy = image.copy()
        image_copy.thumbnail((400, 400))
        image_copy.save(buffered, format="JPEG", quality=80)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Save to history
        add_to_history(
            pest=result["pest"],
            confidence=result["confidence"],
            solution=result["solution"],
            image_base64=img_str,
            boxes=result["boxes"]
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")

@app.post("/predict-frame")
async def predict_frame(request: FrameRequest):
    try:
        # Decode base64 image
        header, encoded = request.image.split(",", 1) if "," in request.image else ("", request.image)
        image_data = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Run prediction
        result = detector.predict(image)
        
        # Convert image to compressed base64 for history
        buffered = io.BytesIO()
        image_copy = image.copy()
        image_copy.thumbnail((300, 300))
        image_copy.save(buffered, format="JPEG", quality=75)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Save to history
        add_to_history(
            pest=result["pest"],
            confidence=result["confidence"],
            solution=result["solution"],
            image_base64=img_str,
            boxes=result["boxes"]
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process frame: {str(e)}")

@app.get("/history")
def get_history():
    return load_history()

@app.delete("/history")
def clear_history():
    save_history([])
    return {"message": "History cleared successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
