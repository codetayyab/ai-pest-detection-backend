"""YOLOv8 inference adapter.

The default yolov8n.pt model is trained on generic COCO objects, not plant
pests. Its results must not be treated as agricultural diagnoses. Set
YOLO_MODEL_PATH to a pest-trained weight file when one is available.
"""

import os
from typing import Any

from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class PestDetector:
    def __init__(self) -> None:
        self.model: Any | None = None
        self.model_path = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
        self.model_name = "YOLOv8 base model unavailable"
        self.classes: list[str] = []

        if YOLO is None:
            return

        try:
            # Ultralytics downloads yolov8n.pt automatically on first use.
            self.model = YOLO(self.model_path)
            self.model_name = f"YOLOv8 ({os.path.basename(self.model_path)})"
            self.classes = [str(name) for name in self.model.names.values()]
        except Exception:
            # Prediction returns an explicit unavailable status rather than a
            # misleading "Healthy Plant" result when loading fails.
            self.model = None

    def predict(self, pil_image: Image.Image) -> dict[str, Any]:
        image = pil_image.convert("RGB")
        width, height = image.size

        if self.model is None:
            return {
                "pest": "Model unavailable",
                "confidence": 0.0,
                "solution": "YOLOv8 could not be loaded. Check deployment logs and YOLO_MODEL_PATH.",
                "width": width,
                "height": height,
                "boxes": [],
                "model": self.model_name,
            }

        try:
            result = self.model(image, verbose=False)[0]
            boxes_data = []
            primary_label = "No COCO object detected"
            primary_confidence = 0.0

            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                label = str(self.model.names[class_id])
                boxes_data.append({
                    "box": [int(value) for value in box.xyxy[0].tolist()],
                    "label": label,
                    "confidence": round(confidence, 2),
                })
                if confidence > primary_confidence:
                    primary_label = label
                    primary_confidence = confidence

            solution = (
                f"YOLOv8 base detected '{primary_label}'. This COCO base model is not trained to diagnose plant pests; "
                "use a pest-trained model before taking treatment action."
                if boxes_data
                else "No COCO object was detected. This is not evidence that the plant is healthy; a pest-trained model is required for diagnosis."
            )
            return {
                "pest": primary_label,
                "confidence": round(primary_confidence, 2),
                "solution": solution,
                "width": width,
                "height": height,
                "boxes": boxes_data,
                "model": self.model_name,
            }
        except Exception as error:
            return {
                "pest": "Inference failed",
                "confidence": 0.0,
                "solution": f"YOLOv8 inference failed: {error}",
                "width": width,
                "height": height,
                "boxes": [],
                "model": self.model_name,
            }
