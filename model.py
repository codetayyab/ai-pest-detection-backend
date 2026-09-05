"""A lightweight, deterministic baseline detector.

This project deliberately does not load or download a trained model yet.  It
keeps the API response shape stable so that a trained detector can replace the
implementation later without requiring frontend changes.
"""

from PIL import Image, ImageStat

class PestDetector:
    def __init__(self):
        self.model_name = "Baseline color heuristic (no trained model)"
        self.classes = ["Aphids", "Spider Mites", "Whiteflies", "Caterpillars", "Healthy Plant"]
        self.solutions = {
            "Aphids": "Apply neem oil spray or introduce natural predators like ladybugs. Keep plants well-watered.",
            "Spider Mites": "Mist leaves regularly to increase humidity. Apply insecticidal soap or miticides under leaves.",
            "Whiteflies": "Use yellow sticky cards to trap them. Apply neem oil or horticultural oils.",
            "Caterpillars": "Handpick them from leaves. Apply organic Bacillus thuringiensis (Bt) spray if infestation is high.",
            "Healthy Plant": "Your plant is healthy! Continue regular watering, adequate sunlight, and balanced fertilization."
        }
        
    def predict(self, pil_image: Image.Image):
        image = pil_image.convert("RGB")
        width, height = image.size
        red, green, blue = ImageStat.Stat(image.resize((64, 64))).mean
        # This is intentionally only a baseline signal, not an AI diagnosis.
        pest_detected = "Healthy Plant" if green >= red * 0.9 else "Aphids"
        confidence = 0.55 if pest_detected != "Healthy Plant" else 0.60
        boxes_data = [{
            "box": [int(width * 0.15), int(height * 0.15), int(width * 0.85), int(height * 0.85)],
            "label": pest_detected,
            "confidence": confidence,
        }]
        return {
            "pest": pest_detected,
            "confidence": confidence,
            "solution": self.solutions[pest_detected],
            "width": width,
            "height": height,
            "boxes": boxes_data,
            "model": self.model_name
        }
