import time
from typing import List

import cv2
import numpy as np
from PIL import Image

from model import PestDetector


def draw_boxes(frame: np.ndarray, boxes: List[dict]) -> None:
    for item in boxes:
        x1, y1, x2, y2 = item["box"]
        label = item["label"]
        confidence = item["confidence"]
        color = (34, 197, 94) if label.lower() == "healthy plant" else (245, 158, 11)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} ({int(confidence * 100)}%)"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - text_height - 8), (x1 + text_width + 8, y1), color, -1)
        cv2.putText(
            frame,
            text,
            (x1 + 4, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )


def is_gui_available() -> bool:
    try:
        cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("test")
        return True
    except Exception:
        return False


def main() -> None:
    detector = PestDetector()
    print(f"Loaded detector: {detector.model_name}")

    use_gui = is_gui_available()
    if not use_gui:
        print("Warning: OpenCV GUI is unavailable. Running in terminal-only mode.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the webcam. Make sure a camera is connected.")
        return

    last_prediction_time = 0.0
    prediction_interval = 1.5  # seconds
    result = None
    last_printed = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read camera frame.")
                break

            now = time.time()
            if now - last_prediction_time >= prediction_interval:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = detector.predict(image)
                last_prediction_time = now

            if result is not None:
                draw_boxes(frame, result.get("boxes", []))
                label_text = f"{result['pest']} ({result['confidence']:.2f})"
                cv2.putText(
                    frame,
                    label_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                if not use_gui:
                    output = (
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                        f"Detected: {result['pest']} ({result['confidence']:.2f}) | "
                        f"Boxes: {len(result.get('boxes', []))}"
                    )
                    if output != last_printed:
                        print(output)
                        for box in result.get('boxes', []):
                            print(f"  - {box['label']} @ {box['box']} ({box['confidence']:.2f})")
                        last_printed = output

            if use_gui:
                cv2.imshow("AI Pest Detection Camera", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # In terminal mode, sleep a short time to keep the loop responsive.
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Exiting.")
    finally:
        cap.release()
        if use_gui:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
