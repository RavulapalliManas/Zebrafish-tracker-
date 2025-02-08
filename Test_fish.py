import cv2
import numpy as np
import os
from ultralytics import YOLO
from tqdm import tqdm

def detect_fish(frame, model):
    # Run YOLO detection
    results = model.track(frame, persist=True)
    detections = []
    
    if results[0].boxes.xywh.cpu().numpy().size > 0:
        boxes = results[0].boxes.xywh.cpu().numpy()
        for box in boxes:
            x, y, w, h = box
            # Convert center coordinates to top-left and bottom-right
            x1, y1 = int(x - w/2), int(y - h/2)
            x2, y2 = int(x + w/2), int(y + h/2)
            detections.append(((x1, y1), (x2, y2)))
    
    return detections

def check_model(model_path):
    print(f"Checking model: {model_path}")
    try:
        model = YOLO(model_path)
        model.model.float()
        print("✅ Model is valid and can be loaded successfully")
        return True, model
    except Exception as e:
        print(f"❌ Model is corrupted or invalid: {str(e)}")
        return False, None

def main():
    print("Starting video processing...")
    video_path = "/Users/manasvenkatasairavulapalli/Desktop/Research Work/ml/random_sample.mp4"
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    # Initialize YOLO models with validation
    print("Loading YOLO models...")
    model_paths = [
        "/Users/manasvenkatasairavulapalli/Desktop/Research Work/ml/yolov8m(1).pt",  # Custom trained model
        # Add additional model paths here
    ]
    
    models = []
    for path in model_paths:
        is_valid, loaded_model = check_model(path)
        if is_valid:
            models.append(loaded_model)
            print(f"Successfully loaded model from {path}")
    
    if not models:
        print("Error: Could not load any models. Exiting.")
        return

    # Process video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pbar = tqdm(total=total_frames, desc="Processing Video", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Create a copy of frame for each model
        frames = [frame.copy() for _ in models]

        # Detect fish using each model
        for i, model in enumerate(models):
            detections = detect_fish(frames[i], model)
            
            # Draw detections with different colors for each model
            color = (0, 255 * (i / len(models)), 255 * (1 - i / len(models)))
            for detection in detections:
                cv2.rectangle(frames[i], detection[0], detection[1], color, 2)

            # Display frame for each model
            cv2.imshow(f"Model {i+1} Detection", frames[i])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        pbar.update(1)

    pbar.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 