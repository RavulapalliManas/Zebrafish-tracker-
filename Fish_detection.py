import cv2
import numpy as np
import os
import sys
sys.path.append('C:/Users/91984/Desktop/ML models/preprocessing')
from tqdm import tqdm

def draw_boxes(event, x, y, flags, param):
    global current_box, boxes, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_box = [(x, y)]

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            current_box[1:] = [(x, y)]

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_box.append((x, y))
        boxes.append(tuple(current_box))


def define_boxes(video_path, original_fps=30, slowed_fps=10):
    """
    Allows user to draw bounding boxes on first frame of video and returns box coordinates with timing data.
    
    Args:
        video_path: Path to input video file
        original_fps: Original video FPS (default 30)
        slowed_fps: Target FPS for processing (default 10)
        
    Returns:
        Dictionary containing box coordinates and timing data
    """
    global current_box, boxes, drawing

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read the video.")
        cap.release()
        return {}

    current_box = []
    boxes = []
    drawing = False

    cv2.namedWindow("Draw Boxes")
    cv2.setMouseCallback("Draw Boxes", draw_boxes)

    print("Draw regions by dragging with the left mouse button. Press 's' to save and exit.")

    while True:
        temp_frame = frame.copy()

        for box in boxes:
            cv2.rectangle(temp_frame, box[0], box[1], (0, 255, 0), 2)

        if drawing and len(current_box) == 2:
            cv2.rectangle(temp_frame, current_box[0], current_box[1], (255, 0, 0), 2)

        cv2.imshow("Draw Boxes", temp_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            break

    cv2.destroyWindow("Draw Boxes")
    cap.release()

    box_data = {f"Box {i+1}": {"coords": box, "time": 0} for i, box in enumerate(boxes)}
    return box_data

def check_video_path(path):
    if not os.path.exists(path):
        print(f"Error: Video file not found at {path}")
        exit()

def initialize_video_capture(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        exit()
    return cap

def preprocess_frame(frame, brightness_increase, clahe):
    denoised = cv2.bilateralFilter(frame, 3, 15, 15)
    hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = cv2.add(hsv[:,:,2], brightness_increase)
    brightened = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(brightened, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    enhanced = clahe.apply(equalized)
    return enhanced

def detect_fish(enhanced, fgbg):
    fg_mask = fgbg.apply(enhanced)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def is_contour_in_box(contour, box):
    box_x1, box_y1 = box["coords"][0]
    box_x2, box_y2 = box["coords"][1]
    x, y, w, h = cv2.boundingRect(contour)
    return (box_x1 <= x <= box_x2 and box_y1 <= y <= box_y2) or \
           (box_x1 <= x + w <= box_x2 and box_y1 <= y + h <= box_y2)

def draw_fish_contours(enhanced, contours, boxes, time_spent, frame_count, original_fps):
    if contours:
        # Track if fish was detected in each box for this frame
        box_detections = [False] * len(boxes)
        
        for contour in contours:
            if cv2.contourArea(contour) > 10:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(enhanced, (x, y), (x + w, y + h), (255, 255, 255), 2)
                
                for i, box in enumerate(boxes):
                    if is_contour_in_box(contour, box):
                        box_detections[i] = True
                        
        # Add time for boxes where any contours were detected
        for i, detected in enumerate(box_detections):
            if detected:
                time_spent[i] += 1 / original_fps

def log_video_info(cap):
    print("Logging video information...")
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video Width: {width}, Height: {height}, FPS: {fps}")

def handle_key_press(key):
    if key == ord('q'):
        print("Quit key pressed. Exiting...")
        return True
    return False

def main():
    print("Starting video processing...")
    path = "C:/Users/91984/Desktop/ML models/videos/random_sample.mp4"
    check_video_path(path)
    cap = initialize_video_capture(path)
    log_video_info(cap)

    box_data = define_boxes(path)
    print("User-defined boxes:", box_data)

    wait_time = int(1000 / 10)
    brightness_increase = 35
    contrast_clip_limit = 0.8

    clahe = cv2.createCLAHE(clipLimit=contrast_clip_limit, tileGridSize=(8,8))
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames
    original_fps = cap.get(cv2.CAP_PROP_FPS)  # Get actual FPS instead of hardcoding
    total_duration = total_frames / original_fps  # Total duration in seconds
    time_spent = [0] * len(box_data)  # Initialize time spent for each box

    # Initialize progress bar with dynamic_ncols=True to prevent line breaks
    pbar = tqdm(total=total_frames, desc="Processing Video", unit="frame", dynamic_ncols=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\nEnd of video or error reading frame.")
            break
        
        # Print current frame number
        print(f"Processing frame {frame_count}/{total_frames}")

        frame_count += 1
        elapsed_time = frame_count / original_fps
        time_left = total_duration - elapsed_time
        percentage_complete = (frame_count / total_frames) * 100

        # Update progress bar without forcing new lines
        pbar.set_postfix_str(f"elapsed: {elapsed_time:.2f}s, remaining: {time_left:.2f}s, progress: {percentage_complete:.2f}%")
        pbar.update(1)
        
        enhanced = preprocess_frame(frame, brightness_increase, clahe)
        contours = detect_fish(enhanced, fgbg)
        draw_fish_contours(enhanced, contours, list(box_data.values()), time_spent, frame_count, original_fps)
        
        cv2.imshow("frame", enhanced)
        
        key = cv2.waitKey(wait_time) & 0xFF
        if handle_key_press(key):
            break

    pbar.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Video processing completed.")
    
    # Print the time spent in each box
    for i, time in enumerate(time_spent):
        print(f"Time spent in Box {i+1}: {time:.2f} seconds")
    
    return box_data

if __name__ == "__main__":
    box_data = main()
    print("Returned box data:", box_data)
