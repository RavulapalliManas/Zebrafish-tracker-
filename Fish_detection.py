import cv2
import numpy as np
import os
import sys
sys.path.append('C:/Users/91984/Desktop/ML models/preprocessing')
from Coordinate import define_boxes


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
    print("Preprocessing frame...")
    denoised = cv2.bilateralFilter(frame, 3, 15, 15)
    hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = cv2.add(hsv[:,:,2], brightness_increase)
    brightened = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(brightened, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    enhanced = clahe.apply(equalized)
    return enhanced

def detect_fish(enhanced, fgbg):
    print("Detecting fish...")
    fg_mask = fgbg.apply(enhanced)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_fish_contours(enhanced, contours):
    print("Drawing contours...")
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 10:
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(enhanced, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.putText(enhanced, "Fish", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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
    path = "C:/Users/91984/Desktop/ML models/videos/slowed_video.mp4"
    check_video_path(path)
    cap = initialize_video_capture(path)
    log_video_info(cap)

    wait_time = int(1000 / 10)
    brightness_increase = 35
    contrast_clip_limit = 0.8

    clahe = cv2.createCLAHE(clipLimit=contrast_clip_limit, tileGridSize=(8,8))
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break
        
        frame_count += 1
        print(f"Processing frame {frame_count}...")
        
        enhanced = preprocess_frame(frame, brightness_increase, clahe)
        contours = detect_fish(enhanced, fgbg)
        draw_fish_contours(enhanced, contours)
        
        cv2.imshow("frame", enhanced)
        
        key = cv2.waitKey(wait_time) & 0xFF
        if handle_key_press(key):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Video processing completed.")

if __name__ == "__main__":
    main()
