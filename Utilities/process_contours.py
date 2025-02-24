import cv2
import numpy as np
from tqdm import tqdm  # Import tqdm for the progress bar

def preprocess_frame(frame, brightness_increase, clahe):
    """
    Preprocess the frame for contour detection.

    Args:
        frame: The input video frame.
        brightness_increase: Value to increase brightness.
        clahe: CLAHE object for contrast enhancement.

    Returns:
        The preprocessed frame.
    """
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Increase brightness
    gray = cv2.add(gray, brightness_increase)
    
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use CLAHE for contrast enhancement
    enhanced = clahe.apply(blurred)
    
    return enhanced

def process_video(video_path):
    """
    Processes the video to extract contour areas.

    Args:
        video_path (str): Path to the video file.

    Returns:
        list: A list of the 10 largest contour areas detected in the video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return []

    contour_areas = []

    # Initialize CLAHE
    contrast_clip_limit = 0.85
    clahe = cv2.createCLAHE(clipLimit=contrast_clip_limit, tileGridSize=(8, 8))
    brightness_increase = 39  # Adjust this value as needed

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames
    with tqdm(total=total_frames, desc="Processing Frames", unit="frame", dynamic_ncols=True) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess the frame
            enhanced = preprocess_frame(frame, brightness_increase, clahe)

            # Use Canny edge detection
            edges = cv2.Canny(enhanced, 50, 150)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Store the area of each contour
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 0:  # Only consider contours with a positive area
                    contour_areas.append(area)

            pbar.update(1)  # Update the progress bar

    cap.release()

    # Get the 10 largest areas
    largest_areas = sorted(contour_areas, reverse=True)[:100]
    return largest_areas

if __name__ == "__main__":
    video_path = "/Users/manasvenkatasairavulapalli/Desktop/Research Work/ml/Vid/originals/n2.mov"  # Update this path to your video file
    largest_areas = process_video(video_path)
    # Print the largest areas
    print("Largest 10 Contour Areas:", largest_areas) 