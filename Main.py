import cv2
import numpy as np
import argparse
import os
import sys
import json
from tqdm import tqdm
import csv  
import tkinter as tk
from tkinter import simpledialog
import pandas as pd
import multiprocessing as mp

from box_manager import BoxManager

def define_boxes(video_path, original_fps=30, slowed_fps=10, config_file=None):
    """
    Allows the user to interactively draw and modify boxes on the first frame of the video.
    
    Controls:
    - Draw a new box by dragging with the left mouse button.
    - Click near a box's corner (handle) to drag and reshape/rotate it.
    - Click inside a box (away from handles) to move the entire box.
    - 'z' to undo the last box.
    - 'r' to reset (remove) all boxes.
    - 's' to save configuration and exit.
    - 'q' to quit without saving.
    """
   
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read the video.")
        cap.release()
        return {}

    box_manager = BoxManager()

    if config_file and os.path.exists(config_file):
        try:
            box_manager.load_configuration(config_file)
            print(f"Loaded existing box configuration from {config_file}")
        except Exception as e:
            print(f"Error loading configuration: {e}")

    window_name = "Draw Boxes"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, box_manager.handle_mouse_event)

    print("\nControls:")
    print("- Draw new box by dragging with the left mouse button")
    print("- Click near a corner to drag it (rotate/reshape the box)")
    print("- Click inside a box to move it entirely")
    print("- Press 'z' to undo last box")
    print("- Press 's' to save configuration and exit")
    print("- Press 'q' to quit without saving")
    print("- Press 'r' to reset all boxes")

    while True:
        display_frame = box_manager.draw_boxes(frame)
        instructions = "Draw/move/resize boxes | 'z': undo | 's': save | 'q': quit | 'r': reset"
        cv2.putText(display_frame, instructions, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            if config_file:
                try:
                    box_manager.save_configuration(config_file)
                    print(f"Saved box configuration to {config_file}")
                except Exception as e:
                    print(f"Error saving configuration: {e}")
            break
        elif key == ord('z'):
            box_manager.remove_last_box()
        elif key == ord('r'):
            box_manager.boxes = []
            box_manager.labels = []
        elif key == ord('q'):
            box_manager.boxes = []
            box_manager.labels = []
            break

    cv2.destroyWindow(window_name)
    cap.release()

    return box_manager.get_box_data()

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

def preprocess_frame(frame, brightness_increase, clahe, scale_factor=0.5):
    if scale_factor != 1.0:
        frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    gray = cv2.add(gray, brightness_increase)
    
    enhanced = clahe.apply(gray)
    
    return enhanced, scale_factor

def detect_fish(enhanced, fgbg, min_contour_area=10):
    """
    Detect fish in the given frame using background subtraction and contour detection.
    """
    fg_mask = fgbg.apply(enhanced)
    
    # Define a kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # Apply erosion to remove noise
    eroded_mask = cv2.erode(fg_mask, kernel, iterations=1)
    
    # Apply dilation to close gaps
    dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=1)
    
    # Use the cleaned mask for contour detection
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return [c for c in contours if cv2.contourArea(c) > min_contour_area]

def process_frame(frame, fgbg, clahe, brightness_increase, scale_factor):
    enhanced, _ = preprocess_frame(frame, brightness_increase, clahe, scale_factor)
    contours = detect_fish(enhanced, fgbg)
    return enhanced, contours

def is_contour_in_box(contour, box):
    """
    Check if a given contour is inside a defined quadrilateral box.
    
    Args:
        contour: Contour points.
        box: A dictionary with box information, 
             where "coords" is a list of four corner tuples.
             
    Returns:
        True if the contour's center is within the box, False otherwise.
    """
    pts = np.array(box["coords"], dtype=np.int32).reshape((-1, 1, 2))
    x, y, w, h = cv2.boundingRect(contour)
    cx, cy = x + w / 2, y + h / 2
    return cv2.pointPolygonTest(pts, (cx, cy), False) >= 0

def draw_fish_contours(enhanced, contours, boxes, time_spent, original_fps, frame_skip=1):
    """
    Draws contours on the frame and updates time spent in each box only once per frame
    if any contour is detected inside the box. This prevents adding extra time if multiple
    contours are detected within the same box on a single frame.

    Args:
        enhanced: The image/frame to draw on.
        contours: List of detected contours.
        boxes: List of box dictionaries with a "coords" key.
        time_spent: List to accumulate time for each box.
        original_fps: The original FPS of the video.
        frame_skip: Number of skipped frames; each processed frame represents frame_skip/original_fps seconds.
    """
    detected_boxes = [False] * len(boxes)

    for contour in contours:
        if contour.dtype != np.int32:
            contour = contour.astype(np.int32)
        if cv2.contourArea(contour) < 10:
            continue  

        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(enhanced, (x, y), (x + w, y + h), (255, 255, 255), 2)

        
        cv2.drawContours(enhanced, [contour], -1, (255, 255, 255), 1)

        for i, box in enumerate(boxes):
            if is_contour_in_box(contour, box):
                detected_boxes[i] = True

    for i, detected in enumerate(detected_boxes):
        if detected:
            time_spent[i] += frame_skip / original_fps

def log_video_info(cap):
    print("Logging video information...")
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video Width: {width}, Height: {height}, FPS: {fps}")

def handle_key_press(key):
    if key == ord('q'):
        print("Quit key pressed. Exiting...")
        sys.exit()  
    return False

def main():
    print("Starting video processing...")
    path = "/Users/manasvenkatasairavulapalli/Desktop/Research Work/ml/Vid/originals/n2.mov"
    check_video_path(path)
    cap = initialize_video_capture(path)
    log_video_info(cap)

    # Create an instance of BoxManager
    box_manager = BoxManager()

    # Prompt user for input method
    user_choice = input("Would you like to (d)raw boxes or provide (c)oordinates? (d/c): ").strip().lower()

    if user_choice == 'c':
        # Allow user to input coordinates
        num_boxes = int(input("Enter the number of boxes you want to define: "))
        for i in range(num_boxes):
            print(f"Enter coordinates for Box {i+1} (format: x1,y1 x2,y2 x3,y3 x4,y4):")
            coords_input = input().strip()
            try:
                # Parse the input into a list of tuples
                coordinates = [tuple(map(int, point.split(','))) for point in coords_input.split()]
                box_manager.add_box_from_coordinates(coordinates, label=f"User Box {i+1}")
            except ValueError:
                print("Invalid input format. Please enter coordinates as x,y pairs separated by spaces.")
                return
        box_data = box_manager.get_box_data()
    else:
        # Default to drawing boxes
        box_data = define_boxes(path)
    
    print("User-defined boxes:", box_data)

    # Extract video filename without extension
    video_filename = os.path.splitext(os.path.basename(path))[0]

    # Create a directory for the current video file
    output_dir = f"/Users/manasvenkatasairavulapalli/Desktop/Research Work/ml/Data/{video_filename}"
    os.makedirs(output_dir, exist_ok=True)

    # Setup for logging coordinates to a CSV file
    coord_filename = os.path.join(output_dir, f"coord_{video_filename}.csv")
    with open(coord_filename, 'w', newline='') as coord_file:
        coord_writer = csv.writer(coord_file)
        # Write header: box name, coordinates
        coord_writer.writerow(["box_name", "coordinates"])
        for box_name, box_info in box_data.items():
            coord_writer.writerow([box_name, box_info["coords"]])

    # Setup for logging time spent in each box to a CSV file
    data_filename = os.path.join(output_dir, f"data_{video_filename}.csv")
    with open(data_filename, 'w', newline='') as data_file:
        data_writer = csv.writer(data_file)
        # Write header: box name, time spent
        data_writer.writerow(["box_name", "time_spent"])

        # Setup for logging center coordinates of contours
        center_filename = os.path.join(output_dir, f"center_{video_filename}.csv")
        with open(center_filename, 'w', newline='') as center_file:
            center_writer = csv.writer(center_file)
            # Write header: frame index, contour id, center_x, center_y
            center_writer.writerow(["frame", "contour_id", "center_x", "center_y"])

            # Processing parameters
            frame_skip = 1
            scale_factor = 1.0
            brightness_increase = 35
            contrast_clip_limit = 0.8
            min_contour_area = 15 

            clahe = cv2.createCLAHE(clipLimit=contrast_clip_limit, tileGridSize=(8,8))
            fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=30, detectShadows=False)

            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            time_spent = [0] * len(box_data)

            pbar = tqdm(total=total_frames, desc="Processing Video", unit="frame", dynamic_ncols=True)

            # Calculate the maximum number of frames to process (5 minutes)
            max_frames = int(original_fps * 5 * 60)

            while True:
                ret, frame = cap.read()
                if not ret or frame_count >= max_frames:
                    break

                # Skip frames based on frame_skip
                if frame_count % frame_skip != 0:
                    frame_count += 1
                    pbar.update(1)
                    continue

                # Process frame
                enhanced, contours = process_frame(frame, fgbg, clahe, brightness_increase, scale_factor)

                # Scale contours back to original size and convert to integers
                if scale_factor != 0.0:
                    contours = [np.round(c / scale_factor).astype(np.int32) for c in contours]

                # Log center coordinates of contours
                for idx, contour in enumerate(contours):
                    if cv2.contourArea(contour) < 10:
                        continue  # Skip tiny contours
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        center_x = int(M["m10"] / M["m00"])
                        center_y = int(M["m01"] / M["m00"])
                        center_writer.writerow([frame_count, idx, center_x, center_y])

                draw_fish_contours(enhanced, contours, list(box_data.values()), time_spent, original_fps, frame_skip=frame_skip)

                cv2.imshow("frame", enhanced)  # Display the enhanced (grayscale) frame with contours
                pbar.update(1)
                frame_count += 1

                key = cv2.waitKey(1) & 0xFF
                box_manager.handle_key_press(key)  # Use the box_manager instance to handle key press

            pbar.close()
            cap.release()
            cv2.destroyAllWindows()

            # Write time spent in each box to the CSV file
            for i, (box_name, box_info) in enumerate(box_data.items()):
                box_info["time"] = time_spent[i]
                data_writer.writerow([box_name, time_spent[i]])

if __name__ == "__main__":
    main()

