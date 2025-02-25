import cv2
import numpy as np
import argparse
import os
import sys
import json
import csv  
import time
from tqdm import tqdm
import csv  
import tkinter as tk
from tkinter import simpledialog
import pandas as pd
import multiprocessing as mp
from box_manager import BoxManager

MAX_DISTANCE_THRESHOLD = 150 

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
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    enhanced = clahe.apply(blurred)
    
    return enhanced, scale_factor

def detect_fish(enhanced, fgbg, min_contour_area=10, max_contour_area=1350):
    """
    Detect the largest fish in the given frame using background subtraction and contour detection.
    """
    fg_mask = fgbg.apply(enhanced)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded_mask = cv2.erode(fg_mask, kernel, iterations=1)
    dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=1)
    edges = cv2.Canny(dilated_mask, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        valid_contours = [cnt for cnt in contours if min_contour_area < cv2.contourArea(cnt) < max_contour_area]
        if valid_contours:
            largest_contour = max(valid_contours, key=cv2.contourArea)
            return [largest_contour]
    
    return []

def process_frame(frame, fgbg, clahe, brightness_increase, scale_factor):
    enhanced, _ = preprocess_frame(frame, brightness_increase, clahe, scale_factor)
    contours = detect_fish(enhanced, fgbg)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            return enhanced, contours, (center_x, center_y)
    
    return enhanced, contours, None

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

def draw_fish_contours(enhanced, contours, boxes, time_spent, original_fps, contour_areas=None):
    """
    Draws contours on the frame and updates time spent in each box.
    """
    detected_boxes = [False] * len(boxes)

    for i, contour in enumerate(contours):
        if contour.dtype != np.int32:
            contour = contour.astype(np.int32)
        
        # Use the pre-calculated area if provided
        area = contour_areas[i] if contour_areas else cv2.contourArea(contour)
        if area < 10 or area > 1350:  # Add max area check here
            continue  

        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(enhanced, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.drawContours(enhanced, [contour], -1, (255, 255, 255), 1)

        for j, box in enumerate(boxes):
            if is_contour_in_box(contour, box):
                detected_boxes[j] = True

    for i, detected in enumerate(detected_boxes):
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
        sys.exit()  
    return False

def write_center_data(center_writer, frame_count, idx, center_x, center_y, instantaneous_speed):
    """
    Write the center data to the CSV file.
    
    Args:
        center_writer: CSV writer object for the center file.
        frame_count: Current frame number.
        idx: Contour index.
        center_x: X-coordinate of the center.
        center_y: Y-coordinate of the center.
        instantaneous_speed: Calculated instantaneous speed in m/s.
    """
    center_writer.writerow([frame_count, idx, center_x, center_y, instantaneous_speed])

def main():
    print("Starting video processing...")
    path = "/Users/manasvenkatasairavulapalli/Desktop/Research Work/ml/Vid/originals/n1.mov"
    check_video_path(path)
    cap = initialize_video_capture(path)
    log_video_info(cap)

    # Add pixel to meter conversion constant
    PIXEL_TO_METER = 0.000099

    box_manager = BoxManager()
    user_choice = input("Would you like to (d)raw boxes or provide (c)oordinates? (d/c): ").strip().lower()

    if user_choice == 'c':
        num_boxes = int(input("Enter the number of boxes you want to define: "))
        for i in range(num_boxes):
            print(f"Enter coordinates for Box {i+1} (format: x1,y1 x2,y2 x3,y3 x4,y4):")
            coords_input = input().strip()
            try:
                coordinates = [tuple(map(int, point.split(','))) for point in coords_input.split()]
                box_manager.add_box_from_coordinates(coordinates, label=f"User Box {i+1}")
            except ValueError:
                print("Invalid input format. Please enter coordinates as x,y pairs separated by spaces.")
                return
        box_data = box_manager.get_box_data()
    else:
        box_data = define_boxes(path)
    
    print("User-defined boxes:", box_data)

    video_filename = os.path.splitext(os.path.basename(path))[0]
    output_dir = f"/Users/manasvenkatasairavulapalli/Desktop/Research Work/ml/Data/{video_filename}"
    os.makedirs(output_dir, exist_ok=True)

    coord_filename = os.path.join(output_dir, f"coord_{video_filename}.csv")
    with open(coord_filename, 'w', newline='') as coord_file:
        coord_writer = csv.writer(coord_file)
        coord_writer.writerow(["box_name", "coordinates"])
        for box_name, box_info in box_data.items():
            coord_writer.writerow([box_name, box_info["coords"]])

    data_filename = os.path.join(output_dir, f"data_{video_filename}.csv")
    with open(data_filename, 'w', newline='') as data_file:
        data_writer = csv.writer(data_file)
        data_writer.writerow(["box_name", "time_spent (s)", "average_speed (m/s)"])

        center_filename = os.path.join(output_dir, f"center_{video_filename}.csv")
        with open(center_filename, 'w', newline='') as center_file:
            center_writer = csv.writer(center_file)
            center_writer.writerow(["frame", "contour_id", "center_x (px)", "center_y (px)", "instantaneous_speed (m/s)"])

            frame_skip = 1
            scale_factor = 1.0
            brightness_increase = 39
            contrast_clip_limit = 0.85
            min_contour_area = 15

            clahe = cv2.createCLAHE(clipLimit=contrast_clip_limit, tileGridSize=(8,8))
            fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)

            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            time_spent = [0] * len(box_data)

            max_frames = min(9000, total_frames)
            pbar = tqdm(total=max_frames, desc="Processing Video", unit="frame", dynamic_ncols=True)

            previous_center = None
            total_speed = 0
            speed_count = 0

            while True:
                ret, frame = cap.read()
                if not ret or frame_count >= max_frames:
                    break

                enhanced, contours, current_center = process_frame(frame, fgbg, clahe, brightness_increase, scale_factor)

                # Initialize instantaneous_speed
                instantaneous_speed = 0  # Default value

                # Check distance and process next largest contour if threshold exceeded
                if current_center and previous_center:
                    dx = current_center[0] - previous_center[0]
                    dy = current_center[1] - previous_center[1]
                    distance = np.sqrt(dx**2 + dy**2)

                    if distance > MAX_DISTANCE_THRESHOLD:
                        # Find the next largest contour
                        if len(contours) > 1:
                            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
                            next_largest_contour = sorted_contours[1]  # Get the second largest contour
                            contours = [next_largest_contour]  # Replace contours with the next largest contour
                        else:
                            contours = []  # No valid contours to process

                    instantaneous_speed = distance * original_fps * PIXEL_TO_METER  # Calculate speed
                    total_speed += instantaneous_speed
                    speed_count += 1
                else:
                    instantaneous_speed = 0  # No movement detected

                previous_center = current_center

                contour_areas = []  # Initialize an empty list to store contour areas
                for idx, contour in enumerate(contours):
                    area = cv2.contourArea(contour)
                    if area < 10:
                        continue
                    contour_areas.append(area)  # Append the area to the list
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        center_x = int(M["m10"] / M["m00"])
                        center_y = int(M["m01"] / M["m00"])
                        # Write center data with speed
                        write_center_data(center_writer, frame_count, idx, center_x, center_y, instantaneous_speed)

                draw_fish_contours(enhanced, contours, list(box_data.values()), time_spent, original_fps, contour_areas=contour_areas)

                pbar.update(1)
                frame_count += 1

            pbar.close()
            cap.release()

            # Update the data writer to show m/s
            data_writer.writerow(["box_name", "time_spent (s)", "average_speed (m/s)"])
            
            for i, (box_name, box_info) in enumerate(box_data.items()):
                box_info["time"] = time_spent[i]
                average_speed = total_speed / speed_count if speed_count > 0 else 0
                data_writer.writerow([box_name, time_spent[i], average_speed])

if __name__ == "__main__":
    main()