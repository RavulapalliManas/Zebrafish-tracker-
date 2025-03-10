import cv2 as cv2
import numpy as np
import os
import sys
import csv
from tqdm import tqdm
import json
import yaml
from box_manager import BoxManager
import tkinter as tk
from tkinter import messagebox
import shutil
import time
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import requests
from office365.runtime.auth.authentication_context import AuthenticationContext
from office365.sharepoint.client_context import ClientContext
from office365.sharepoint.files.file import File
import re
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle

# Constants
MAX_DISTANCE_THRESHOLD = 200 
PIXEL_TO_METER = 0.000099  
VISUALIZATION_FRAME_SKIP = 15 

def download_from_google_drive(file_id, destination=None):
    """
    Download a file from Google Drive.
    
    Args:
        file_id (str): The ID of the file to download.
        destination (str, optional): The path where the file will be saved.
                                    If None, will use the original filename.
    
    Returns:
        tuple: (bool, str) - Success status and the path where the file was saved.
    """
    try:
        # Load credentials from environment
        creds_json = os.environ.get('GOOGLE_CREDENTIALS')
        if not creds_json:
            print("Google credentials not found. Please run setup_google_credentials() first.")
            return False, None
            
        creds_data = json.loads(creds_json)
        
        # Create credentials object
        creds = Credentials(
            token=creds_data.get('token'),
            refresh_token=creds_data.get('refresh_token'),
            token_uri=creds_data.get('token_uri'),
            client_id=creds_data.get('client_id'),
            client_secret=creds_data.get('client_secret'),
            scopes=creds_data.get('scopes')
        )
        
        # Build the Drive API client
        service = build('drive', 'v3', credentials=creds)
        
        # Get file metadata to retrieve the original filename
        file_metadata = service.files().get(fileId=file_id, fields="name").execute()
        original_filename = file_metadata.get('name', 'downloaded_file')
        
        # If destination is not specified, use the original filename
        if destination is None:
            destination = os.path.join(os.getcwd(), original_filename)
        
        # Create a BytesIO object to store the downloaded file
        request = service.files().get_media(fileId=file_id)
        file_handle = io.BytesIO()
        downloader = MediaIoBaseDownload(file_handle, request)
        
        # Download the file
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}%")
        
        # Save the file to the destination
        with open(destination, 'wb') as f:
            f.write(file_handle.getvalue())
            
        return True, destination
    except Exception as e:
        print(f"Error downloading from Google Drive: {e}")
        return False, None

def download_from_onedrive(file_url, destination):
    """
    Download a file from OneDrive.
    
    Args:
        file_url (str): The URL of the file to download.
        destination (str): The path where the file will be saved.
    
    Returns:
        bool: True if download was successful, False otherwise.
    """
    try:
        # Get credentials from environment variables
        username = os.environ.get('ONEDRIVE_USERNAME')
        password = os.environ.get('ONEDRIVE_PASSWORD')
        site_url = os.environ.get('ONEDRIVE_SITE_URL')
        
        # Authenticate
        ctx_auth = AuthenticationContext(site_url)
        ctx_auth.acquire_token_for_user(username, password)
        ctx = ClientContext(site_url, ctx_auth)
        
        # Download the file
        response = File.open_binary(ctx, file_url)
        
        # Save the file to the destination
        with open(destination, 'wb') as f:
            f.write(response.content)
            
        return True
    except Exception as e:
        print(f"Error downloading from OneDrive: {e}")
        return False

def get_video_files_from_google_drive(folder_id):
    """
    Get a list of video files from a Google Drive folder.
    
    Args:
        folder_id (str): The ID of the folder to search.
    
    Returns:
        list: A list of dictionaries containing file IDs and names.
    """
    try:
        # Load credentials from environment
        creds_json = os.environ.get('GOOGLE_CREDENTIALS')
        if not creds_json:
            print("Google credentials not found. Please run setup_google_credentials() first.")
            return []
            
        creds_data = json.loads(creds_json)
        
        # Create credentials object
        creds = Credentials(
            token=creds_data.get('token'),
            refresh_token=creds_data.get('refresh_token'),
            token_uri=creds_data.get('token_uri'),
            client_id=creds_data.get('client_id'),
            client_secret=creds_data.get('client_secret'),
            scopes=creds_data.get('scopes')
        )
        
        # Build the Drive API client
        service = build('drive', 'v3', credentials=creds)
        
        # Query for video files in the folder
        query = f"'{folder_id}' in parents and (mimeType contains 'video/')"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        
        return results.get('files', [])
    except Exception as e:
        print(f"Error getting files from Google Drive: {e}")
        return []

def get_video_files_from_onedrive(folder_url):
    """
    Get a list of video files from a OneDrive folder.
    
    Args:
        folder_url (str): The URL of the folder to search.
    
    Returns:
        list: A list of dictionaries containing file URLs and names.
    """
    try:
        # Get credentials from environment variables
        username = os.environ.get('ONEDRIVE_USERNAME')
        password = os.environ.get('ONEDRIVE_PASSWORD')
        site_url = os.environ.get('ONEDRIVE_SITE_URL')
        
        # Authenticate
        ctx_auth = AuthenticationContext(site_url)
        ctx_auth.acquire_token_for_user(username, password)
        ctx = ClientContext(site_url, ctx_auth)
        
        # Get files from the folder
        folder = ctx.web.get_folder_by_server_relative_url(folder_url)
        files = folder.files
        ctx.load(files)
        ctx.execute_query()
        
        # Filter for video files
        video_extensions = ['.mp4', '.mov', '.avi', '.wmv', '.mkv']
        video_files = []
        
        for file in files:
            file_name = file.properties['Name']
            file_ext = os.path.splitext(file_name)[1].lower()
            if file_ext in video_extensions:
                video_files.append({
                    'url': file.properties['ServerRelativeUrl'],
                    'name': file_name
                })
        
        return video_files
    except Exception as e:
        print(f"Error getting files from OneDrive: {e}")
        return []

def get_local_video_files(folder_path):
    """
    Get a list of video files from a local folder.
    
    Args:
        folder_path (str): The path to the folder to search.
    
    Returns:
        list: A list of dictionaries containing file paths and names.
    """
    video_extensions = ['.mp4', '.mov', '.avi', '.wmv', '.mkv']
    video_files = []
    
    try:
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                file_ext = os.path.splitext(file_name)[1].lower()
                if file_ext in video_extensions:
                    video_files.append({
                        'path': file_path,
                        'name': file_name
                    })
        return video_files
    except Exception as e:
        print(f"Error getting local video files: {e}")
        return []

def define_boxes(video_path, original_fps=30, slowed_fps=10, config_file=None):
    """
    Allows the user to interactively draw and modify boxes on the first frame of the video.

    Controls:
        - Draw a new box by dragging with the left mouse button.
        - Click near a box's corner (handle) to drag and reshape/rotate it.
        - Click inside a box (away from handles) to move the entire box.
        - Double-click on an existing corner to use it when creating a new box.
        - 'z' to undo the last box.
        - 'r' to reset (remove) all boxes.
        - 's' to save configuration and exit.
        - 'q' to quit without saving.

    Args:
        video_path (str): Path to the video file.
        original_fps (int, optional): Original frames per second of the video. Defaults to 30.
        slowed_fps (int, optional): Slowed frames per second for processing. Defaults to 10.
        config_file (str, optional): Path to a configuration file to load existing boxes. Defaults to None.

    Returns:
        dict: Dictionary containing box data if saved, empty dictionary otherwise.
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
    print("- Double-click on an existing corner to use it when creating a new box")
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
    """
    Checks if the video file exists at the given path.

    Args:
        path (str): Path to the video file.
    """
    if not os.path.exists(path):
        print(f"Error: Video file not found at {path}")
        exit()

def initialize_video_capture(path):
    """
    Initializes video capture from the given path.

    Args:
        path (str): Path to the video file.

    Returns:
        cv2.VideoCapture: Video capture object.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        exit()
    return cap

def preprocess_frame(frame, brightness_increase, clahe, scale_factor=0.5):
    """
    Preprocesses a single video frame.

    Includes resizing, grayscale conversion, brightness adjustment, Gaussian blur, 
    sharpening, and CLAHE enhancement.

    Args:
        frame (np.array): Input video frame.
        brightness_increase (int): Value to increase brightness by.
        clahe (cv2.CLAHE): CLAHE object for contrast enhancement.
        scale_factor (float, optional): Scaling factor for resizing the frame. Defaults to 0.5.

    Returns:
        tuple: Enhanced frame and the scale factor used.
    """
    if scale_factor != 1.0:
        frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.add(gray, brightness_increase)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    sharpening_kernel = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]])
    sharpened = cv2.filter2D(blurred, -1, sharpening_kernel)
    
    enhanced = clahe.apply(sharpened)

    return enhanced, scale_factor

def detect_fish(enhanced, bg_subtractor, min_contour_area=10, max_contour_area=1350):
    """
    Detect the largest fish in the given frame using background subtraction and contour detection.

    Args:
        enhanced (np.array): Preprocessed grayscale frame.
        bg_subtractor (cv2.BackgroundSubtractorKNN): Background subtractor object.
        min_contour_area (int, optional): Minimum contour area to consider as fish. Defaults to 10.
        max_contour_area (int, optional): Maximum contour area to consider as fish. Defaults to 1350.

    Returns:
        list: List of largest contour if found within area limits, otherwise empty list.
    """
    fg_mask = bg_subtractor.apply(enhanced)
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

def process_frame(frame, bg_subtractor, clahe, brightness_increase, scale_factor):
    """
    Processes a single frame to detect fish and its center.

    Args:
        frame (np.array): Input video frame.
        bg_subtractor (cv2.BackgroundSubtractorKNN): Background subtractor object.
        clahe (cv2.CLAHE): CLAHE object for contrast enhancement.
        brightness_increase (int): Brightness increase value.
        scale_factor (float): Frame scaling factor.

    Returns:
        tuple: Enhanced frame, detected contours, and center of the largest contour if found, 
               otherwise None for center.
    """
    enhanced, _ = preprocess_frame(frame, brightness_increase, clahe, scale_factor)
    contours = detect_fish(enhanced, bg_subtractor)

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
        contour (np.array): Contour points.
        box (dict): A dictionary with box information,
                   where "coords" is a list of four corner tuples.

    Returns:
        bool: True if the contour's center is within the box, False otherwise.
    """
    pts = np.array(box["coords"], dtype=np.int32).reshape((-1, 1, 2))

    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return cv2.pointPolygonTest(pts, (cx, cy), False) >= 0
    return False

def draw_fish_contours(enhanced, contours, boxes, time_spent, distance_in_box, prev_box_positions, 
                      original_fps, current_frame, contour_areas=None):
    """
    Draws contours on the frame and updates time spent and distance in each box.
    """
    detected_boxes = [False] * len(boxes)
    current_box_positions = [None] * len(boxes)

    # Track if any valid fish contour was detected in this frame
    valid_fish_detected = False

    for i, contour in enumerate(contours):
        if contour.dtype != np.int32:
            contour = contour.astype(np.int32)

        area = contour_areas[i] if contour_areas else cv2.contourArea(contour)
        if area < 10 or area > 1350:
            continue
            
        # If we found a valid fish contour
        valid_fish_detected = True

        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(enhanced, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.drawContours(enhanced, [contour], -1, (255, 255, 255), 1)

        M = cv2.moments(contour)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            current_center = (center_x, center_y)
            
            # Find which box the fish is in (only count the fish in ONE box per frame)
            box_found = False
            for j, box in enumerate(boxes):
                if is_contour_in_box(contour, box) and not box_found:
                    box_found = True
                    detected_boxes[j] = True
                    current_box_positions[j] = current_center
                    
                    # Calculate distance traveled within the box
                    if prev_box_positions[j] is not None:
                        dx = current_center[0] - prev_box_positions[j][0]
                        dy = current_center[1] - prev_box_positions[j][1]
                        distance = np.sqrt(dx**2 + dy**2)
                        
                        # Filter out unreasonable movements
                        if distance <= MAX_DISTANCE_THRESHOLD:
                            distance_in_box[j] += distance * PIXEL_TO_METER

    # Only update time if a valid fish was detected
    if valid_fish_detected:
        # Only increment time for ONE box per frame (the one with the fish)
        # This ensures total time doesn't exceed video length
        box_with_fish = None
        for i, detected in enumerate(detected_boxes):
            if detected:
                if box_with_fish is None:
                    box_with_fish = i
                    time_spent[i] += 1 / original_fps
                    prev_box_positions[i] = current_box_positions[i]
                else:
                    # Don't count time for additional boxes in the same frame
                    prev_box_positions[i] = current_box_positions[i]
            else:
                # Reset previous position if fish is no longer in the box
                prev_box_positions[i] = None

def log_video_info(cap):
    """
    Logs video width, height, and FPS to the console.

    Args:
        cap (cv2.VideoCapture): Video capture object.
    """
    print("Logging video information...")
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video Width: {width}, Height: {height}, FPS: {fps}")

def handle_key_press(key):
    """
    Handles key press events for controlling video processing.

    Args:
        key (int): Key code of the pressed key.

    Returns:
        bool: True if a handled key was pressed, False otherwise.
    """
    if key == ord('q'):
        print("Quit key pressed. Exiting...")
        sys.exit()
    return False

def write_center_data(center_writer, frame_count, idx, center_x, center_y, speed):
    """
    Write the center data to the CSV file.

    Args:
        center_writer (csv.writer): CSV writer object for the center file.
        frame_count (int): Current frame number.
        idx (int): Contour index.
        center_x (int): X-coordinate of the center.
        center_y (int): Y-coordinate of the center.
        speed (float): Calculated speed in m/s.
    """
    center_writer.writerow([frame_count, idx, center_x, center_y, speed])

def create_tank_mask(frame, points=None):
    """
    Create a binary mask for the tank area.

    If points are not provided, the user is prompted to define the tank area interactively
    on the first frame.

    Args:
        frame (np.array): Input video frame.
        points (list, optional): List of points defining the tank boundary. Defaults to None.

    Returns:
        tuple: Binary mask where tank area is white (255) and outside area is black (0),
               and the list of points defining the mask.
    """
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    if points is None:
        clone = frame.copy()
        cv2.namedWindow("Define Tank Area")
        points = []

        def click_and_crop(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                if len(points) > 1:
                    cv2.line(clone, points[-2], points[-1], (0, 255, 0), 2)
                    cv2.imshow("Define Tank Area", clone)

        cv2.setMouseCallback("Define Tank Area", click_and_crop)

        while True:
            cv2.imshow("Define Tank Area", clone)
            key = cv2.waitKey(1) & 0xFF

            if (key == ord("c") or key == ord("C")) and len(points) > 2:
                cv2.line(clone, points[0], points[-1], (0, 255, 0), 2)
                cv2.imshow("Define Tank Area", clone)
                break

        cv2.destroyWindow("Define Tank Area")
    else:
        if len(points) > 2:
            pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)

    return mask, points

def apply_tank_mask(frame, mask):
    """
    Apply tank mask to the frame.

    Args:
        frame (np.array): Input frame.
        mask (np.array): Tank mask.

    Returns:
        np.array: Frame with tank mask applied.
    """
    return cv2.bitwise_and(frame, frame, mask=mask)

def is_fish_movement_valid(current_center, previous_center, previous_valid_center, max_distance):
    """
    Check if fish movement is valid based on distance and consistency.

    Args:
        current_center (tuple): Current detected position (x, y).
        previous_center (tuple): Position from the previous frame (x, y).
        previous_valid_center (tuple): Last known valid position (x, y).
        max_distance (int): Maximum allowed movement distance in pixels.

    Returns:
        tuple: Tuple of (is_valid, center_to_use), where is_valid is a boolean indicating
               if the movement is valid, and center_to_use is the center to be used.
    """
    if previous_center is None:
        return True, current_center

    dx = current_center[0] - previous_center[0]
    dy = current_center[1] - previous_center[1]
    distance = np.sqrt(dx**2 + dy**2)

    if distance <= max_distance:
        return True, current_center

    return False, previous_valid_center

def visualize_processing(original, masked, enhanced, fg_mask, edges, fish_vis, current_center, previous_center,
                       distance=None, instantaneous_speed=None, is_valid_movement=True,
                       is_outside_tank=False, is_showing_previous=False, reflection_detected=False, 
                       status_text=None, cumulative_speed=None, window_size=1.0):
    """
    Visualize the processing steps in separate windows.

    Args:
        original (np.array): Original frame.
        masked (np.array): Frame after tank mask applied.
        enhanced (np.array): Preprocessed frame.
        fg_mask (np.array): Background subtraction mask.
        edges (np.array): Edge detection result.
        fish_vis (np.array): Fish visualization frame.
        current_center (tuple): Current detected center (x, y).
        previous_center (tuple): Previous center point (x, y).
        distance (float, optional): Distance between current and previous centers. Defaults to None.
        instantaneous_speed (float, optional): Calculated speed. Defaults to None.
        is_valid_movement (bool, optional): Whether movement is valid. Defaults to True.
        is_outside_tank (bool, optional): Whether detection is outside tank. Defaults to False.
        is_showing_previous (bool, optional): Whether using previous position. Defaults to False.
        reflection_detected (bool, optional): Whether reflection was detected. Defaults to False.
        status_text (str, optional): Status text to display on visualization. Defaults to None.
        cumulative_speed (float, optional): Speed calculated over a window of frames.
        window_size (float, optional): Size of the window in seconds.

    Returns:
        int: Key pressed by user (for handling quit).
    """
    cv2.imshow("Original", original)
    cv2.imshow("Masked", masked)
    cv2.imshow("Enhanced", enhanced)
    cv2.imshow("Background Mask", fg_mask)
    cv2.imshow("Edges", edges)

    if current_center and previous_center:
        cv2.line(fish_vis, previous_center, current_center, (255, 255, 0), 2)

    if current_center:
        if reflection_detected:
            cv2.circle(fish_vis, current_center, 8, (0, 255, 255), -1)  # Yellow for reflection
            cv2.putText(fish_vis, "Reflection detected", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        elif is_outside_tank:
            cv2.circle(fish_vis, current_center, 8, (0, 0, 255), -1)
            cv2.putText(fish_vis, "Detection outside tank", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif is_showing_previous:
            cv2.circle(fish_vis, current_center, 8, (0, 0, 255), -1)
            cv2.putText(fish_vis, "Using previous position", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.circle(fish_vis, current_center, 5, (255, 0, 0), -1)

    # Only show cumulative speed (remove instantaneous speed display)
    if cumulative_speed is not None:
        speed_text = f"Avg Speed ({window_size:.1f}s): {cumulative_speed:.4f} m/s"
        cv2.putText(fish_vis, speed_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if status_text:
        cv2.putText(fish_vis, status_text, (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Detected Fish", fish_vis)

    return cv2.waitKey(1) & 0xFF

def is_reflection(frame, center, tolerance=10):
    """
    Check if a detected point is likely a reflection based on RGB values.
    
    When RGB values are very close to each other, it often indicates a white/gray reflection.
    
    Args:
        frame (np.array): Original color frame.
        center (tuple): Center coordinates (x, y) to check.
        tolerance (int, optional): Maximum allowed difference between RGB channels. Defaults to 10.
        
    Returns:
        bool: True if likely a reflection, False otherwise.
    """
    try:
        # Check if center coordinates are valid
        if center is None:
            return False
            
        x, y = center
        height, width = frame.shape[:2]
        
        # Boundary check
        if not (0 <= x < width and 0 <= y < height):
            return False
            
        # Check if frame has 3 channels (BGR)
        if len(frame.shape) != 3:
            return False
            
        # Safely extract BGR values
        b = int(frame[y, x, 0])
        g = int(frame[y, x, 1])
        r = int(frame[y, x, 2])
        
        # Ensure values are within valid range (0-255)
        b = max(0, min(b, 255))
        g = max(0, min(g, 255))
        r = max(0, min(r, 255))
        
        # Check if RGB values are within tolerance of each other
        return (abs(r - g) <= tolerance and 
                abs(r - b) <= tolerance and 
                abs(g - b) <= tolerance)
                
    except Exception as e:
        print(f"Warning: Error in reflection detection: {e}")
        return False

def show_error(message):
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    messagebox.showerror("Error", message)
    root.destroy()

def process_video(video_path, output_dir, box_data=None, tank_points=None, enable_visualization=False):
    """
    Process a single video file for fish tracking.
    
    Args:
        video_path (str): Path to the video file.
        output_dir (str): Directory to save output files.
        box_data (dict, optional): Box data to use for processing. If None, user will be prompted.
        tank_points (list, optional): Tank boundary points. If None, user will be prompted.
        enable_visualization (bool, optional): Whether to enable visualization. Defaults to False.
        
    Returns:
        tuple: Box data and tank points used for processing.
    """
    check_video_path(video_path)
    cap = initialize_video_capture(video_path)
    log_video_info(cap)

    box_manager = BoxManager()
    
    # If box_data is not provided, prompt user to define boxes
    if box_data is None:
        user_choice = input("Would you like to (d)raw boxes, provide (c)oordinates, or load tank (j)son? (d/c/j): ").strip().lower()
        
        if user_choice == 'j':
            json_file = input("Enter the path to the JSON file with box and tank coordinates: ").strip()
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    tank_points = data.get("tank_coordinates", [])
                    if not tank_points:
                        show_error("Error: No tank coordinates found in the JSON file.")
                        return None, None
                    # Load box data if needed
                    box_data = {key: value for key, value in data.items() if key != "tank_coordinates"}
            except Exception as e:
                show_error(f"Error loading JSON file: {e}")
                return None, None
        elif user_choice == 'c':
            num_boxes = int(input("Enter the number of boxes you want to define: "))
            for i in range(num_boxes):
                print(f"Enter coordinates for Box {i+1} (format: x1,y1 x2,y2 x3,y3 x4,y4):")
                coords_input = input().strip()
                try:
                    coordinates = [tuple(map(int, point.split(','))) for point in coords_input.split()]
                    if len(coordinates) != 4:
                        print(f"Error: You must provide exactly 4 points for Box {i+1}. Got {len(coordinates)}.")
                        continue
                    box_manager.add_box_from_coordinates(coordinates, label=f"User Box {i+1}")
                except ValueError:
                    print("Invalid input format. Please enter coordinates as x,y pairs separated by spaces.")
                    return None, None
            box_data = box_manager.get_box_data()
        else:
            box_data = define_boxes(video_path)

    print("Using boxes:", box_data)

    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_filename)
    os.makedirs(video_output_dir, exist_ok=True)

    coord_filename = os.path.join(video_output_dir, f"box_details_{video_filename}.csv")
    with open(coord_filename, 'w', newline='') as coord_file:
        coord_writer = csv.writer(coord_file)
        coord_writer.writerow(["box_name", "coordinates"])
        for box_name, box_info in box_data.items():
            coord_writer.writerow([box_name, box_info["coords"]])

    data_filename = os.path.join(video_output_dir, f"fish_data_{video_filename}.csv")
    with open(data_filename, 'w', newline='') as data_file:
        data_writer = csv.writer(data_file)
        data_writer.writerow(["box_name", "time_spent (s)", "distance_traveled (m)", "average_speed (m/s)"])

        center_filename = os.path.join(video_output_dir, f"fish_coords_{video_filename}.csv")
        with open(center_filename, 'w', newline='') as center_file:
            center_writer = csv.writer(center_file)
            center_writer.writerow(["frame", "contour_id", "center_x (px)", "center_y (px)", 
                                     "speed (m/s)"])

            frame_skip = 1
            scale_factor = 1.0
            brightness_increase = 39
            contrast_clip_limit = 0.85
            min_contour_area = 20
            sharpening_strength = 9
            
            sharpening_kernel = np.array([[-1, -1, -1],
                                         [-1, sharpening_strength, -1],
                                         [-1, -1, -1]])

            clahe = cv2.createCLAHE(clipLimit=contrast_clip_limit, tileGridSize=(8,8))
            bg_subtractor = cv2.createBackgroundSubtractorKNN(history=1000, dist2Threshold=400.0, detectShadows=True)

            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            time_spent = [0] * len(box_data)
            distance_in_box = [0] * len(box_data)
            prev_box_positions = [None] * len(box_data)

            max_frames = min(9000, total_frames)
            pbar = tqdm(total=max_frames, desc=f"Processing {video_filename}", unit="frame", dynamic_ncols=True)

            previous_center = None
            total_speed = 0
            speed_count = 0
            visualization_frame_count = 0

            position_history = []  # Will store (frame_num, x, y) tuples
            speed_window_size = int(original_fps)  # Use 1 second window (adjust as needed)

            ret, first_frame = cap.read()
            if not ret:
                print("Error: Cannot read the first frame.")
                return None, None

            # Use provided tank_points or prompt user
            if tank_points is None:
                tank_mask, tank_points = create_tank_mask(first_frame)
            else:
                tank_mask, _ = create_tank_mask(first_frame, points=tank_points)
            
            print(f"Tank boundary defined with {len(tank_points)} points")

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            if enable_visualization:
                cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
                cv2.namedWindow("Masked", cv2.WINDOW_NORMAL)
                cv2.namedWindow("Enhanced", cv2.WINDOW_NORMAL)
                cv2.namedWindow("Background Mask", cv2.WINDOW_NORMAL)
                cv2.namedWindow("Edges", cv2.WINDOW_NORMAL)
                cv2.namedWindow("Detected Fish", cv2.WINDOW_NORMAL)

                cv2.resizeWindow("Original", 640, 480)
                cv2.resizeWindow("Masked", 640, 480)
                cv2.resizeWindow("Enhanced", 640, 480)
                cv2.resizeWindow("Background Mask", 640, 480)
                cv2.resizeWindow("Edges", 640, 480)
                cv2.resizeWindow("Detected Fish", 640, 480)

            previous_valid_center = None
            previous_valid_contour = None

            while True:
                ret, frame = cap.read()
                if not ret or frame_count >= max_frames:
                    break

                masked_frame = apply_tank_mask(frame, tank_mask)

                if scale_factor != 1.0:
                    frame_resized = cv2.resize(masked_frame, None, fx=scale_factor, fy=scale_factor)
                else:
                    frame_resized = masked_frame.copy()

                gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                gray = cv2.add(gray, brightness_increase)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                
                sharpened = cv2.filter2D(blurred, -1, sharpening_kernel)
                
                enhanced = clahe.apply(sharpened)

                fg_mask = bg_subtractor.apply(enhanced)
                fg_mask = cv2.bitwise_and(fg_mask, tank_mask)

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                eroded_mask = cv2.erode(fg_mask, kernel, iterations=1)
                dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=1)
                edges = cv2.Canny(dilated_mask, 50, 150)

                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                valid_contours = []
                for cnt in contours:
                    area = cv2.contourArea(cnt)

                    if min_contour_area < area < 1350:
                        valid_contours.append(cnt)

                fish_vis = None
                if enable_visualization:
                    fish_vis = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                    cv2.drawContours(fish_vis, valid_contours, -1, (0, 255, 0), 2)

                    for i, box in enumerate(box_data.values()):
                        pts = np.array(box["coords"], dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(fish_vis, [pts], True, (0, 0, 255), 2)
                        
                        # Add box number and speed
                        center_x = int(np.mean([pt[0] for pt in box["coords"]]))
                        center_y = int(np.mean([pt[1] for pt in box["coords"]]))
                        
                        # Calculate current average speed in the box
                        if time_spent[i] > 0:
                            avg_speed = distance_in_box[i] / time_spent[i]
                            box_text = f"Box {i+1}: {avg_speed:.4f} m/s"
                            cv2.putText(fish_vis, box_text, (center_x, center_y), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                current_center = None
                current_contour = None
                is_outside_tank = False
                is_showing_previous = False
                is_reflection_detected = False

                if valid_contours:
                    largest_contour = max(valid_contours, key=cv2.contourArea)
                    current_contour = largest_contour
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        center_x = int(M["m10"] / M["m00"])
                        center_y = int(M["m01"] / M["m00"])
                        current_center = (center_x, center_y)

                        if is_reflection(frame, current_center):
                            is_reflection_detected = True
                            if previous_valid_contour is not None:
                                current_contour = previous_valid_contour
                                M = cv2.moments(previous_valid_contour)
                                if M["m00"] != 0:
                                    center_x = int(M["m10"] / M["m00"])
                                    center_y = int(M["m01"] / M["m00"])
                                    current_center = (center_x, center_y)
                                    is_showing_previous = True

                        if not is_reflection_detected:
                            if center_y < tank_mask.shape[0] and center_x < tank_mask.shape[1] and tank_mask[center_y, center_x] > 0:
                                is_valid_movement, center_to_use = is_fish_movement_valid(
                                    current_center,
                                    previous_center,
                                    previous_valid_center if previous_valid_center else current_center,
                                    MAX_DISTANCE_THRESHOLD
                                )

                                if is_valid_movement:
                                    previous_valid_center = current_center
                                    previous_valid_contour = current_contour
                                else:
                                    current_center = center_to_use
                                    current_contour = previous_valid_contour if previous_valid_contour is not None else current_contour
                                    is_showing_previous = True
                            else:
                                is_outside_tank = True
                                current_center = previous_valid_center
                                current_contour = previous_valid_contour if previous_valid_contour is not None else current_contour

                distance = None
                cumulative_speed = 0

                if current_center and previous_center:
                    dx = current_center[0] - previous_center[0]
                    dy = current_center[1] - previous_center[1]
                    distance = np.sqrt(dx**2 + dy**2)

                    # Store current position with frame number
                    position_history.append((frame_count, current_center[0], current_center[1]))
                    
                    # Remove positions older than the window size
                    while len(position_history) > 0 and position_history[0][0] < frame_count - speed_window_size:
                        position_history.pop(0)
                    
                    # Calculate cumulative distance if we have enough positions
                    if len(position_history) > 1:
                        cumulative_distance = 0
                        for i in range(1, len(position_history)):
                            prev_pos = position_history[i-1]
                            curr_pos = position_history[i]
                            
                            # Calculate distance between consecutive positions
                            pos_dx = curr_pos[1] - prev_pos[1]
                            pos_dy = curr_pos[2] - prev_pos[2]
                            pos_dist = np.sqrt(pos_dx**2 + pos_dy**2)
                            
                            cumulative_distance += pos_dist
                        
                        # Calculate speed based on cumulative distance
                        window_time = (position_history[-1][0] - position_history[0][0]) / original_fps
                        if window_time > 0:  # Avoid division by zero
                            cumulative_speed = cumulative_distance * PIXEL_TO_METER / window_time
            
                if enable_visualization:
                    if visualization_frame_count % VISUALIZATION_FRAME_SKIP == 0:
                        status_text = ""
                        if is_reflection_detected:
                            status_text = "Reflection detected"
                        elif is_outside_tank:
                            status_text = "Detection outside tank"
                        elif is_showing_previous:
                            status_text = "Using previous position"
                        
                        key = visualize_processing(
                            frame, masked_frame, enhanced, fg_mask, edges, fish_vis,
                            current_center, previous_center, distance, None,  # Pass None for instantaneous_speed
                            not is_showing_previous, is_outside_tank, is_showing_previous,
                            reflection_detected=is_reflection_detected, status_text=status_text,
                            cumulative_speed=cumulative_speed, window_size=speed_window_size/original_fps
                        )
                        if key == ord('q'):
                            break
                    visualization_frame_count += 1

                previous_center = current_center

                contour_areas = []
                for idx, contour in enumerate(valid_contours):
                    area = cv2.contourArea(contour)
                    contour_areas.append(area)
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        center_x = int(M["m10"] / M["m00"])
                        center_y = int(M["m01"] / M["m00"])
                        # Just use cumulative speed
                        write_center_data(center_writer, frame_count, idx, center_x, center_y, cumulative_speed)

                draw_fish_contours(enhanced, valid_contours, list(box_data.values()), 
                                  time_spent, distance_in_box, prev_box_positions, 
                                  original_fps, frame_count, contour_areas=contour_areas)

                pbar.update(1)
                frame_count += 1

            pbar.close()
            cap.release()
            if enable_visualization:
                cv2.destroyAllWindows()

            for i, (box_name, box_info) in enumerate(box_data.items()):
                box_info["time"] = time_spent[i]
                box_info["distance"] = distance_in_box[i]
                
                # Calculate average speed in box
                if time_spent[i] > 0:
                    avg_speed_in_box = distance_in_box[i] / time_spent[i]
                else:
                    avg_speed_in_box = 0
                
                data_writer.writerow([box_name, time_spent[i], distance_in_box[i], avg_speed_in_box])
    
    print(f"Processing complete for {video_filename}")
    return box_data, tank_points

def analyze_processed_data(output_dir):
    """
    Analyze all processed data in the output directory and create simplified CSV files.
    
    Args:
        output_dir (str): Directory containing processed data.
    """
    print("Analyzing processed data...")
    
    # Find all video subdirectories
    video_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    
    if not video_dirs:
        print("No processed video data found.")
        return

    # Process each video directory and save results in the video-specific folder
    for video_dir in video_dirs:
        video_path = os.path.join(output_dir, video_dir)
        video_name = video_dir
        
        print(f"Analyzing data for {video_name}...")
        
        # Create output files in the video's directory - just the summary and visits files
        summary_file = os.path.join(video_path, "summary_statistics.csv")
        visits_file = os.path.join(video_path, "box_visits.csv")
        
        # Find fish data file
        fish_data_file = None
        for file in os.listdir(video_path):
            if file.startswith("fish_data_") and file.endswith(".csv"):
                fish_data_file = os.path.join(video_path, file)
                break
        
        if not fish_data_file:
            print(f"No fish data found for {video_name}, skipping...")
            continue
        
        # Find fish coordinates file
        fish_coords_file = None
        for file in os.listdir(video_path):
            if file.startswith("fish_coords_") and file.endswith(".csv"):
                fish_coords_file = os.path.join(video_path, file)
                break
        
        if not fish_coords_file:
            print(f"No fish coordinates found for {video_name}, skipping visit analysis...")
        
        # Process fish data for this video
        box_data = {}
        total_time = 0
        total_distance = 0
        
        with open(fish_data_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            # Convert to numpy arrays for faster processing
            box_names = np.array([row["box_name"] for row in rows])
            time_spent = np.array([float(row["time_spent (s)"]) for row in rows])
            distance = np.array([float(row["distance_traveled (m)"]) for row in rows])
            speed = np.array([float(row["average_speed (m/s)"]) for row in rows])
            
            # Get unique box names
            unique_boxes = np.unique(box_names)
            
            for box_name in unique_boxes:
                # Use numpy mask to filter data for this box
                mask = box_names == box_name
                box_data[box_name] = {
                    "time_spent": np.sum(time_spent[mask]),
                    "distance": np.sum(distance[mask]),
                    "speed": np.mean(speed[mask])
                }
            
            # Calculate totals
            total_time = np.sum(time_spent)
            total_distance = np.sum(distance)
        
        # Calculate mean speed overall
        mean_speed_overall = total_distance / total_time if total_time > 0 else 0
        
        # Identify left, right, and central boxes
        left_box = None
        right_box = None
        central_box = None
        
        # Find box details file to determine box positions
        box_details_file = None
        for file in os.listdir(video_path):
            if file.startswith("box_details_") and file.endswith(".csv"):
                box_details_file = os.path.join(video_path, file)
                break
                
        if box_details_file:
            with open(box_details_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                boxes = list(reader)
                
                # Simple heuristic: sort boxes by x-coordinate and assign left/right/central
                if len(boxes) >= 3:
                    # Extract x-coordinate from the first point of each box
                    for box in boxes:
                        coords_str = box["coordinates"]
                        try:
                            # Assuming format like "[(x1,y1), (x2,y2), ...]"
                            coords = np.array(eval(coords_str))
                            # Calculate centroid using numpy mean
                            box["x_center"] = np.mean(coords[:, 0]) if coords.size > 0 else 0
                        except:
                            box["x_center"] = 0
                    
                    # Sort by x-coordinate
                    sorted_boxes = sorted(boxes, key=lambda b: b["x_center"])
                    
                    left_box = sorted_boxes[0]["box_name"]
                    central_box = sorted_boxes[1]["box_name"] if len(sorted_boxes) > 2 else None
                    right_box = sorted_boxes[-1]["box_name"]
                elif len(boxes) == 2:
                    # With only two boxes, assume left and right
                    left_box = boxes[0]["box_name"]
                    right_box = boxes[1]["box_name"]
                
                # Store box coordinates in box_data for use in is_point_in_box
                for box in boxes:
                    box_name = box["box_name"]
                    if box_name in box_data:
                        try:
                            box_data[box_name]["coords"] = eval(box["coordinates"])
                        except:
                            # If we can't parse the coordinates, create an empty list
                            box_data[box_name]["coords"] = []
        
        # Initialize visit counts
        left_visits = 0
        right_visits = 0
        current_box = None

        # Read coordinates to count visits - use fish_coords_file instead of fish_data_file
        if fish_coords_file and box_details_file:  # Only proceed if we have both files
            with open(fish_coords_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Access the correct column names from the coordinates file
                    center_x = float(row['center_x (px)'])
                    center_y = float(row['center_y (px)'])

                    # Determine which box the fish is in based on its coordinates
                    if left_box and "coords" in box_data[left_box] and is_point_in_box((center_x, center_y), box_data[left_box]["coords"]):
                        if current_box != left_box:
                            left_visits += 1
                            current_box = left_box
                    elif right_box and "coords" in box_data[right_box] and is_point_in_box((center_x, center_y), box_data[right_box]["coords"]):
                        if current_box != right_box:
                            right_visits += 1
                            current_box = right_box
                    else:
                        current_box = None  # Fish is not in any box

        # Create data in vertical format for summary statistics
        summary_data = [
            ["metric", "value"],
            ["video_name", video_name],
            ["cumulative_time_spent_total", total_time],
            ["cumulative_time_spent_left_box", box_data.get(left_box, {}).get("time_spent", 0) if left_box else 0],
            ["cumulative_time_spent_right_box", box_data.get(right_box, {}).get("time_spent", 0) if right_box else 0],
            ["cumulative_time_spent_central_region", box_data.get(central_box, {}).get("time_spent", 0) if central_box else 0],
            ["mean_speed_overall", mean_speed_overall],
            ["mean_speed_left_box", box_data.get(left_box, {}).get("speed", 0) if left_box else 0],
            ["mean_speed_right_box", box_data.get(right_box, {}).get("speed", 0) if right_box else 0],
            ["mean_speed_central_region", box_data.get(central_box, {}).get("speed", 0) if central_box else 0],
            ["overall_distance_travelled", total_distance],
            ["distance_travelled_left_box", box_data.get(left_box, {}).get("distance", 0) if left_box else 0],
            ["distance_travelled_right_box", box_data.get(right_box, {}).get("distance", 0) if right_box else 0],
            ["distance_travelled_central_region", box_data.get(central_box, {}).get("distance", 0) if central_box else 0],
            ["number_of_visits_left_box", left_visits],
            ["number_of_visits_right_box", right_visits]
        ]
        
        # Write summary statistics CSV in vertical format
        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(summary_data)
        
        print(f"Analysis for {video_name} complete. Results saved to:")
        print(f"  - {summary_file}")

    print("All video analysis complete!")

def is_point_in_box(point, box_coords):
    """
    Check if a point is inside a box defined by its coordinates.
    
    Args:
        point (tuple): The (x, y) coordinates of the point.
        box_coords (list): List of (x, y) tuples defining the box corners.
    
    Returns:
        bool: True if the point is inside the box, False otherwise.
    """
    # Convert to numpy arrays for better performance
    point = np.array(point)
    box_coords = np.array(box_coords)
    
    # Use numpy's cross product for point-in-polygon test
    n = len(box_coords)
    inside = False
    
    # Use numpy's vectorized operations for ray casting algorithm
    x, y = point
    
    j = n - 1
    for i in range(n):
        xi, yi = box_coords[i]
        xj, yj = box_coords[j]
        
        intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
        if intersect:
            inside = not inside
        j = i
    
    return inside

def batch_process_videos(video_files, output_dir, source_type, batch_size=5):
    """
    Process videos in batches.
    
    Args:
        video_files (list): List of video files to process.
        output_dir (str): Directory to save output files.
        source_type (str): Source type ('g' for Google Drive, 'o' for OneDrive, 'i' for internal).
        batch_size (int, optional): Number of videos to process in each batch. Defaults to 5.
        
    Returns:
        bool: True if processing was successful, False otherwise.
    """
    if not video_files:
        print("No video files to process.")
        return False
        
    print(f"Found {len(video_files)} videos to process.")
    
    temp_dir = os.path.join(output_dir, "temp_videos")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Get box data and tank points from the first video
    print("Processing first video to establish box and tank coordinates...")
    
    first_video = video_files[0]
    first_video_path = ""
    
    try:
        if source_type == 'g':
            success, first_video_path = download_from_google_drive(first_video['id'])
            if not success:
                print("Failed to download the first video.")
                return False
            # Move to temp directory
            new_path = os.path.join(temp_dir, os.path.basename(first_video_path))
            shutil.move(first_video_path, new_path)
            first_video_path = new_path
            print(f"Successfully downloaded: {os.path.basename(first_video_path)}")
        elif source_type == 'o':
            first_video_path = os.path.join(temp_dir, first_video['name'])
            if not download_from_onedrive(first_video['url'], first_video_path):
                return False
        else:  # internal
            first_video_path = first_video['path']
        
        # Process the first video and get box data and tank points
        enable_visualization = input("Enable visualization for the first video? (y/n): ").strip().lower() == 'y'
        box_data, tank_points = process_video(first_video_path, output_dir, enable_visualization=enable_visualization)
        
        if box_data is None or tank_points is None:
            print("Failed to process the first video. Aborting batch processing.")
            return False
        
        # Save box data and tank points for future reference
        config_data = {**box_data, "tank_coordinates": tank_points}
        config_file = os.path.join(output_dir, "batch_config.json")
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        # Clean up the first video if it was downloaded
        if source_type in ['g', 'o']:
            os.remove(first_video_path)
        
        # Process the remaining videos in batches of exactly 5 (or fewer for the last batch)
        remaining_videos = video_files[1:]
        total_batches = (len(remaining_videos) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            print(f"\nProcessing batch {batch_idx + 1} of {total_batches}...")
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(remaining_videos))
            batch_videos = remaining_videos[batch_start:batch_end]
            
            print(f"Downloading {len(batch_videos)} videos for this batch...")
            
            # Download batch videos if needed
            batch_video_paths = []
            for video in batch_videos:
                if source_type == 'g':
                    print(f"Downloading: {video['name']}...")
                    success, video_path = download_from_google_drive(video['id'])
                    if success:
                        # Move to temp directory
                        new_path = os.path.join(temp_dir, os.path.basename(video_path))
                        shutil.move(video_path, new_path)
                        batch_video_paths.append(new_path)
                        print(f"Successfully downloaded: {os.path.basename(new_path)}")
                    else:
                        print(f"Failed to download: {video['name']}")
                elif source_type == 'o':
                    video_path = os.path.join(temp_dir, video['name'])
                    if download_from_onedrive(video['url'], video_path):
                        batch_video_paths.append(video_path)
                else:  # internal
                    batch_video_paths.append(video['path'])
            
            # Process each video in the batch
            for video_path in batch_video_paths:
                print(f"Processing: {os.path.basename(video_path)}...")
                process_video(video_path, output_dir, box_data, tank_points, enable_visualization=False)
                
                # Clean up downloaded videos immediately after processing
                if source_type in ['g', 'o'] and os.path.exists(video_path):
                    os.remove(video_path)
                    print(f"Removed temporary file: {os.path.basename(video_path)}")
            
            print(f"Completed batch {batch_idx + 1} of {total_batches}")
    
    except Exception as e:
        print(f"Error during batch processing: {e}")
        return False
    finally:
        # Clean up temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print("Cleaned up temporary directory")
    
    return True

def extract_id_from_drive_link(link):
    """
    Extract the file or folder ID from a Google Drive link.
    
    Args:
        link (str): Google Drive link
        
    Returns:
        str: The extracted ID or None if not found
    """
    # Pattern for folder links
    folder_pattern = r'(?:https?://drive\.google\.com/(?:drive/folders/|file/d/|open\?id=))([a-zA-Z0-9_-]+)'
    
    match = re.search(folder_pattern, link)
    if match:
        return match.group(1)
    
    return None

def setup_google_credentials():
    """
    Set up Google Drive credentials with proper OAuth flow.
    
    Returns:
        bool: True if credentials are set up successfully, False otherwise
    """
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    import pickle
    
    # Define the scopes needed for Google Drive access
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    
    # Path to store the credentials
    token_path = os.path.join(os.path.expanduser('~'), '.fish_tracking_token.pickle')
    
    creds = None
    
    # Check if we have valid saved credentials
    if os.path.exists(token_path):
        try:
            with open(token_path, 'rb') as token:
                creds = pickle.load(token)
        except Exception as e:
            print(f"Error loading saved credentials: {e}")
    
    # If credentials don't exist or are invalid, get new ones
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"Error refreshing credentials: {e}")
                creds = None
        
        # If still no valid credentials, need to go through OAuth flow
        if not creds:
            print("\nNo valid Google Drive credentials found.")
            print("You need to set up credentials to access Google Drive.")
            print("Options:")
            print("1. Use OAuth 2.0 client credentials file")
            print("2. Enter client ID and secret manually")
            
            choice = input("Enter your choice (1/2): ").strip()
            
            try:
                if choice == '1':
                    credentials_path = input("Enter path to client credentials JSON file: ").strip()
                    flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
                elif choice == '2':
                    client_id = input("Enter client ID: ").strip()
                    client_secret = input("Enter client secret: ").strip()
                    
                    # Create a client config dictionary
                    client_config = {
                        "installed": {
                            "client_id": client_id,
                            "client_secret": client_secret,
                            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                            "token_uri": "https://oauth2.googleapis.com/token",
                            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                            "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob", "http://localhost"]
                        }
                    }
                    
                    flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
                else:
                    print("Invalid choice")
                    return False
                
                # Run the OAuth flow
                print("\nA browser window will open for you to authenticate with Google.")
                print("If no browser opens, check the console for a URL to visit.")
                creds = flow.run_local_server(port=0)
                
                # Save the credentials for future use
                with open(token_path, 'wb') as token:
                    pickle.dump(creds, token)
                
                print("Authentication successful! Credentials saved for future use.")
                
            except Exception as e:
                print(f"Error during authentication: {e}")
                return False
    
    # Store credentials in environment variable as JSON string
    # This is needed for compatibility with the existing code
    if creds:
        creds_data = {
            'token': creds.token,
            'refresh_token': creds.refresh_token,
            'token_uri': creds.token_uri,
            'client_id': creds.client_id,
            'client_secret': creds.client_secret,
            'scopes': creds.scopes
        }
        os.environ['GOOGLE_CREDENTIALS'] = json.dumps(creds_data)
        return True
    
    return False

def main():
    """
    Main function to execute fish tracking and analysis.
    """
    print("Fish Tracking System")
    print("===================")
    
    # Prompt for processing mode
    processing_mode = input("Select processing mode: (s)ingle video or (b)atch processing? (s/b): ").strip().lower()
    if processing_mode not in ['s', 'b']:
        print("Invalid selection. Exiting.")
        return
    
    # Prompt for file source
    file_source = input("Select file source: (g)oogle Drive, (o)neDrive, or (i)nternal files? (g/o/i): ").strip().lower()
    if file_source not in ['g', 'o', 'i']:
        print("Invalid selection. Exiting.")
        return
    
    # Set up output directory
    output_dir = input("Enter output directory path (default: ./Data): ").strip()
    if not output_dir:
        output_dir = "./Data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up Google credentials if needed
    if file_source == 'g' and not setup_google_credentials():
        print("Failed to set up Google credentials. Exiting.")
        return
    
    if processing_mode == 's':
        # Single video processing
        video_path = ""
        
        if file_source == 'g':
            drive_link = input("Enter Google Drive link to the video: ").strip()
            file_id = extract_id_from_drive_link(drive_link)
            
            if not file_id:
                print("Invalid Google Drive link. Exiting.")
                return
                
            success, video_path = download_from_google_drive(file_id)
            if not success:
                print("Failed to download video. Exiting.")
                return
            # Move the file to the output directory with a temp prefix
            video_name = os.path.basename(video_path)
            new_path = os.path.join(output_dir, "temp_" + video_name)
            shutil.move(video_path, new_path)
            video_path = new_path
        elif file_source == 'o':
            file_url = input("Enter OneDrive file URL: ").strip()
            video_name = input("Enter video filename (with extension): ").strip()
            video_path = os.path.join(output_dir, "temp_" + video_name)
            if not download_from_onedrive(file_url, video_path):
                print("Failed to download video. Exiting.")
                return
        else:  # internal
            video_path = input("Enter path to video file: ").strip()
        
        enable_visualization = input("Enable visualization? (y/n): ").strip().lower() == 'y'
        
        # Process the video
        process_video(video_path, output_dir, enable_visualization=enable_visualization)
        
        # Clean up downloaded video if needed
        if file_source in ['g', 'o'] and os.path.exists(video_path):
            os.remove(video_path)
        
        # Add this line to analyze the single video data
        analyze_processed_data(output_dir)
        
    else:  # Batch processing
        video_files = []
        
        if file_source == 'g':
            drive_link = input("Enter Google Drive link to the folder: ").strip()
            folder_id = extract_id_from_drive_link(drive_link)
            
            if not folder_id:
                print("Invalid Google Drive link. Exiting.")
                return
                
            video_files = get_video_files_from_google_drive(folder_id)
        elif file_source == 'o':
            folder_url = input("Enter OneDrive folder URL: ").strip()
            video_files = get_video_files_from_onedrive(folder_url)
        else:  # internal
            folder_path = input("Enter path to folder containing videos: ").strip()
            video_files = get_local_video_files(folder_path)
        
        if not video_files:
            print("No video files found. Exiting.")
            return
        
        print(f"Found {len(video_files)} video files.")
        
        # Process videos in batches
        if batch_process_videos(video_files, output_dir, file_source):
            # Analyze all processed data
            analyze_processed_data(output_dir)
        else:
            print("Batch processing failed.")

if __name__ == "__main__":
    main()