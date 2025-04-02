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
from filterpy.kalman import KalmanFilter
import math
import datetime
import traceback

# Constants
MAX_DISTANCE_THRESHOLD = 200 
PIXEL_TO_METER = 0.000099  
VISUALIZATION_FRAME_SKIP = 15 
RETINEX_SIGMA_LIST = [15, 80, 250]  # Multiple scales for MSR
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)
LAB_L_CLIP_LIMIT = 2.0
MIN_FISH_AREA = 25  # Reduce minimum area (from 100)
MAX_FISH_AREA = 500  # Also reduce maximum area (from 2000)
FISH_ASPECT_RATIO_RANGE = (0.3, 3.0)  # Expected fish shape ratio
MOVEMENT_THRESHOLD = 3  # Lower movement threshold (from 5)
HISTORY_LENGTH = 10  # Frames to keep in motion history

def download_from_google_drive(file_id, destination=None, max_retries=3):
    """
    Download a file from Google Drive with improved progress tracking and error handling.
    
    Args:
        file_id (str): The ID of the file to download.
        destination (str, optional): The path where the file will be saved.
                                    If None, will use the original filename.
        max_retries (int, optional): Maximum number of retry attempts. Defaults to 3.
    
    Returns:
        tuple: (bool, str, dict) - Success status, the path where the file was saved, and download stats.
    """
    stats = {
        "success": False,
        "attempts": 0,
        "error": None,
        "final_size": 0,
        "download_speed": 0
    }
    
    for attempt in range(max_retries):
        stats["attempts"] += 1
        start_time = time.time()
        
        try:
            # Load credentials from environment
            creds_json = os.environ.get('GOOGLE_CREDENTIALS')
            if not creds_json:
                stats["error"] = "Google credentials not found. Please run setup_google_credentials() first."
                print(stats["error"])
                return False, None, stats
                
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
            
            # Get file metadata to retrieve the original filename and file size
            file_metadata = service.files().get(fileId=file_id, fields="name,size").execute()
            original_filename = file_metadata.get('name', 'downloaded_file')
            file_size = int(file_metadata.get('size', 0))
            
            # If destination is not specified, use the original filename
            if destination is None:
                destination = os.path.join(os.getcwd(), original_filename)
            
            print(f"Downloading file: {original_filename} (Size: {file_size/(1024*1024):.2f} MB)")
            
            # Create a BytesIO object to store the downloaded file
            request = service.files().get_media(fileId=file_id)
            file_handle = io.BytesIO()
            downloader = MediaIoBaseDownload(file_handle, request)
            
            # Initialize the progress bar
            pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc=original_filename)
            
            # Download the file with progress tracking
            done = False
            last_progress = 0
            while not done:
                status, done = downloader.next_chunk()
                progress = int(status.progress() * 100)
                
                # Update the progress bar
                downloaded_size = file_size * status.progress()
                pbar.update(downloaded_size - pbar.n)  # Update progress bar with the new downloaded size
                
                # Only print if progress has changed significantly to avoid console spam
                if progress >= last_progress + 5 or progress == 100:
                    elapsed_time = time.time() - start_time
                    speed = downloaded_size / (1024 * 1024 * elapsed_time) if elapsed_time > 0 else 0
                    eta = (file_size - downloaded_size) / (speed * 1024 * 1024) if speed > 0 else 0
                    
                    print(f"Download progress: {progress}% - {downloaded_size/(1024*1024):.2f} MB / {file_size/(1024*1024):.2f} MB "
                          f"(Speed: {speed:.2f} MB/s, ETA: {eta:.1f} seconds)")
                    last_progress = progress
            
            pbar.close()  # Close the progress bar
            
            # Calculate final download stats
            total_time = time.time() - start_time
            stats["download_speed"] = file_size / (1024 * 1024 * total_time) if total_time > 0 else 0
            stats["final_size"] = file_size
            
            # Save the file to the destination
            with open(destination, 'wb') as f:
                f.write(file_handle.getvalue())
            
            print(f"Successfully downloaded {original_filename} to {destination} "
                  f"({file_size/(1024*1024):.2f} MB in {total_time:.1f} seconds, Avg speed: {stats['download_speed']:.2f} MB/s)")
            
            stats["success"] = True
            return True, destination, stats
        
        except Exception as e:
            error_message = f"Error downloading from Google Drive (attempt {attempt+1}/{max_retries}): {str(e)}"
            stats["error"] = str(e)
            print(error_message)
            
            if attempt < max_retries - 1:
                retry_delay = (attempt + 1) * 5  # Progressive backoff (5s, 10s, 15s...)
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Failed to download after {max_retries} attempts.")
                return False, None, stats
    
    return False, None, stats

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
    Get a list of video files from a Google Drive folder with enhanced metadata.
    
    Args:
        folder_id (str): The ID of the folder to search.
    
    Returns:
        list: A list of dictionaries containing file IDs, names, sizes, creation dates, and other metadata.
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
        
        print("Retrieving file list from Google Drive folder...")
        
        # Query for video files in the folder with additional metadata
        query = f"'{folder_id}' in parents and (mimeType contains 'video/') and trashed=false"
        fields = "nextPageToken, files(id, name, size, createdTime, modifiedTime, mimeType, md5Checksum)"
        
        # Use pagination to retrieve all files
        all_files = []
        page_token = None
        
        while True:
            results = service.files().list(
                q=query,
                spaces='drive',
                fields=fields,
                pageToken=page_token,
                pageSize=100
            ).execute()
            
            files = results.get('files', [])
            all_files.extend(files)
            
            # Get the page token for the next page of files
            page_token = results.get('nextPageToken')
            if not page_token:
                break
        
        # Format file sizes and dates for better readability
        for file in all_files:
            if 'size' in file:
                size_bytes = int(file['size'])
                file['size_bytes'] = size_bytes
                file['size_formatted'] = format_file_size(size_bytes)
            else:
                file['size_bytes'] = 0
                file['size_formatted'] = 'Unknown'
                
            if 'createdTime' in file:
                file['created_formatted'] = format_datetime(file['createdTime'])
            
            if 'modifiedTime' in file:
                file['modified_formatted'] = format_datetime(file['modifiedTime'])
        
        # Sort files by name for more consistent processing
        all_files.sort(key=lambda f: f.get('name', ''))
        
        if all_files:
            print(f"Found {len(all_files)} video files in Google Drive folder.")
            
            # Display a table with file information
            print("\nFile List:")
            print(f"{'Name':<40} {'Size':<10} {'Created':<20}")
            print("-" * 72)
            
            for file in all_files:
                print(f"{file.get('name', 'Unknown'):<40} "
                      f"{file.get('size_formatted', 'Unknown'):<10} "
                      f"{file.get('created_formatted', 'Unknown'):<20}")
            print()
        else:
            print("No video files found in the specified Google Drive folder.")
            
        return all_files
    except Exception as e:
        print(f"Error getting files from Google Drive: {e}")
        return []

def format_file_size(size_bytes):
    """Format file size in human-readable form."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes/1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes/(1024*1024):.1f} MB"
    else:
        return f"{size_bytes/(1024*1024*1024):.2f} GB"

def format_datetime(datetime_str):
    """Format datetime string from Google Drive API."""
    try:
        # Parse ISO 8601 format
        dt = datetime.datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
        # Format in a more readable way
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        return datetime_str

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

def apply_msrcr(img, sigma_list=RETINEX_SIGMA_LIST):
    """
    Apply Multi-Scale Retinex with Color Restoration (MSRCR).
    Helps enhance image details in varying illumination conditions.
    
    Args:
        img (np.array): Input BGR image
        sigma_list (list): List of Gaussian blur sigmas for multiple scales
        
    Returns:
        np.array: Enhanced image
    """
    img = img.astype(np.float32) / 255.0
    img = np.clip(img, 0.01, 1.0)
    
    # Convert to log domain
    img_log = np.log10(img)
    
    # Initialize retinex output
    retinex = np.zeros_like(img_log)
    
    # Multi-scale retinex
    for sigma in sigma_list:
        # Calculate Gaussian blur
        gaussian = cv2.GaussianBlur(img_log, (0, 0), sigma)
        # Calculate difference
        retinex += img_log - gaussian
    
    # Average over scales
    retinex = retinex / len(sigma_list)
    
    # Color restoration
    img_sum = np.sum(img, axis=2, keepdims=True)
    color_restoration = np.log10(img * 125.0 / (img_sum + 1e-6))
    
    # Combine MSR with color restoration
    msrcr = retinex * color_restoration
    
    # Normalize and convert back to linear domain
    msrcr = (msrcr - np.min(msrcr)) / (np.max(msrcr) - np.min(msrcr))
    msrcr = np.power(10, msrcr)
    
    # Final normalization and conversion to uint8
    msrcr = (msrcr * 255.0).astype(np.uint8)
    return msrcr

def enhance_lab_image(img):
    """
    Enhance image using LAB color space.
    LAB is better for separating luminance from color information.
    
    Args:
        img (np.array): Input BGR image
        
    Returns:
        np.array: Enhanced image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=LAB_L_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
    l_enhanced = clahe.apply(l)
    
    # Merge channels back
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return enhanced

def apply_adaptive_threshold(img, block_size=11, c=2):
    """
    Apply adaptive thresholding to handle varying illumination.
    
    Args:
        img (np.array): Grayscale input image
        block_size (int): Size of pixel neighborhood for thresholding
        c (int): Constant subtracted from mean
        
    Returns:
        np.array: Binary threshold image
    """
    # Ensure odd block size
    block_size = block_size if block_size % 2 == 1 else block_size + 1
    
    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c
    )
    return thresh

def detect_fish(enhanced, bg_subtractor, min_contour_area=MIN_FISH_AREA, max_contour_area=MAX_FISH_AREA, previous_contour=None, previous_center=None):
    """Enhanced fish detection focused on tracking just one fish"""
    # Apply background subtraction with shadow detection
    fg_mask = bg_subtractor.apply(enhanced, learningRate=0.005)
    
    # Remove shadows (typically marked as 127 in mask)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    
    # Noise reduction
    fg_mask = cv2.medianBlur(fg_mask, 5)
    
    # Morphological operations - make kernel size resolution-dependent
    kernel_size = 5 if min_contour_area < 150 else 7
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return [], None, 0.0
    
    # Score and filter contours
    scored_contours = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_contour_area < area < max_contour_area:
            # Get rotated rectangle
            rect = cv2.minAreaRect(cnt)
            width = rect[1][0]
            height = rect[1][1]
            
            # Aspect ratio check
            aspect_ratio = min(width, height) / max(width, height) if max(width, height) > 0 else 0
            
            # Calculate a confidence score based on multiple factors
            score = 0.0
            
            # Score based on aspect ratio (fish are usually slightly elongated)
            ar_score = 0.0
            if FISH_ASPECT_RATIO_RANGE[0] <= aspect_ratio <= FISH_ASPECT_RATIO_RANGE[1]:
                ar_target = 0.5  # Ideal aspect ratio for fish
                ar_score = 1.0 - min(abs(aspect_ratio - ar_target), 0.5) / 0.5
            
            # Convexity check
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            # Score for solidity (0.7-0.85 is ideal for fish)
            sol_score = 0.0
            if 0.5 < solidity < 0.95:
                sol_ideal = 0.75
                sol_score = 1.0 - min(abs(solidity - sol_ideal), 0.25) / 0.25
            
            # Perimeter and circularity
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Score for circularity (0.4-0.6 is ideal for fish)
            circ_score = 0.0
            if 0.2 < circularity < 0.8:
                circ_ideal = 0.5
                circ_score = 1.0 - min(abs(circularity - circ_ideal), 0.3) / 0.3
            
            # Area score - prefer mid-range areas
            area_range = max_contour_area - min_contour_area
            area_midpoint = min_contour_area + area_range / 2
            area_score = 1.0 - min(abs(area - area_midpoint), area_range/2) / (area_range/2)
            
            # Position consistency score (if we have previous position)
            pos_score = 0.0
            if previous_center is not None:
                # Get current center
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                    
                    # Calculate distance to previous center
                    dx = center_x - previous_center[0]
                    dy = center_y - previous_center[1]
                    distance = np.sqrt(dx**2 + dy**2)
                    
                    # Score based on distance (closer is better)
                    max_expected_movement = 100  # Maximum reasonable movement between frames
                    if distance <= max_expected_movement:
                        pos_score = 1.0 - (distance / max_expected_movement)
                    
            # Combined confidence score (weighted average)
            weight_ar = 0.2     # Aspect ratio weight
            weight_sol = 0.15   # Solidity weight
            weight_circ = 0.15  # Circularity weight
            weight_area = 0.2   # Area weight
            weight_pos = 0.3    # Position consistency weight
            
            # If we don't have position history, redistribute the weights
            if previous_center is None:
                weight_ar = 0.3
                weight_sol = 0.2
                weight_circ = 0.2
                weight_area = 0.3
                weight_pos = 0.0
                
            score = (ar_score * weight_ar + 
                     sol_score * weight_sol + 
                     circ_score * weight_circ + 
                     area_score * weight_area + 
                     pos_score * weight_pos)
            
            scored_contours.append((cnt, score, area))
    
    # Sort contours by score (highest first)
    scored_contours.sort(key=lambda x: x[1], reverse=True)
    
    # If we have valid contours, return the highest scored one
    if scored_contours:
        best_contour = scored_contours[0][0]
        best_score = scored_contours[0][1]
        best_area = scored_contours[0][2]
        return [best_contour], best_contour, best_score
    
    return [], None, 0.0

def preprocess_frame(frame, brightness_increase, clahe, scale_factor=0.5):
    """Enhanced preprocessing with better noise handling"""
    if scale_factor != 1.0:
        frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
    
    # Denoise first
    denoised = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
    
    # Apply retinex enhancement
    enhanced = apply_msrcr(denoised)
    
    # Enhance in LAB color space
    enhanced = enhance_lab_image(enhanced)
    
    # Convert to grayscale
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    
    # Local contrast enhancement
    clahe_enhanced = clahe.apply(gray)
    
    # Edge-preserving smoothing
    smoothed = cv2.bilateralFilter(clahe_enhanced, 9, 75, 75)
    
    return smoothed, scale_factor

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

def create_tank_mask(frame, points=None, scale_factor=1.0):
    """
    Create a binary mask for the tank area.

    If points are not provided, the user is prompted to define the tank area interactively
    on the first frame.

    Args:
        frame (np.array): Input video frame.
        points (list, optional): List of points defining the tank boundary. Defaults to None.
        scale_factor (float, optional): Scale factor for display and interactive selection.

    Returns:
        tuple: Binary mask where tank area is white (255) and outside area is black (0),
               and the list of points defining the mask.
    """
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if points is None:
        # For better visualization, resize large frames
        display_scale = 1.0 if scale_factor >= 1.0 else 1.0/scale_factor
        if display_scale != 1.0:
            display_frame = cv2.resize(frame, None, fx=display_scale, fy=display_scale)
        else:
            display_frame = frame.copy()
            
        cv2.namedWindow("Define Tank Area", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Define Tank Area", min(w, 1024), min(h, 768))
        
        points = []

        def click_and_crop(event, x, y, flags, param):
            nonlocal points, display_frame, display_scale
            
            if event == cv2.EVENT_LBUTTONDOWN:
                # Convert clicked point back to original frame coordinates
                original_x = int(x / display_scale) if display_scale != 1.0 else x
                original_y = int(y / display_scale) if display_scale != 1.0 else y
                
                points.append((original_x, original_y))
                
                # Draw on the display frame
                if len(points) > 1:
                    p1 = (int(points[-2][0] * display_scale), int(points[-2][1] * display_scale))
                    p2 = (int(points[-1][0] * display_scale), int(points[-1][1] * display_scale))
                    cv2.line(display_frame, p1, p2, (0, 255, 0), 2)
                    cv2.imshow("Define Tank Area", display_frame)
                    
                # Draw the point
                cv2.circle(display_frame, (x, y), 3, (0, 0, 255), -1)
                cv2.imshow("Define Tank Area", display_frame)

        cv2.setMouseCallback("Define Tank Area", click_and_crop)
        
        # Instructions
        instructions = "Click to define tank boundary. Press 'c' when complete."
        font_scale = 0.6 * display_scale
        cv2.putText(display_frame, instructions, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
        cv2.imshow("Define Tank Area", display_frame)

        while True:
            key = cv2.waitKey(1) & 0xFF

            if (key == ord("c") or key == ord("C")) and len(points) > 2:
                # Draw the closing line
                p1 = (int(points[0][0] * display_scale), int(points[0][1] * display_scale))
                p2 = (int(points[-1][0] * display_scale), int(points[-1][1] * display_scale))
                cv2.line(display_frame, p1, p2, (0, 255, 0), 2)
                cv2.imshow("Define Tank Area", display_frame)
                cv2.waitKey(500)  # Show completed polygon briefly
                break

        cv2.destroyWindow("Define Tank Area")

    # Create the mask using the points (original frame coordinates)
    if len(points) > 2:
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)

    return mask, points

def apply_tank_mask(frame, mask, scale_factor=1.0):
    """
    Apply tank mask to the frame.

    Args:
        frame (np.array): Input frame.
        mask (np.array): Tank mask.
        scale_factor (float, optional): Scaling factor for resolution adjustment.

    Returns:
        np.array: Frame with tank mask applied.
    """
    # Ensure mask is properly sized for the frame
    if frame.shape[:2] != mask.shape[:2]:
        h, w = frame.shape[:2]
        resized_mask = cv2.resize(mask, (w, h))
    else:
        resized_mask = mask
        
    return cv2.bitwise_and(frame, frame, mask=resized_mask)

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

    # Calculate pixel distance between current and previous position
    dx = current_center[0] - previous_center[0]
    dy = current_center[1] - previous_center[1]
    distance = np.sqrt(dx**2 + dy**2)

    # If movement is within threshold, consider it valid
    if distance <= max_distance:
        return True, current_center
    
    # For larger movements, apply additional validation
    # Using velocity consistency check if we have multiple previous positions
    if previous_valid_center and previous_center:
        # Calculate previous movement vector
        prev_dx = previous_center[0] - previous_valid_center[0]
        prev_dy = previous_center[1] - previous_valid_center[1]
        
        # Check if current movement is in similar direction (using dot product)
        if prev_dx * dx + prev_dy * dy > 0 and distance <= max_distance * 1.5:
            # Movement is in consistent direction, might be valid even if faster
            return True, current_center
    
    # Movement failed validation
    return False, previous_valid_center

def visualize_processing(original, masked, enhanced, fg_mask, edges, fish_vis, current_center, previous_center,
                       distance=None, instantaneous_speed=None, is_valid_movement=True,
                       is_outside_tank=False, is_showing_previous=False, reflection_detected=False, 
                       status_text=None, cumulative_speed=None, window_size=1.0, scale_factor=1.0, is_prediction=False):
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
        scale_factor (float, optional): Scale factor for display. Defaults to 1.0.
        is_prediction (bool, optional): Whether the current detection is a prediction. Defaults to False.

    Returns:
        int: Key pressed by user (for handling quit).
    """
    # Resize frames for display if needed
    display_scale = 1.0 if scale_factor >= 1.0 else 1.0/scale_factor
    
    if display_scale != 1.0:
        display_original = cv2.resize(original, None, fx=display_scale, fy=display_scale)
        display_masked = cv2.resize(masked, None, fx=display_scale, fy=display_scale)
        # Don't resize enhanced and binary images since they might already be at processing resolution
        display_fish_vis = cv2.resize(fish_vis, None, fx=display_scale, fy=display_scale)
    else:
        display_original = original
        display_masked = masked
        display_fish_vis = fish_vis
    
    cv2.imshow("Original", display_original)
    cv2.imshow("Masked", display_masked)
    cv2.imshow("Enhanced", enhanced)
    cv2.imshow("Background Mask", fg_mask)
    cv2.imshow("Edges", edges)

    if current_center and previous_center:
        # Draw trajectory line on fish_vis
        cv2.line(display_fish_vis, previous_center, current_center, (255, 255, 0), 2)

    if current_center:
        radius = int(5 * display_scale)  # Scale circle radius based on display scale
        thickness = int(max(1, 2 * display_scale))  # Scale line thickness
        font_scale = 0.7 * display_scale  # Scale font size
        
        if reflection_detected:
            cv2.circle(display_fish_vis, current_center, radius, (0, 255, 255), -1)
            cv2.putText(display_fish_vis, "Reflection detected", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
        elif is_outside_tank:
            cv2.circle(display_fish_vis, current_center, radius, (0, 0, 255), -1)
            cv2.putText(display_fish_vis, "Detection outside tank", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
        elif is_showing_previous:
            cv2.circle(display_fish_vis, current_center, radius, (0, 0, 255), -1)
            cv2.putText(display_fish_vis, "Using previous position", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
        else:
            cv2.circle(display_fish_vis, current_center, radius, (255, 0, 0), -1)

    # Only show cumulative speed (remove instantaneous speed display)
    if cumulative_speed is not None:
        speed_text = f"Avg Speed ({window_size:.1f}s): {cumulative_speed:.4f} m/s"
        cv2.putText(display_fish_vis, speed_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7 * display_scale, (0, 255, 0), 
                    int(max(1, 2 * display_scale)))
    
    if status_text:
        cv2.putText(display_fish_vis, status_text, (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7 * display_scale, (255, 255, 255), 
                    int(max(1, 2 * display_scale)))

    cv2.imshow("Detected Fish", display_fish_vis)

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

    # Get video dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate appropriate scale factor based on resolution
    # For HD (1920x1080) or higher, use 0.5 scale, otherwise use 1.0
    base_resolution = 1280 * 720  # 720p reference resolution
    current_resolution = frame_width * frame_height
    scale_factor = 0.5 if current_resolution > base_resolution else 1.0
    
    # Scale thresholds based on resolution ratio compared to base resolution
    resolution_factor = current_resolution / base_resolution if scale_factor == 1.0 else 1.0
    
    # Adjust area thresholds based on resolution
    min_contour_area = int(MIN_FISH_AREA * resolution_factor)
    max_contour_area = int(MAX_FISH_AREA * resolution_factor)
    
    # Adjust distance threshold based on resolution
    max_distance_threshold = int(MAX_DISTANCE_THRESHOLD * math.sqrt(resolution_factor))
    
    print(f"Using resolution factor: {resolution_factor:.2f}")
    print(f"Adjusted min contour area: {min_contour_area}")
    print(f"Adjusted max contour area: {max_contour_area}")
    print(f"Adjusted max distance threshold: {max_distance_threshold}")

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
                tank_mask, tank_points = create_tank_mask(first_frame, scale_factor=scale_factor)
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
                
                # Calculate display size based on video resolution and screen size
                # This will make windows proportionally sized and fit to screen
                display_width = min(frame_width, 800)  # Cap at 800 pixels for high-res videos
                display_height = int(display_width * frame_height / frame_width)
                
                cv2.resizeWindow("Original", display_width, display_height)
                cv2.resizeWindow("Masked", display_width, display_height)
                cv2.resizeWindow("Enhanced", display_width, display_height)
                cv2.resizeWindow("Background Mask", display_width, display_height)
                cv2.resizeWindow("Edges", display_width, display_height)
                cv2.resizeWindow("Detected Fish", display_width, display_height)

            previous_valid_center = None
            previous_valid_contour = None

            kalman_filter = initialize_kalman_filter(resolution_factor)
            kalman_initialized = False

            while True:
                ret, frame = cap.read()
                if not ret or frame_count >= max_frames:
                    break

                masked_frame = apply_tank_mask(frame, tank_mask, scale_factor)

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
                if contours:
                    # Find the single largest contour that meets the area criteria
                    filtered_contours = []
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if min_contour_area < area < max_contour_area:
                            filtered_contours.append((cnt, area))
                    
                    # If we found any valid contours, keep only the largest one
                    if filtered_contours:
                        largest_contour, largest_area = max(filtered_contours, key=lambda x: x[1])
                        valid_contours = [largest_contour]
                        if enable_visualization:
                            print(f"Frame {frame_count}: Largest contour area = {largest_area}")

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
                is_prediction = False  # Add this line to initialize the variable

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
                                    max_distance_threshold  # Use adjusted threshold
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
                        elif is_prediction:
                            status_text = "Using prediction"
                        
                        key = visualize_processing(
                            frame, masked_frame, enhanced, fg_mask, edges, fish_vis,
                            current_center, previous_center, distance, None,
                            not (is_showing_previous or is_prediction), is_outside_tank, 
                            is_showing_previous, reflection_detected=is_reflection_detected, 
                            status_text=status_text, cumulative_speed=cumulative_speed, 
                            window_size=speed_window_size/original_fps,
                            scale_factor=scale_factor,
                            is_prediction=is_prediction
                        )
                        if key == ord('q'):
                            break
                    visualization_frame_count += 1

                previous_center = current_center

                if current_center:
                    # Log only the main tracked fish (which is already the largest contour)
                    write_center_data(center_writer, frame_count, 0, current_center[0], current_center[1], cumulative_speed)
                
                # For visualization, we only need to pass the single contour:
                contour_area = cv2.contourArea(current_contour) if current_contour is not None else 0
                draw_fish_contours(enhanced, valid_contours, list(box_data.values()), 
                                  time_spent, distance_in_box, prev_box_positions, 
                                  original_fps, frame_count, contour_areas=[contour_area])

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
    """
    print("Analyzing processed data...")
    
    # Find all video subdirectories
    video_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    
    if not video_dirs:
        print("No processed video data found.")
        return

    # Store accuracy data for all videos
    accuracy_data = []
    
    for video_dir in video_dirs:
        video_path = os.path.join(output_dir, video_dir)
        video_name = video_dir
        
        print(f"Analyzing data for {video_name}...")
        
        # Create output files
        summary_file = os.path.join(video_path, "summary_statistics.csv")
        visits_file = os.path.join(video_path, "box_visits.csv")
        crossings_file = os.path.join(video_path, "box_crossings.csv")
        crossings_matrix_file = os.path.join(video_path, "box_crossings_matrix.csv")
        
        # Find required files
        fish_data_file = None
        fish_coords_file = None
        box_details_file = None
        
        for file in os.listdir(video_path):
            if file.startswith("fish_data_") and file.endswith(".csv"):
                fish_data_file = os.path.join(video_path, file)
            elif file.startswith("fish_coords_") and file.endswith(".csv"):
                fish_coords_file = os.path.join(video_path, file)
            elif file.startswith("box_details_") and file.endswith(".csv"):
                box_details_file = os.path.join(video_path, file)
        
        if not fish_data_file:
            print(f"No fish data found for {video_name}, skipping...")
            continue
            
        # Create case-insensitive box name mapping
        box_name_map = {}
        box_data = {}
        total_time = 0
        total_distance = 0
        
        with open(fish_data_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            # Create case-insensitive mapping for box names
            for row in rows:
                orig_name = row["box_name"]
                lower_name = orig_name.lower()
                box_name_map[lower_name] = orig_name
            
            # Convert to numpy arrays
            box_names = np.array([row["box_name"] for row in rows])
            time_spent = np.array([float(row["time_spent (s)"]) for row in rows])
            distance = np.array([float(row["distance_traveled (m)"]) for row in rows])
            speed = np.array([float(row["average_speed (m/s)"]) for row in rows])
            
            # Process data using original box names
            unique_boxes = np.unique(box_names)
            for box_name in unique_boxes:
                mask = box_names == box_name
                box_data[box_name] = {
                    "time_spent": np.sum(time_spent[mask]),
                    "distance": np.sum(distance[mask]),
                    "speed": np.mean(speed[mask])
                }
            
            total_time = np.sum(time_spent)
            total_distance = np.sum(distance)
        
        # Calculate accuracy percentage based on total time
        # Assuming ideal tracking time is 300 seconds (5 minutes)
        accuracy_percentage = math.ceil((total_time / 300) * 100)
        
        mean_speed_overall = total_distance / total_time if total_time > 0 else 0
        
        # Store accuracy information for later reporting
        accuracy_data.append({
            'video_name': video_name,
            'total_time': total_time,
            'accuracy_percentage': accuracy_percentage
        })
        
        # Process box positions
        left_box = None
        right_box = None
        central_box = None
        all_boxes = []
        
        if box_details_file:
            with open(box_details_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                boxes = list(reader)
                
                for box in boxes:
                    coords_str = box["coordinates"]
                    try:
                        coords = np.array(eval(coords_str))
                        box["x_center"] = np.mean(coords[:, 0]) if coords.size > 0 else 0
                        all_boxes.append(box["box_name"])
                    except:
                        box["x_center"] = 0
                
                if len(boxes) >= 3:
                    sorted_boxes = sorted(boxes, key=lambda b: b["x_center"])
                    
                    # Store original box names
                    left_box = sorted_boxes[0]["box_name"]
                    central_box = sorted_boxes[1]["box_name"] if len(sorted_boxes) > 2 else None
                    right_box = sorted_boxes[-1]["box_name"]
                    
                    # Store coordinates in box_data
                    for box in boxes:
                        box_name = box["box_name"]
                        if box_name in box_data:
                            try:
                                box_data[box_name]["coords"] = eval(box["coordinates"])
                            except:
                                box_data[box_name]["coords"] = []
        
        # Track box visits and crossings with enhanced detail
        box_visits = {box_name: 0 for box_name in all_boxes}
        crossings = []  # Will store all crossings between boxes
        current_box = None
        prev_frame = -1
        min_frames_in_box = 3  # Minimum frames required in a box to count as a visit
        frames_in_current_box = 0
        
        # Add trajectory tracking for more accurate box detection
        position_history = []  # Will store (frame, x, y) positions
        trajectory_window = 5  # Consider this many frames for trajectory analysis
        
        # Create a transition matrix to count movements between boxes
        transition_matrix = {from_box: {to_box: 0 for to_box in all_boxes} for from_box in all_boxes}
        
        if fish_coords_file and box_details_file:
            with open(fish_coords_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                # Get video FPS if available
                video_fps = 30  # Default assumption
                
                # Group by frame to handle multiple contours per frame
                frame_data = {}
                for row in rows:
                    frame = int(row['frame'])
                    if frame not in frame_data:
                        frame_data[frame] = []
                    frame_data[frame].append(row)
                
                # Process frame by frame
                for frame in sorted(frame_data.keys()):
                    rows_in_frame = frame_data[frame]
                    
                    # Find the largest contour in this frame (likely the fish)
                    largest_contour = max(rows_in_frame, key=lambda x: float(x.get('contour_area', 0)) 
                                         if 'contour_area' in x else 0)
                    
                    center_x = float(largest_contour['center_x (px)'])
                    center_y = float(largest_contour['center_y (px)'])
                    speed = float(largest_contour['speed (m/s)'])
                    
                    # Add to position history
                    position_history.append((frame, center_x, center_y))
                    # Keep only recent history
                    while len(position_history) > trajectory_window:
                        position_history.pop(0)
                    
                    # Use trajectory to determine current box more reliably
                    detected_box = determine_current_box_with_trajectory(
                        position_history, 
                        all_boxes, 
                        box_data
                    )
                    
                    # If the fish is in a box
                    if detected_box:
                        # If this is a different box than before, we have a transition
                        if current_box is not None and detected_box != current_box and frames_in_current_box >= min_frames_in_box:
                            # Record the crossing
                            crossing = {
                                'frame': frame,
                                'time': frame / video_fps,
                                'from_box': current_box,
                                'to_box': detected_box,
                                'speed': speed,
                                'position_x': center_x,
                                'position_y': center_y
                            }
                            crossings.append(crossing)
                            
                            # Update transition matrix
                            transition_matrix[current_box][detected_box] += 1
                            
                        # If this is a new box, count as a visit
                        if detected_box != current_box:
                            if frames_in_current_box >= min_frames_in_box:
                                # Only count as a visit if we spent enough frames in the previous box
                                box_visits[detected_box] += 1
                            frames_in_current_box = 1
                            current_box = detected_box
                        else:
                            frames_in_current_box += 1
                    else:
                        # Fish is outside any box
                        if current_box is not None and frames_in_current_box >= min_frames_in_box:
                            # Record leaving a box as a special crossing
                            crossing = {
                                'frame': frame,
                                'time': frame / video_fps,
                                'from_box': current_box,
                                'to_box': 'outside',
                                'speed': speed,
                                'position_x': center_x,
                                'position_y': center_y
                            }
                            crossings.append(crossing)
                        
                        current_box = None
                        frames_in_current_box = 0
                    
                    prev_frame = frame
        
        # Save crossing data to CSV
        with open(crossings_file, 'w', newline='') as f:
            fieldnames = ['frame', 'time', 'from_box', 'to_box', 'speed', 'position_x', 'position_y']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for crossing in crossings:
                writer.writerow(crossing)
        
        # Create a crossing matrix summary
        with open(crossings_matrix_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header row with destination boxes
            header = ['From / To'] + all_boxes + ['outside']
            writer.writerow(header)
            
            # Write transition counts for each source box
            for from_box in all_boxes:
                row = [from_box]  # First column is the source box
                
                # Add counts for transitions to each destination box
                for to_box in all_boxes:
                    row.append(transition_matrix.get(from_box, {}).get(to_box, 0))
                
                # Add count for transitions to outside
                outside_count = sum(1 for c in crossings if c['from_box'] == from_box and c['to_box'] == 'outside')
                row.append(outside_count)
                
                writer.writerow(row)
            
            # Add a summary row showing total transitions into each box
            total_row = ['Total Entries']
            for to_box in all_boxes:
                total_entries = sum(transition_matrix.get(from_box, {}).get(to_box, 0) for from_box in all_boxes)
                total_row.append(total_entries)
            
            # Add placeholder for total exits to "outside" 
            total_row.append('-')
            writer.writerow(total_row)
        
        # Create summary statistics
        summary_data = [
            ["metric", "value"],
            ["video_name", video_name],
            ["cumulative_time_spent_total", total_time],
            ["accuracy_percentage", accuracy_percentage],  # Add the accuracy percentage
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
            ["number_of_visits_left_box", box_visits.get(left_box, 0) if left_box else 0],
            ["number_of_visits_right_box", box_visits.get(right_box, 0) if right_box else 0],
            ["number_of_crossings_left_to_right", transition_matrix.get(left_box, {}).get(right_box, 0) if left_box and right_box else 0],
            ["number_of_crossings_right_to_left", transition_matrix.get(right_box, {}).get(left_box, 0) if left_box and right_box else 0],
            ["total_box_crossings", sum(crossing.get('to_box') != 'outside' for crossing in crossings)]
        ]
        
        # Add additional crossing metrics to summary
        for from_box in all_boxes:
            for to_box in all_boxes:
                if from_box != to_box:
                    crossing_key = f"crossings_{from_box.replace(' ', '_').lower()}_to_{to_box.replace(' ', '_').lower()}"
                    crossing_value = transition_matrix.get(from_box, {}).get(to_box, 0)
                    summary_data.append([crossing_key, crossing_value])
        
        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(summary_data)
        
        # Save detailed visits data
        with open(visits_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["box_name", "visits"])
            for box_name, visits in box_visits.items():
                writer.writerow([box_name, visits])
        
        print(f"Analysis for {video_name} complete. Results saved to:")
        print(f"  - {summary_file}")
        print(f"  - {visits_file}")
        print(f"  - {crossings_file}")
        print(f"  - {crossings_matrix_file}")  # Add this line

    # Print accuracy report after processing all videos
    print("\n===== VIDEO TRACKING ACCURACY REPORT =====")
    print(f"{'Video Name':<40} {'Total Time (s)':<15} {'Accuracy %':<10} {'Status'}")
    print("=" * 80)
    
    # Sort by accuracy percentage (descending)
    accuracy_data.sort(key=lambda x: x['accuracy_percentage'], reverse=True)
    
    high_accuracy_videos = []
    low_accuracy_videos = []
    
    for video in accuracy_data:
        # Format status based on accuracy threshold
        if video['accuracy_percentage'] >= 90:
            status = "GOOD"
            high_accuracy_videos.append(video['video_name'])
        else:
            status = "NEEDS ATTENTION"
            low_accuracy_videos.append(video['video_name'])
        
        # Print row with appropriate formatting
        print(f"{video['video_name']:<40} {video['total_time']:<15.2f} {video['accuracy_percentage']:<10} {status}")
    
    print("\n===== HIGH ACCURACY VIDEOS (≥90%) =====")
    for video in high_accuracy_videos:
        print(f"- {video}")
    
    print("\n===== VIDEOS REQUIRING ATTENTION (<90%) =====")
    if low_accuracy_videos:
        for video in low_accuracy_videos:
            print(f"- {video}")
    else:
        print("All videos have good accuracy!")
    
    print("\nAll video analysis complete!")

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
    Process videos in batches with enhanced download tracking and error handling.
    
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
    
    # Create directories
    temp_dir = os.path.join(output_dir, "temp_videos")
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create download status tracking
    download_status = {}
    failed_downloads = []
    
    # Create a log file for this batch process
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"batch_process_{timestamp}.log")
    download_log = os.path.join(log_dir, f"downloads_{timestamp}.csv")
    
    # Create CSV file for download stats
    with open(download_log, 'w') as f:
        f.write("file_name,file_id,status,size_bytes,download_time_seconds,speed_mbps,attempts,error\n")
    
    # Initialize logging
    def log_message(message, also_print=True):
        with open(log_file, 'a') as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {message}\n")
        if also_print:
            print(message)
    
    log_message(f"Starting batch processing of {len(video_files)} videos from {source_type} source.")
    
    # Get box data and tank points from the first video
    log_message("Processing first video to establish box and tank coordinates...")
    
    first_video = video_files[0]
    first_video_path = ""
    
    try:
        if source_type == 'g':
            log_message(f"Downloading first video: {first_video['name']} (ID: {first_video['id']})")
            success, first_video_path, stats = download_from_google_drive(first_video['id'])
            
            # Log download status
            download_status[first_video['id']] = {
                'name': first_video['name'],
                'success': success,
                'stats': stats
            }
            
            if not success:
                log_message("Failed to download the first video.", True)
                failed_downloads.append(first_video)
                return False
                
            # Move to temp directory
            new_path = os.path.join(temp_dir, os.path.basename(first_video_path))
            shutil.move(first_video_path, new_path)
            first_video_path = new_path
            log_message(f"Successfully downloaded: {os.path.basename(first_video_path)}")
            
            # Log download stats to CSV
            with open(download_log, 'a') as f:
                f.write(f"{first_video['name']},{first_video['id']},success,{stats.get('final_size', 0)},"
                        f"{time.time() - stats.get('start_time', time.time())},{stats.get('download_speed', 0)},"
                        f"{stats.get('attempts', 1)},{stats.get('error', '')}\n")
                
        elif source_type == 'o':
            first_video_path = os.path.join(temp_dir, first_video['name'])
            if not download_from_onedrive(first_video['url'], first_video_path):
                log_message("Failed to download the first video from OneDrive.", True)
                return False
        else:  # internal
            first_video_path = first_video['path']
        
        # Process the first video and get box data and tank points
        enable_visualization = input("Enable visualization for the first video? (y/n): ").strip().lower() == 'y'
        box_data, tank_points = process_video(first_video_path, output_dir, enable_visualization=enable_visualization)
        
        if box_data is None or tank_points is None:
            log_message("Failed to process the first video. Aborting batch processing.", True)
            return False
        
        # Save box data and tank points for future reference
        config_data = {**box_data, "tank_coordinates": tank_points}
        config_file = os.path.join(output_dir, "batch_config.json")
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        log_message(f"Created configuration file: {config_file}")
        
        # Clean up the first video if it was downloaded
        if source_type in ['g', 'o']:
            os.remove(first_video_path)
            log_message(f"Removed temporary file: {first_video_path}")
        
        # Process the remaining videos in batches
        remaining_videos = video_files[1:]
        total_batches = (len(remaining_videos) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            log_message(f"\nProcessing batch {batch_idx + 1} of {total_batches}...")
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(remaining_videos))
            batch_videos = remaining_videos[batch_start:batch_end]
            
            log_message(f"Downloading {len(batch_videos)} videos for this batch...")
            
            # Download batch videos if needed
            batch_video_paths = []
            for video in batch_videos:
                if source_type == 'g':
                    log_message(f"Downloading: {video['name']} (ID: {video['id']})...")
                    success, video_path, stats = download_from_google_drive(video['id'])
                    
                    # Log download status
                    download_status[video['id']] = {
                        'name': video['name'],
                        'success': success,
                        'stats': stats
                    }
                    
                    # Log download stats to CSV
                    with open(download_log, 'a') as f:
                        download_time = stats.get('download_time', time.time() - stats.get('start_time', time.time()))
                        f.write(f"{video['name']},{video['id']},{['failed', 'success'][success]},"
                                f"{stats.get('final_size', 0)},{download_time},{stats.get('download_speed', 0)},"
                                f"{stats.get('attempts', 1)},\"{stats.get('error', '')}\"\n")
                    
                    if success:
                        # Move to temp directory
                        new_path = os.path.join(temp_dir, os.path.basename(video_path))
                        shutil.move(video_path, new_path)
                        batch_video_paths.append(new_path)
                        log_message(f"Successfully downloaded: {os.path.basename(new_path)}")
                    else:
                        log_message(f"Failed to download: {video['name']}")
                        failed_downloads.append(video)
                        
                elif source_type == 'o':
                    video_path = os.path.join(temp_dir, video['name'])
                    if download_from_onedrive(video['url'], video_path):
                        batch_video_paths.append(video_path)
                    else:
                        log_message(f"Failed to download: {video['name']} from OneDrive")
                else:  # internal
                    batch_video_paths.append(video['path'])
            
            # Process the successfully downloaded videos in this batch
            if batch_video_paths:
                log_message(f"Processing {len(batch_video_paths)} successfully downloaded videos...")
                
                for video_path in batch_video_paths:
                    log_message(f"Processing video: {os.path.basename(video_path)}")
                    
                    try:
                        # Process the video with the box data and tank points from the first video
                        success, _ = process_video(video_path, output_dir, box_data=box_data, tank_points=tank_points)
                        
                        if success:
                            log_message(f"Successfully processed: {os.path.basename(video_path)}")
                        else:
                            log_message(f"Failed to process: {os.path.basename(video_path)}")
                    except Exception as e:
                        log_message(f"Error processing {os.path.basename(video_path)}: {str(e)}")
                    
                    # Clean up the video file
                    if source_type in ['g', 'o'] and os.path.exists(video_path):
                        os.remove(video_path)
                        log_message(f"Removed temporary file: {video_path}")
            else:
                log_message("No videos were successfully downloaded in this batch.")
        
        # Summarize download results
        log_message("\nDownload Summary:")
        log_message(f"Total files: {len(video_files)}")
        log_message(f"Successfully downloaded: {len(video_files) - len(failed_downloads)}")
        log_message(f"Failed downloads: {len(failed_downloads)}")
        
        if failed_downloads:
            log_message("\nFailed Downloads:")
            for video in failed_downloads:
                error = download_status.get(video['id'], {}).get('stats', {}).get('error', 'Unknown error')
                log_message(f"- {video['name']} (ID: {video['id']}): {error}")
            
            # Save failed downloads to a file for potential retry
            failed_file = os.path.join(log_dir, f"failed_downloads_{timestamp}.json")
            with open(failed_file, 'w') as f:
                json.dump(failed_downloads, f, indent=2)
            log_message(f"Failed downloads saved to: {failed_file}")
            
            # Ask if user wants to retry failed downloads
            if len(failed_downloads) > 0:
                retry = input("\nDo you want to retry failed downloads? (y/n): ").strip().lower() == 'y'
                if retry:
                    log_message("Retrying failed downloads...")
                    return batch_process_videos(failed_downloads, output_dir, source_type, batch_size)
        
        return True
    except Exception as e:
        log_message(f"Error in batch processing: {str(e)}", True)
        traceback.print_exc()
        return False

def extract_id_from_drive_link(link):
    """
    Extract the file or folder ID from a Google Drive link.
    
    Args:
        link (str): Google Drive link
        
    Returns:
        str: The extracted ID or None if not found
    """
    # Patterns for various Google Drive URL formats
    patterns = [
        # Standard formats
        r'(?:https?://drive\.google\.com/(?:drive/folders/|file/d/|open\?id=))([a-zA-Z0-9_-]+)',
        # User-specific or shared links
        r'(?:https?://drive\.google\.com/drive/u/\d+/folders/)([a-zA-Z0-9_-]+)',
        # File with view parameter
        r'(?:https?://drive\.google\.com/file/d/)([a-zA-Z0-9_-]+)(?:/view)',
        # Shared links formats
        r'(?:https?://drive\.google\.com/drive/shared-with-me/)([a-zA-Z0-9_-]+)',
        # Direct share link
        r'(?:https?://drive\.google\.com/drive/)([a-zA-Z0-9_-]+)',
        # Just the ID itself (for cases where user might paste just the ID)
        r'^([a-zA-Z0-9_-]{28,})$'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, link)
        if match:
            return match.group(1)
    
    # For shared links that include parameters
    if 'id=' in link:
        param_match = re.search(r'id=([a-zA-Z0-9_-]+)', link)
        if param_match:
            return param_match.group(1)
    
    print(f"Could not extract ID from link: {link}")
    print("Valid formats include:")
    print("- https://drive.google.com/drive/folders/FOLDER_ID")
    print("- https://drive.google.com/file/d/FILE_ID")
    print("- https://drive.google.com/file/d/FILE_ID/view")
    print("- https://drive.google.com/drive/u/0/folders/FOLDER_ID")
    
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

def determine_current_box_with_trajectory(position_history, all_boxes, box_data):
    """
    Use trajectory information to more reliably determine which box the fish is in.
    
    Args:
        position_history (list): List of (frame, x, y) tuples for recent positions
        all_boxes (list): List of box names
        box_data (dict): Dictionary containing box information
        
    Returns:
        str: Name of the detected box, or None if fish is not in any box
    """
    if len(position_history) < 2:
        # Not enough trajectory data, use simple point-in-box check
        current_pos = position_history[-1]
        center_x, center_y = current_pos[1], current_pos[2]
        
        for box_name in all_boxes:
            if box_name in box_data and "coords" in box_data[box_name]:
                if is_point_in_box((center_x, center_y), box_data[box_name]["coords"]):
                    return box_name
        return None
    
    # Get current position and direction
    current_pos = position_history[-1]
    prev_pos = position_history[-2]
    
    center_x, center_y = current_pos[1], current_pos[2]
    prev_x, prev_y = prev_pos[1], prev_pos[2]
    
    # Calculate direction vector
    dx = center_x - prev_x
    dy = center_y - prev_y
    
    # Check current position first
    current_box = None
    for box_name in all_boxes:
        if box_name in box_data and "coords" in box_data[box_name]:
            if is_point_in_box((center_x, center_y), box_data[box_name]["coords"]):
                current_box = box_name
                break
    
    # If we're in a box, return it
    if current_box:
        return current_box
    
    # For fast-moving fish that might "skip" a box between frames,
    # check if the trajectory passes through any box
    if abs(dx) > 20 or abs(dy) > 20:  # Only for significant movements
        # Create a few interpolated points along the trajectory
        points_to_check = 5
        for i in range(1, points_to_check):
            # Check points along the trajectory
            check_x = prev_x + (dx * i / points_to_check)
            check_y = prev_y + (dy * i / points_to_check)
            
            for box_name in all_boxes:
                if box_name in box_data and "coords" in box_data[box_name]:
                    if is_point_in_box((check_x, check_y), box_data[box_name]["coords"]):
                        return box_name
    
    return None

def initialize_kalman_filter(resolution_factor=1.0):
    """
    Initialize a Kalman filter for tracking fish movement.
    
    Args:
        resolution_factor (float): Factor to adjust parameters based on video resolution
    
    Returns:
        KalmanFilter: Configured Kalman filter object
    """
    kf = KalmanFilter(dim_x=4, dim_z=2)  # State: [x, y, dx, dy], Measurement: [x, y]
    
    # State transition matrix (physics model)
    kf.F = np.array([
        [1, 0, 1, 0],  # x = x + dx
        [0, 1, 0, 1],  # y = y + dy
        [0, 0, 1, 0],  # dx = dx
        [0, 0, 0, 1]   # dy = dy
    ])
    
    # Measurement function (we only measure position, not velocity)
    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    
    # Measurement noise - adjust based on resolution
    meas_noise = 10 * resolution_factor
    kf.R = np.array([
        [meas_noise, 0],
        [0, meas_noise]
    ])
    
    # Process noise - adjust based on resolution
    process_noise = 0.1 * resolution_factor
    kf.Q = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 10, 0],  # Higher uncertainty for velocity
        [0, 0, 0, 10]
    ]) * process_noise
    
    # Initial state uncertainty - adjust based on resolution
    init_uncertainty = 100 * resolution_factor
    kf.P = np.array([
        [init_uncertainty, 0, 0, 0],
        [0, init_uncertainty, 0, 0],
        [0, 0, init_uncertainty, 0],
        [0, 0, 0, init_uncertainty]
    ])
    
    return kf

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