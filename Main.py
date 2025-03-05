import cv2 as cv2
import numpy as np
import os
import sys
import csv
from tqdm import tqdm
import json
import yaml
from box_manager import BoxManager

# Constants
MAX_DISTANCE_THRESHOLD = 150 
PIXEL_TO_METER = 0.000099  
VISUALIZATION_FRAME_SKIP = 5  


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

    Args:
        enhanced (np.array): Frame to draw contours on.
        contours (list): List of contours to draw.
        boxes (list): List of box dictionaries.
        time_spent (list): List to store time spent in each box.
        distance_in_box (list): List to store total distance traveled in each box.
        prev_box_positions (list): List to store previous positions in each box.
        original_fps (float): Original frames per second of the video.
        current_frame (int): Current frame number.
        contour_areas (list, optional): Pre-calculated contour areas. Defaults to None.
    """
    detected_boxes = [False] * len(boxes)
    current_box_positions = [None] * len(boxes)

    for i, contour in enumerate(contours):
        if contour.dtype != np.int32:
            contour = contour.astype(np.int32)

        area = contour_areas[i] if contour_areas else cv2.contourArea(contour)
        if area < 10 or area > 1350:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(enhanced, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.drawContours(enhanced, [contour], -1, (255, 255, 255), 1)

        M = cv2.moments(contour)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            current_center = (center_x, center_y)
            
            for j, box in enumerate(boxes):
                if is_contour_in_box(contour, box):
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

    for i, detected in enumerate(detected_boxes):
        if detected:
            time_spent[i] += 1 / original_fps
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

def main():
    """
    Main function to execute fish tracking and analysis.

    Sets up video processing, handles user input for box definition and visualization,
    processes each frame to detect fish, calculates time spent in boxes and speed,
    and saves the results to CSV files.
    """
    print("Starting video processing...")
    path = "/Users/manasvenkatasairavulapalli/Desktop/Research Work/ml/Vid/originals/"
    check_video_path(path)
    cap = initialize_video_capture(path)
    log_video_info(cap)

    enable_visualization = input("Enable visualization? (y/n): ").strip().lower() == 'y'

    box_manager = BoxManager()
    user_choice = input("Would you like to (d)raw boxes or provide (c)oordinates? (d/c): ").strip().lower()

    if user_choice == 'c':
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
                return
        box_data = box_manager.get_box_data()
    else:
        box_data = define_boxes(path)

    print("User-defined boxes:", box_data)

    video_filename = os.path.splitext(os.path.basename(path))[0]
    output_dir = f"/Users/manasvenkatasairavulapalli/Desktop/Research Work/ml/Data/{video_filename}"
    os.makedirs(output_dir, exist_ok=True)

    coord_filename = os.path.join(output_dir, f"box_details_{video_filename}.csv")
    with open(coord_filename, 'w', newline='') as coord_file:
        coord_writer = csv.writer(coord_file)
        coord_writer.writerow(["box_name", "coordinates"])
        for box_name, box_info in box_data.items():
            coord_writer.writerow([box_name, box_info["coords"]])

    data_filename = os.path.join(output_dir, f"fish_data_{video_filename}.csv")
    with open(data_filename, 'w', newline='') as data_file:
        data_writer = csv.writer(data_file)
        data_writer.writerow(["box_name", "time_spent (s)", "distance_traveled (m)", "average_speed (m/s)"])

        center_filename = os.path.join(output_dir, f"fish_coords_{video_filename}.csv")
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
            pbar = tqdm(total=max_frames, desc="Processing Video", unit="frame", dynamic_ncols=True)

            previous_center = None
            total_speed = 0
            speed_count = 0
            visualization_frame_count = 0

            position_history = []  # Will store (frame_num, x, y) tuples
            speed_window_size = int(original_fps)  # Use 1 second window (adjust as needed)

            ret, first_frame = cap.read()
            if not ret:
                print("Error: Cannot read the first frame.")
                sys.exit(1)

            tank_mask, tank_points = create_tank_mask(first_frame)
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

if __name__ == "__main__":
    main()