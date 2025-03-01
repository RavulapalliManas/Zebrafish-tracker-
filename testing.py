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
VISUALIZATION_FRAME_SKIP = 5  # Increase to skip more frames for faster visualization


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
        slowed_fps (int, optional): Slowed frames per second for processing (not currently used). Defaults to 10.
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

    Includes resizing, grayscale conversion, brightness adjustment, Gaussian blur, and CLAHE enhancement.

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
    enhanced = clahe.apply(blurred)

    return enhanced, scale_factor

def detect_fish(enhanced, bg_subtractor, min_contour_area=10, max_contour_area=1350):
    """
    Detect the largest fish in the given frame using Haar cascades.

    Args:
        enhanced (np.array): Preprocessed grayscale frame.
        bg_subtractor (cv2.BackgroundSubtractorKNN): Background subtractor object (not used in Haar cascade method).
        min_contour_area (int, optional): Minimum contour area (not used in Haar cascade method). Defaults to 10.
        max_contour_area (int, optional): Maximum contour area (not used in Haar cascade method). Defaults to 1350.

    Returns:
        list: List of fish detections (rectangles) if found, otherwise empty list.
    """
    # IMPORTANT: You MUST replace 'haarcascade_frontalface_default.xml' with the *correct path*
    # to your Haar cascade file.  The error "Could not load cascade classifier" means
    # OpenCV *cannot find the file at the path you provided*.

    # 1.  Find the 'haarcascade_frontalface_default.xml' file on your computer.
    #     It's often located within your OpenCV installation directory.
    #     (See previous detailed instructions on how to find it based on your OS and install method).

    # 2.  Once you find the file, copy its *full path*.
    #     Example (this is just an example, your path will be different):
    #     macOS: '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
    #     Windows: 'C:\Users\YourUsername\Anaconda3\envs\myenv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml'

    # 3.  Paste the full path into the cv2.CascadeClassifier() function, replacing the current path.
    #     Make sure to use raw strings (r'...') for Windows paths to avoid issues with backslashes.

    cascade_path = 'haarcascade_frontalface_default.xml' # <-- REPLACE THIS WITH THE CORRECT PATH
    if not os.path.exists(cascade_path):
        print(f"Error: Cascade file not found at path: {cascade_path}")
        print("Please check that the path to 'haarcascade_frontalface_default.xml' is correct.")
        print("Refer to the comments in the detect_fish function for detailed instructions.")
        return []

    fish_cascade = cv2.CascadeClassifier(cascade_path)

    if fish_cascade.empty():
        print("Error: Could not load cascade classifier.") # This error is now likely due to a corrupted file, not path
        return []

    fishes = fish_cascade.detectMultiScale(
        enhanced,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    detected_contours = []
    for (x, y, w, h) in fishes:
        # Haar cascade returns rectangles, convert to contours for consistency if needed
        contour = np.array([[[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]], dtype=np.int32)
        detected_contours.append(contour)

    return detected_contours

def process_frame(frame, bg_subtractor, clahe, brightness_increase, scale_factor):
    """
    Processes a single frame to detect fish and its center using Haar cascades.

    Args:
        frame (np.array): Input video frame.
        bg_subtractor (cv2.BackgroundSubtractorKNN): Background subtractor object (not used anymore).
        clahe (cv2.CLAHE): CLAHE object for contrast enhancement.
        brightness_increase (int): Brightness increase value.
        scale_factor (float): Frame scaling factor.

    Returns:
        tuple: Enhanced frame, detected contours, and center of the largest contour if found, otherwise None for center.
    """
    enhanced, _ = preprocess_frame(frame, brightness_increase, clahe, scale_factor)
    contours = detect_fish(enhanced, None)

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

def draw_fish_contours(enhanced, contours, boxes, time_spent, original_fps, contour_areas=None):
    """
    Draws contours on the frame and updates time spent in each box.

    Args:
        enhanced (np.array): Frame to draw contours on.
        contours (list): List of contours to draw.
        boxes (list): List of box dictionaries.
        time_spent (list): List to store time spent in each box.
        original_fps (float): Original frames per second of the video.
        contour_areas (list, optional): Pre-calculated contour areas. Defaults to None.
    """
    detected_boxes = [False] * len(boxes)

    for i, contour in enumerate(contours):
        if contour.dtype != np.int32:
            contour = contour.astype(np.int32)

        area = contour_areas[i] if contour_areas else cv2.contourArea(contour)
        if area < 10 or area > 1350:
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

    Currently, only 'q' key is handled to quit the program.

    Args:
        key (int): Key code of the pressed key.

    Returns:
        bool: True if a handled key was pressed, False otherwise. Currently always returns False.
    """
    if key == ord('q'):
        print("Quit key pressed. Exiting...")
        sys.exit()
    return False

def write_center_data(center_writer, frame_count, idx, center_x, center_y, instantaneous_speed):
    """
    Write the center data to the CSV file.

    Args:
        center_writer (csv.writer): CSV writer object for the center file.
        frame_count (int): Current frame number.
        idx (int): Contour index.
        center_x (int): X-coordinate of the center.
        center_y (int): Y-coordinate of the center.
        instantaneous_speed (float): Calculated instantaneous speed in m/s.
    """
    center_writer.writerow([frame_count, idx, center_x, center_y, instantaneous_speed])

def create_tank_mask(frame, points=None):
    """
    Create a binary mask for the tank area.

    If points are not provided, the user is prompted to define the tank area interactively
    on the first frame.

    Args:
        frame (np.array): Input video frame.
        points (list, optional): List of points defining the tank boundary. If None, user must select points. Defaults to None.

    Returns:
        tuple: Binary mask where tank area is white (255) and outside area is black (0), and the list of points defining the mask.
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
    """Apply tank mask to the frame.

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

    If the movement distance exceeds max_distance, the movement is considered invalid,
    and the function returns the previous valid center.

    Args:
        current_center (tuple): Current detected position (x, y).
        previous_center (tuple): Position from the previous frame (x, y).
        previous_valid_center (tuple): Last known valid position (x, y).
        max_distance (int): Maximum allowed movement distance in pixels.

    Returns:
        tuple: Tuple of (is_valid, center_to_use), where is_valid is a boolean indicating
               if the movement is valid, and center_to_use is the center to be used
               (current_center if valid, previous_valid_center if invalid).
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
                       is_outside_tank=False, is_showing_previous=False):
    """
    Visualize the processing steps in separate windows.

    Displays original frame, masked frame, enhanced frame, background subtraction mask,
    edge detection result, and fish visualization with contours, boxes, and tracking information.

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

    Returns:
        int: Key pressed by user (for handling quit).
    """
    cv2.imshow("Original", original)
    cv2.imshow("Masked", masked)
    cv2.imshow("Enhanced", enhanced)
    # Only display fg_mask and edges if they are not None and valid images
    # if fg_mask is not None: # No longer needed as fg_mask and edges are not used.
    #     cv2.imshow("Background Mask", fg_mask)
    # if edges is not None:
    #     cv2.imshow("Edges", edges)

    if current_center and previous_center:
        cv2.line(fish_vis, previous_center, current_center, (255, 255, 0), 2)

    if current_center:
        if is_outside_tank:
            cv2.circle(fish_vis, current_center, 8, (0, 0, 255), -1)
            cv2.putText(fish_vis, "Detection outside tank", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif is_showing_previous:
            cv2.circle(fish_vis, current_center, 8, (0, 0, 255), -1)
            cv2.putText(fish_vis, "Using previous position", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.circle(fish_vis, current_center, 5, (255, 0, 0), -1)

    if instantaneous_speed is not None:
        speed_text = f"Speed: {instantaneous_speed:.4f} m/s"
        cv2.putText(fish_vis, speed_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Detected Fish", fish_vis)

    return cv2.waitKey(1) & 0xFF

def main():
    """
    Main function to execute fish tracking and analysis.

    Sets up video processing, handles user input for box definition and visualization,
    processes each frame to detect fish, calculates time spent in boxes and speed,
    and saves the results to CSV files.
    """
    print("Starting video processing...")
    path = "/Users/manasvenkatasairavulapalli/Desktop/Research Work/ml/Vid/originals/n1.mov"
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
        data_writer.writerow(["box_name", "time_spent (s)", "average_speed (m/s)"])

        center_filename = os.path.join(output_dir, f"fish_coords_{video_filename}.csv")
        with open(center_filename, 'w', newline='') as center_file:
            center_writer = csv.writer(center_file)
            center_writer.writerow(["frame", "contour_id", "center_x (px)", "center_y (px)", "instantaneous_speed (m/s)"])

            frame_skip = 1
            scale_factor = 1.0
            brightness_increase = 39
            contrast_clip_limit = 0.85
            min_contour_area = 15

            clahe = cv2.createCLAHE(clipLimit=contrast_clip_limit, tileGridSize=(8,8))

            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            time_spent = [0] * len(box_data)

            max_frames = min(9000, total_frames)
            pbar = tqdm(total=max_frames, desc="Processing Video", unit="frame", dynamic_ncols=True)

            previous_center = None
            total_speed = 0
            speed_count = 0
            visualization_frame_count = 0 # Counter for visualization frame skipping

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
                enhanced = clahe.apply(blurred)

                contours = detect_fish(enhanced, None)

                fish_vis = None
                if enable_visualization:
                    fish_vis = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                    cv2.drawContours(fish_vis, contours, -1, (0, 255, 0), 2)

                    for box in box_data.values():
                        pts = np.array(box["coords"], dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(fish_vis, [pts], True, (0, 0, 255), 2)

                current_center = None
                is_outside_tank = False
                is_showing_previous = False

                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        center_x = int(M["m10"] / M["m00"])
                        center_y = int(M["m01"] / M["m00"])
                        current_center = (center_x, center_y)

                        if center_y < tank_mask.shape[0] and center_x < tank_mask.shape[1] and tank_mask[center_y, center_x] > 0:
                            is_valid_movement, center_to_use = is_fish_movement_valid(
                                current_center,
                                previous_center,
                                previous_valid_center if previous_valid_center else current_center,
                                MAX_DISTANCE_THRESHOLD
                            )

                            if is_valid_movement:
                                previous_valid_center = current_center
                            else:
                                current_center = center_to_use
                                is_showing_previous = True
                        else:
                            is_outside_tank = True
                            current_center = previous_valid_center

                instantaneous_speed = 0
                distance = None

                if current_center and previous_center:
                    dx = current_center[0] - previous_center[0]
                    dy = current_center[1] - previous_center[1]
                    distance = np.sqrt(dx**2 + dy**2)

                    if distance <= MAX_DISTANCE_THRESHOLD:
                        instantaneous_speed = distance * original_fps * PIXEL_TO_METER
                        total_speed += instantaneous_speed
                        speed_count += 1

                if enable_visualization:
                    # Visualize only every VISUALIZATION_FRAME_SKIP frames
                    if visualization_frame_count % VISUALIZATION_FRAME_SKIP == 0:
                        key = visualize_processing(
                            frame, masked_frame, enhanced, None, None, fish_vis,
                            current_center, previous_center, distance, instantaneous_speed,
                            not is_showing_previous, is_outside_tank, is_showing_previous
                        )
                        if key == ord('q'):
                            break
                    visualization_frame_count += 1 # Increment visualization frame counter

                previous_center = current_center

                contour_areas = []
                for idx, contour in enumerate(contours):
                    area = cv2.contourArea(contour)
                    contour_areas.append(area)
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        center_x = int(M["m10"] / M["m00"])
                        center_y = int(M["m01"] / M["m00"])
                        write_center_data(center_writer, frame_count, idx, center_x, center_y, instantaneous_speed)

                draw_fish_contours(enhanced, contours, list(box_data.values()), time_spent, original_fps, contour_areas=contour_areas)

                pbar.update(1)
                frame_count += 1

            pbar.close()
            cap.release()
            if enable_visualization:
                cv2.destroyAllWindows()

            for i, (box_name, box_info) in enumerate(box_data.items()):
                box_info["time"] = time_spent[i]
                average_speed = total_speed / speed_count if speed_count > 0 else 0
                data_writer.writerow([box_name, time_spent[i], average_speed])

if __name__ == "__main__":
    main()