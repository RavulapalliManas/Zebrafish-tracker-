"""
This module defines the BoxManager class for handling interactive box drawing, manipulation,
and storage for video analysis. It supports creating, modifying, and saving quadrilateral
regions of interest (boxes) on video frames using mouse interactions.
"""
import cv2
import numpy as np
import json
import sys

# Constants
DOUBLE_CLICK_THRESHOLD = 300  # milliseconds - Time threshold for double-click detection

class BoxManager:
    """
    Manages interactive drawing, manipulation, and storage of quadrilateral boxes on video frames.

    Handles mouse events for drawing new boxes, selecting and moving existing boxes,
    and adjusting box corners. Supports saving and loading box configurations to/from JSON files.

    Attributes:
        boxes (list): List of boxes, where each box is a list of four (x, y) corner tuples.
        labels (list): List of labels corresponding to each box.
        drawing_polygon (bool): Flag indicating if a new polygon box is currently being drawn.
        current_polygon (list): List of points for the polygon currently being drawn.
        frame (np.array): The current video frame (not actively used in current implementation).
        current_box_start (tuple): Starting point of the current box being drawn (not actively used).
        current_box_end (tuple): Ending point of the current box being drawn (not actively used).
        selected_box_index (int): Index of the currently selected box for manipulation.
        selected_corner_index (int): Index of the currently selected corner for adjustment.
        moving_box (bool): Flag indicating if a box is currently being moved.
        move_start (tuple): Starting point of box movement.
        last_click_time (float): Timestamp of the last mouse click for double-click detection.
        double_click_threshold (int): Time threshold in milliseconds to detect double clicks.
    """
    def __init__(self):
        """
        Initializes BoxManager with empty lists for boxes and labels, and sets interaction flags to default.
        """
        self.boxes = []
        self.labels = []
        self.drawing_polygon = False
        self.current_polygon = []
        self.frame = None  # Potentially for future use, not currently used
        self.current_box_start = None # Not currently used
        self.current_box_end = None # Not currently used
        self.selected_box_index = None
        self.selected_corner_index = None
        self.moving_box = False
        self.move_start = None
        self.last_click_time = 0
        self.double_click_threshold = DOUBLE_CLICK_THRESHOLD

    def get_near_corner(self, box, point, threshold=10):
        """
        Checks if a given point is near any corner of a box.

        Args:
            box (list): List of corner points defining the box.
            point (tuple): The point to check for proximity to box corners.
            threshold (int, optional): Maximum pixel distance to consider a point 'near' a corner. Defaults to 10.

        Returns:
            int or None: Index of the nearest corner if a corner is within the threshold distance, otherwise None.
        """
        for i, corner in enumerate(box):
            if np.hypot(corner[0] - point[0], corner[1] - point[1]) < threshold:
                return i
        return None

    def point_in_box(self, point, box):
        """
        Checks if a point is inside a given quadrilateral box.

        Args:
            point (tuple): The point to check.
            box (list): List of corner points defining the box.

        Returns:
            bool: True if the point is inside the box, False otherwise.
        """
        pts = np.array(box, np.int32).reshape((-1, 1, 2))
        return cv2.pointPolygonTest(pts, point, False) >= 0

    def find_existing_corner(self, point, threshold=10):
        """
        Searches for an existing corner point from any box that is near the given point.

        Used for implementing double-click corner reuse feature.

        Args:
            point (tuple): The point to check for nearby existing corners.
            threshold (int, optional): Maximum pixel distance to consider a point 'near' an existing corner. Defaults to 10.

        Returns:
            tuple or None: Coordinates of the nearest existing corner if found within the threshold, otherwise None.
        """
        for box in self.boxes:
            for corner in box:
                if np.hypot(corner[0] - point[0], corner[1] - point[1]) < threshold:
                    return corner
        return None

    def handle_mouse_event(self, event, x, y, flags, param):
        """
        Handles mouse events for interactive box drawing and manipulation.

        This is the callback function for mouse events, handling box creation, selection,
        moving, and corner adjustments based on different mouse actions (click, drag, move).

        Args:
            event (int): OpenCV mouse event type.
            x (int): x-coordinate of the mouse event.
            y (int): y-coordinate of the mouse event.
            flags (int): Any flags passed by OpenCV (not used).
            param: Any extra parameters passed by OpenCV (not used).
        """
        point = (x, y)
        current_time = cv2.getTickCount() / cv2.getTickFrequency() * 1000  # Current time in ms

        if event == cv2.EVENT_LBUTTONDOWN:
            time_diff = current_time - self.last_click_time
            self.last_click_time = current_time

            if time_diff < self.double_click_threshold:
                existing_corner = self.find_existing_corner(point)
                if existing_corner and (self.drawing_polygon or not self.current_polygon):
                    if not self.drawing_polygon:
                        self.drawing_polygon = True
                        self.current_polygon = [existing_corner]
                    else:
                        self.current_polygon.append(existing_corner)

                    if len(self.current_polygon) == 4:
                        self.boxes.append(self.current_polygon.copy())
                        self.labels.append(f"Box {len(self.boxes)}")
                        self.drawing_polygon = False
                        self.current_polygon = []
                    return

            for i, box in enumerate(self.boxes):
                corner_idx = self.get_near_corner(box, point)
                if corner_idx is not None:
                    self.selected_box_index = i
                    self.selected_corner_index = corner_idx
                    return

            for i, box in enumerate(self.boxes):
                if self.point_in_box(point, box):
                    self.selected_box_index = i
                    self.moving_box = True
                    self.move_start = point
                    return

            if not self.drawing_polygon:
                self.drawing_polygon = True
                self.current_polygon = [point]
            else:
                self.current_polygon.append(point)
                if len(self.current_polygon) == 4:
                    self.boxes.append(self.current_polygon.copy())
                    self.labels.append(f"Box {len(self.boxes)}")
                    self.drawing_polygon = False
                    self.current_polygon = []

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.selected_corner_index is not None and self.selected_box_index is not None:
                self.boxes[self.selected_box_index][self.selected_corner_index] = point
            elif self.moving_box and self.selected_box_index is not None and self.move_start is not None:
                dx = x - self.move_start[0]
                dy = y - self.move_start[1]
                self.boxes[self.selected_box_index] = [
                    (cx + dx, cy + dy) for (cx, cy) in self.boxes[self.selected_box_index]
                ]
                self.move_start = point

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.drawing_polygon:
                self.drawing_polygon = False
                self.current_polygon = []

        elif event == cv2.EVENT_LBUTTONUP:
            if not self.drawing_polygon:
                self.selected_box_index = None
                self.selected_corner_index = None
                self.moving_box = False
                self.move_start = None

    def draw_boxes(self, frame):
        """
        Draws the stored boxes and any in-progress polygon on the given frame.

        Args:
            frame (np.array): The frame on which to draw the boxes.

        Returns:
            np.array: The frame with boxes and labels drawn on it.
        """
        temp_frame = frame.copy()

        for i, box in enumerate(self.boxes):
            pts = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(temp_frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            if i < len(self.labels):
                cv2.putText(temp_frame, self.labels[i], (box[0][0], box[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            for corner in box:
                cv2.circle(temp_frame, corner, radius=5, color=(0, 0, 255), thickness=-1)

        if self.drawing_polygon and self.current_polygon:
            for i in range(len(self.current_polygon) - 1):
                cv2.line(temp_frame, self.current_polygon[i], self.current_polygon[i+1],
                         (255, 0, 0), 2)

            for point in self.current_polygon:
                cv2.circle(temp_frame, point, radius=5, color=(0, 0, 255), thickness=-1)

            remaining = 4 - len(self.current_polygon)
            if remaining > 0:
                cv2.putText(temp_frame, f"Add {remaining} more point{'s' if remaining > 1 else ''}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        return temp_frame

    def remove_last_box(self):
        """
        Removes the last added box and its label, if any.
        """
        if self.boxes:
            self.boxes.pop()
            if self.labels:
                self.labels.pop()

    def get_box_data(self):
        """
        Returns the current box data.

        Returns:
            dict: A dictionary containing box labels as keys and box coordinates and time spent as values.
        """
        return {label: {"coords": box, "time": 0} for label, box in zip(self.labels, self.boxes)}

    def save_configuration(self, filename):
        """
        Saves the current box configuration (boxes and labels) to a JSON file.

        Args:
            filename (str): Path to the file where the configuration should be saved.
        """
        config = {"boxes": self.boxes, "labels": self.labels}
        with open(filename, 'w') as f:
            json.dump(config, f)

    def load_configuration(self, filename):
        """
        Loads box configuration (boxes and labels) from a JSON file.

        Args:
            filename (str): Path to the configuration file to load.
        """
        try:
            with open(filename, 'r') as f:
                config = json.load(f)
                self.boxes = config["boxes"]
                self.labels = config["labels"]
        except FileNotFoundError:
            print(f"Warning: Configuration file not found at {filename}. Starting with default boxes.")
        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON format in {filename}. Starting with default boxes.")
        except KeyError as e:
            print(f"Warning: Key {e} missing in {filename}. Starting with default boxes.")

    def handle_key_press(self, key):
        """
        Handles key press events for box management.

        'z': Undo last point while drawing polygon or remove last box if not drawing.
        'r': Reset all boxes.
        'q': Quit application (exits program).
        'c': Cancel current polygon drawing.

        Args:
            key (int): Key code of the pressed key.
        """
        if key == ord('z'):
            if self.drawing_polygon and self.current_polygon:
                if len(self.current_polygon) > 0:
                    self.current_polygon.pop()
                if len(self.current_polygon) == 0:
                    self.drawing_polygon = False
            else:
                self.remove_last_box()
        elif key == ord('r'):
            self.boxes = []
            self.labels = []
            self.drawing_polygon = False
            self.current_polygon = []
        elif key == ord('q'):
            print("Quit key pressed. Exiting...")
            sys.exit()
        elif key == ord('c'):
            self.drawing_polygon = False
            self.current_polygon = []

    def add_box_from_coordinates(self, coordinates, label=None):
        """
        Adds a box to the manager using provided coordinates.

        This method is used when boxes are defined programmatically or loaded from a configuration,
        rather than drawn interactively.

        Args:
            coordinates (list): A list of exactly four (x, y) tuples representing the box corners.
            label (str, optional): Label for the box. If None, a default label is assigned. Defaults to None.

        Raises:
            ValueError: If the coordinates list does not contain exactly four points.
        """
        if len(coordinates) != 4:
            raise ValueError("Coordinates must contain exactly four points for a quadrilateral.")
        self.boxes.append(coordinates)
        if label is None:
            label = f"Box {len(self.boxes)}"
        self.labels.append(label) 