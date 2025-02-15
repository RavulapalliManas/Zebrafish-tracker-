import cv2
import numpy as np
import json
import sys

class BoxManager:
    def __init__(self):
        self.boxes = [] 
        self.labels = []
        self.drawing = False
        self.current_box_start = None
        self.current_box_end = None
        self.selected_box_index = None
        self.selected_corner_index = None
        self.moving_box = False
        self.move_start = None

    def get_near_corner(self, box, point, threshold=10):
        """Return the index of the corner if point is within threshold; else None."""
        for i, corner in enumerate(box):
            if np.hypot(corner[0] - point[0], corner[1] - point[1]) < threshold:
                return i
        return None

    def point_in_box(self, point, box):
        """Return True if the point is inside the polygon defined by the box."""
        pts = np.array(box, np.int32).reshape((-1, 1, 2))
        return cv2.pointPolygonTest(pts, point, False) >= 0

    def handle_mouse_event(self, event, x, y, flags, param):
        point = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
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
            self.drawing = True
            self.current_box_start = point
            self.current_box_end = point

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_box_end = point
            elif self.selected_corner_index is not None and self.selected_box_index is not None:
                self.boxes[self.selected_box_index][self.selected_corner_index] = point
            elif self.moving_box and self.selected_box_index is not None and self.move_start is not None:
                dx = x - self.move_start[0]
                dy = y - self.move_start[1]
                self.boxes[self.selected_box_index] = [
                    (cx + dx, cy + dy) for (cx, cy) in self.boxes[self.selected_box_index]
                ]
                self.move_start = point

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                x1, y1 = self.current_box_start
                x2, y2 = self.current_box_end
                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)
                new_box = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
                self.boxes.append(new_box)
                self.labels.append(f"Box {len(self.boxes)}")
                self.current_box_start = None
                self.current_box_end = None
            self.selected_box_index = None
            self.selected_corner_index = None
            self.moving_box = False
            self.move_start = None

    def draw_boxes(self, frame):
        temp_frame = frame.copy()
        for i, box in enumerate(self.boxes):
            pts = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(temp_frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            if i < len(self.labels):
                cv2.putText(temp_frame, self.labels[i], (box[0][0], box[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            for corner in box:
                cv2.circle(temp_frame, corner, radius=5, color=(0, 0, 255), thickness=-1)
        if self.drawing and self.current_box_start and self.current_box_end:
            cv2.rectangle(temp_frame, self.current_box_start, self.current_box_end, (255, 0, 0), 2)
        return temp_frame

    def remove_last_box(self):
        if self.boxes:
            self.boxes.pop()
            self.labels.pop()

    def get_box_data(self):
        return {label: {"coords": box, "time": 0} for label, box in zip(self.labels, self.boxes)}

    def save_configuration(self, filename):
        config = {"boxes": self.boxes, "labels": self.labels}
        with open(filename, 'w') as f:
            json.dump(config, f)

    def load_configuration(self, filename):
        with open(filename, 'r') as f:
            config = json.load(f)
            self.boxes = config["boxes"]
            self.labels = config["labels"]

    def handle_key_press(self, key):
        if key == ord('z'):
            self.remove_last_box()
        elif key == ord('r'):
            self.boxes = []
            self.labels = []
        elif key == ord('q'):
            print("Quit key pressed. Exiting...")
            sys.exit()  

    def add_box_from_coordinates(self, coordinates, label=None):
        """
        Add a box using a list of coordinates.
        
        Args:
            coordinates: A list of four tuples, each representing a corner of the box.
            label: Optional label for the box.
        """
        if len(coordinates) != 4:
            raise ValueError("Coordinates must contain exactly four points.")
        self.boxes.append(coordinates)
        if label is None:
            label = f"Box {len(self.boxes)}"
        self.labels.append(label) 