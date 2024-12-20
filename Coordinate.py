import cv2
import numpy as np

def draw_boxes(event, x, y, flags, param):
    global current_box, boxes, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_box = [(x, y)]

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            current_box[1:] = [(x, y)]

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_box.append((x, y))
        boxes.append(tuple(current_box))


def define_boxes(video_path, original_fps=30, slowed_fps=10):
    """
    Allows user to draw bounding boxes on first frame of video and returns box coordinates with timing data.
    
    Args:
        video_path: Path to input video file
        original_fps: Original video FPS (default 30)
        slowed_fps: Target FPS for processing (default 10)
        
    Returns:
        Dictionary containing box coordinates and timing data
    """
    global current_box, boxes, drawing

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read the video.")
        cap.release()
        return {}

    current_box = []
    boxes = []
    drawing = False

    cv2.namedWindow("Draw Boxes")
    cv2.setMouseCallback("Draw Boxes", draw_boxes)

    print("Draw regions by dragging with the left mouse button. Press 's' to save and exit.")

    while True:
        temp_frame = frame.copy()

        for box in boxes:
            cv2.rectangle(temp_frame, box[0], box[1], (0, 255, 0), 2)

        if drawing and len(current_box) == 2:
            cv2.rectangle(temp_frame, current_box[0], current_box[1], (255, 0, 0), 2)

        cv2.imshow("Draw Boxes", temp_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            break

    cv2.destroyWindow("Draw Boxes")
    cap.release()

    box_data = {f"Box {i+1}": {"coords": box, "time": 0} for i, box in enumerate(boxes)}

    return box_data
