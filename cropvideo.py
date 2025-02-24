import cv2
import numpy as np
from box_manager import BoxManager

def crop_video(input_path, output_path):
    # Open the video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read video.")
        return

    # Create box manager and window
    box_manager = BoxManager()
    box_manager.frame = frame.copy()
    cv2.namedWindow('Draw Boxes')
    cv2.setMouseCallback('Draw Boxes', box_manager.handle_mouse_event)
    
    # Display instructions
    print("Instructions:")
    print("- Click and drag to draw boxes")
    print("- Click and drag box corners to adjust")
    print("- Click and drag inside box to move it")
    print("- Press 'z' to undo last box")
    print("- Press 'r' to reset all boxes")
    print("- Press 'c' to clear all boxes")
    print("- Press 'space' or 'enter' to confirm selection")
    print("- Press 'q' to quit without saving")
    
    while True:
        # Draw boxes on frame copy
        frame_display = box_manager.draw_boxes(frame)
        cv2.imshow('Draw Boxes', frame_display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):  # Clear boxes
            box_manager.boxes = []
            box_manager.labels = []
        elif key == ord('q'):  # Quit without saving
            cv2.destroyAllWindows()
            return
        elif key in [ord(' '), 13]:  # Space or Enter to confirm
            break
        else:
            box_manager.handle_key_press(key)
    
    cv2.destroyAllWindows()
    
    if not box_manager.boxes:
        print("No boxes drawn. Exiting.")
        return
        
    # Calculate the bounding rectangle that contains all boxes
    all_points = []
    for box in box_manager.boxes:
        all_points.extend(box)  # Now each box has 4 points
    x_coords = [p[0] for p in all_points]
    y_coords = [p[1] for p in all_points]
    
    x1, y1 = min(x_coords), min(y_coords)
    x2, y2 = max(x_coords), max(y_coords)
    
    # Get video properties
    width = x2 - x1
    height = y2 - y1
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Process the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("\nProcessing video...")
    
    for frame_count in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Crop frame using bounding rectangle coordinates
        cropped_frame = frame[y1:y2, x1:x2]
        
        # Write the cropped frame
        out.write(cropped_frame)
        
        # Show progress
        if frame_count % 30 == 0:  # Update progress every 30 frames
            progress = (frame_count + 1) / total_frames * 100
            print(f"\rProgress: {progress:.1f}%", end="")

    print("\nDone!")

    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    input_video = "/Users/manasvenkatasairavulapalli/Desktop/Research Work/ml/Vid/originals/n1.mov"
    output_video = "cropped_video.mp4"
    crop_video(input_video, output_video)
