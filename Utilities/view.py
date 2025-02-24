import cv2

# Global variables to store the pixel value and mouse position
pixel_value = None
mouse_x, mouse_y = 0, 0  # Initialize mouse position

def mouse_callback(event, x, y, flags, param):
    global pixel_value, mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        # Ensure the coordinates are within the frame dimensions
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            # Get the pixel value at the current mouse position
            b, g, r = frame[y, x]  # OpenCV uses BGR format
            pixel_value = f"Position: ({x}, {y}) - Value: (R: {r}, G: {g}, B: {b})"
            mouse_x, mouse_y = x, y  # Update global mouse position

def view_video(video_path):
    global frame  # Declare frame as global to access in the callback
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    while True:
        # Set the current frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the frame
        cv2.imshow('Video Frame', frame)
        cv2.setMouseCallback('Video Frame', mouse_callback)  # Set mouse callback

        key = cv2.waitKey(1)  # Wait for a key press for a short time

        if key == ord('q'):  # Quit if 'q' is pressed
            break
        elif key == ord('a'):  # 'a' key for previous frame
            current_frame = max(0, current_frame - 1)  # Go to previous frame
        elif key == ord('d'):  # 'd' key for next frame
            current_frame = min(frame_count - 1, current_frame + 1)  # Go to next frame

        # Display pixel value if available
        if pixel_value and pixel_value.startswith("Position"):
            # Display the pixel value at the cursor position
            cv2.putText(frame, pixel_value.split(" - ")[0], (mouse_x, mouse_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "/Users/manasvenkatasairavulapalli/Desktop/Research Work/ml/Vid/originals/n1cropped.mp4"  # Update this path to your video file
    view_video(video_path)
