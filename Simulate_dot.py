import cv2
import numpy as np
import pandas as pd

def create_moving_dot_animation(csv_path, output_path, frame_size=(1920, 1080),
                                dot_radius=10, dot_color=(0, 0, 255),
                                duration=20, fps=30.0):
    """
    Create a video animation of a moving dot based on coordinates from a CSV file.
    
    Only coordinates corresponding to the specified duration (in seconds)
    are processed.
    
    Args:
        csv_path (str): Path to CSV file containing frame, x, y coordinates.
        output_path (str): Path to save output video.
        frame_size (tuple): Width and height of output video frames.
        dot_radius (int): Radius of the moving dot.
        dot_color (tuple): BGR color of the dot.
        duration (int): Duration in seconds to use.
        fps (float): Frames per second to use.
    """
    # Read coordinates from CSV
    df = pd.read_csv(csv_path)
    print("CSV columns:", df.columns)  # Debug line to check column names

    # Limit the number of rows to "duration" seconds worth of frames.
    max_frames = int(duration * fps)
    if len(df) > max_frames:
        print(f"Limiting coordinates to the first {max_frames} rows for {duration} seconds.")
        df = df.head(max_frames)
    else:
        print(f"CSV contains only {len(df)} frames. Processing all rows.")

    # Initialize video writer.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    # Create frames and draw dot.
    for _, row in df.iterrows():
        # Create a blank frame.
        frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
        
        # Get coordinates - adjust the header names if needed.
        try:
            x, y = int(row['center_x']), int(row['center_y'])  # Updated column names
        except KeyError as e:
            print(f"KeyError: {e}. Check the CSV headers.")
            return
        
        # Draw dot.
        cv2.circle(frame, (x, y), dot_radius, dot_color, -1)
        
        # Write frame.
        out.write(frame)
    
    # Release video writer.
    out.release()

def main():
    # Example usage
    csv_path = "/Users/manasvenkatasairavulapalli/Desktop/Research Work/ml/fish_coordinates.csv"  # CSV file with frame, x, y. Ensure headers are as expected.
    output_path = "dot.mp4"  # Output video file.
    
    create_moving_dot_animation(
        csv_path=csv_path,
        output_path=output_path,
        frame_size=(1920, 1080),
        dot_radius=10,
        dot_color=(0, 0, 255),  # Red dot in BGR.
        duration=20,           # Only process 20 seconds worth of frames.
        fps=30.0               # FPS used to compute total frames.
    )
    
if __name__ == "__main__":
    main()
