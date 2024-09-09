import cv2
import os

def extract_frames(video_path, output_folder, frame_interval=10):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame at specified intervals
        if frame_count % frame_interval == 0:
            output_path = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames from the video.")

# Usage
video_path = "data/raw_videos/hsv_calibration2.MOV"
output_folder = "data/extracted_frames/sunny"
extract_frames(video_path, output_folder, frame_interval=10)