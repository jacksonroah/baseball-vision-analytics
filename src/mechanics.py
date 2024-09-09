import cv2
import numpy as np
from ultralytics import YOLO
import face_recognition

# Initialize YOLO model
model = YOLO('yolov8n-pose.pt')

def extract_pose_data(yolo_results):
    pose_data = []
    for result in yolo_results:
        if result.keypoints is not None:
            # Extract keypoints
            keypoints = result.keypoints.cpu().numpy()
            
            # YOLO returns keypoints in the format [x, y, confidence]
            # We'll keep only the x and y coordinates
            frame_pose = keypoints[0, :, :2]  # Shape: (num_keypoints, 2)
            
            # Ensure consistent number of keypoints (e.g., 17 for COCO format)
            if frame_pose.shape[0] == 17:
                pose_data.append(frame_pose)
            else:
                # If the number of keypoints is inconsistent, pad with NaN values
                padded_pose = np.full((17, 2), np.nan)
                padded_pose[:frame_pose.shape[0], :] = frame_pose
                pose_data.append(padded_pose)
    
    if pose_data:
        return np.array(pose_data)
    else:
        return np.array([]) 

def analyze_pose_data(pose_data):
    if len(pose_data) == 0:
        return None
    
    # Remove frames with all NaN values
    valid_frames = pose_data[~np.isnan(pose_data).all(axis=(1, 2))]
    
    if len(valid_frames) < 2:  # Need at least 2 frames for velocity calculation
        return None
    
    # Calculate velocities and accelerations using valid frames
    velocities = np.diff(valid_frames, axis=0)
    accelerations = np.diff(velocities, axis=0)
    
    # Example: Track the movement of the right wrist (keypoint index 10)
    right_wrist_trajectory = valid_frames[:, 10, :]
    right_wrist_velocity = velocities[:, 10, :]
    right_wrist_acceleration = accelerations[:, 10, :]
    
    return {
        'right_wrist_trajectory': right_wrist_trajectory,
        'right_wrist_velocity': right_wrist_velocity,
        'right_wrist_acceleration': right_wrist_acceleration
    }

def visualize_trajectory(frame, trajectory, color=(0, 255, 0), thickness=2):
    for i in range(1, len(trajectory)):
        start_point = tuple(trajectory[i-1].astype(int))
        end_point = tuple(trajectory[i].astype(int))
        cv2.line(frame, start_point, end_point, color, thickness)
    return frame

# Initialize video capture
video_path = 'data/raw_videos/hsv_calibration3.MOV'
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('jackson_pose_analysis_output.mp4', fourcc, 30.0, (frame_width, frame_height))

# Initialize persistent name variable and flag for face recognition
persistent_name = None
run_face_recognition = True

print("Starting video processing...")
frame_buffer = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Reached end of video.")
        break

    print("Processing frame...")

    # Face Recognition (using your existing code)
    if run_face_recognition:
        face_locations = face_recognition.face_locations(frame)
        print(f"Found {len(face_locations)} face(s) in this frame.")
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                persistent_name = known_face_names[best_match_index]
                run_face_recognition = False
                break
    print(f"Identified faces: {persistent_name if persistent_name else 'None'}")

    # YOLO Pose Estimation
    yolo_results = model(frame)
    frame_with_pose = yolo_results[0].plot(labels=False, boxes=False, conf=False)

    # Extract pose data
    pose_data = extract_pose_data(yolo_results)
    frame_buffer.append(pose_data[0] if len(pose_data) > 0 else None)

    # Analyze pose data if we have enough frames
    if len(frame_buffer) >= 30:  # Analyze last 30 frames
        valid_frames = [f for f in frame_buffer if f is not None]
        if len(valid_frames) > 0:
            analysis = analyze_pose_data(np.array(valid_frames))
            if analysis:
                frame_with_pose = visualize_trajectory(frame_with_pose, analysis['right_wrist_trajectory'][-30:])
        frame_buffer.pop(0)  # Remove oldest frame

    # Write names for Face Recognition
    if persistent_name:
        cv2.putText(frame_with_pose, persistent_name, (50, frame_height - 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

    cv2.imshow('Video', frame_with_pose)
    out.write(frame_with_pose)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Releasing video objects...")
out.release()
cap.release()
cv2.destroyAllWindows()