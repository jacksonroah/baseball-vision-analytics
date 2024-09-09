import cv2
import numpy as np

def nothing(x):
    pass

def adjust_hsv_range(frame, initial_lower, initial_upper):
    cv2.namedWindow('HSV Adjust')
    cv2.createTrackbar('H_low', 'HSV Adjust', initial_lower[0], 179, nothing)
    cv2.createTrackbar('S_low', 'HSV Adjust', initial_lower[1], 255, nothing)
    cv2.createTrackbar('V_low', 'HSV Adjust', initial_lower[2], 255, nothing)
    cv2.createTrackbar('H_high', 'HSV Adjust', initial_upper[0], 179, nothing)
    cv2.createTrackbar('S_high', 'HSV Adjust', initial_upper[1], 255, nothing)
    cv2.createTrackbar('V_high', 'HSV Adjust', initial_upper[2], 255, nothing)

    while True:
        h_low = cv2.getTrackbarPos('H_low', 'HSV Adjust')
        s_low = cv2.getTrackbarPos('S_low', 'HSV Adjust')
        v_low = cv2.getTrackbarPos('V_low', 'HSV Adjust')
        h_high = cv2.getTrackbarPos('H_high', 'HSV Adjust')
        s_high = cv2.getTrackbarPos('S_high', 'HSV Adjust')
        v_high = cv2.getTrackbarPos('V_high', 'HSV Adjust')

        hsv_lower = np.array([h_low, s_low, v_low])
        hsv_upper = np.array([h_high, s_high, v_high])

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow('HSV Adjust', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyWindow('HSV Adjust')
    return hsv_lower, hsv_upper

def detect_ball(frame, prev_frame, hsv_lower, hsv_upper, prev_ball_location):
    # Convert frames to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Compute absolute difference between current and previous frame
    frame_diff = cv2.absdiff(gray, prev_gray)
    
    # Threshold the difference image
    _, motion_mask = cv2.threshold(frame_diff, 15, 255, cv2.THRESH_BINARY)
    
    # Convert current frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create color mask
    color_mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    
    # Combine motion and color masks
    combined_mask = cv2.bitwise_and(motion_mask, color_mask)
    
    # Apply morphological operations to remove noise (using smaller kernel)
    kernel = np.ones((3,3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the combined mask
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Define dynamic size thresholds based on y-position
    height, width = frame.shape[:2]
    max_y = height * 0.8  # Assume the ball doesn't go below 80% of frame height

    best_ball = None
    min_dist = float('inf')

    for contour in contours:
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        
        # Dynamic size thresholds
        min_radius = max(2, int(3 * (1 - y / max_y)))
        max_radius = max(10, int(30 * (1 - y / max_y)))
        
        if min_radius < radius < max_radius:
            # If we have a previous ball location, prefer closer matches
            if prev_ball_location:
                dist = np.sqrt((x - prev_ball_location[0])**2 + (y - prev_ball_location[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    best_ball = (int(x), int(y), int(radius))
            else:
                best_ball = (int(x), int(y), int(radius))
                break  # Take the first valid ball if no previous location

    return best_ball

def main():
    video_path = "data/raw_videos/hsv_calibration3.MOV"
    output_path = "data/raw_videos/OUTPUT1.mp4"

    cap = cv2.VideoCapture(video_path)

    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read video")
        return
    
     # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Define initial HSV range
    hsv_lower = np.array([27, 15, 90])
    hsv_upper = np.array([84, 180, 255])

    prev_ball_location = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        ball_location = detect_ball(frame, prev_frame, hsv_lower, hsv_upper, prev_ball_location)
        if ball_location:
            x, y, r = ball_location
            cv2.circle(frame, (x, y), r, (0, 0, 255), 2)
            prev_ball_location = (x, y)
        else:
            prev_ball_location = None
        
        cv2.imshow('Frame', frame)

        out.write(frame)
        
        # Update previous frame
        prev_frame = frame.copy()
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            # Adjust HSV range interactively
            cv2.destroyAllWindows()
            hsv_lower, hsv_upper = adjust_hsv_range(frame, hsv_lower, hsv_upper)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()