import cv2
import numpy as np

def nothing(x):
    pass

# Load an image
image = cv2.imread('data/calibration_frames/farC.jpg')
image = cv2.resize(image, (600, 400))  # Resize for display

# Create a window
cv2.namedWindow('image')

# Create trackbars for color change
cv2.createTrackbar('H_low', 'image', 0, 179, nothing)
cv2.createTrackbar('S_low', 'image', 0, 255, nothing)
cv2.createTrackbar('V_low', 'image', 0, 255, nothing)
cv2.createTrackbar('H_high', 'image', 179, 179, nothing)
cv2.createTrackbar('S_high', 'image', 255, 255, nothing)
cv2.createTrackbar('V_high', 'image', 255, 255, nothing)

while(1):
    # Get current positions of trackbars
    h_low = cv2.getTrackbarPos('H_low', 'image')
    s_low = cv2.getTrackbarPos('S_low', 'image')
    v_low = cv2.getTrackbarPos('V_low', 'image')
    h_high = cv2.getTrackbarPos('H_high', 'image')
    s_high = cv2.getTrackbarPos('S_high', 'image')
    v_high = cv2.getTrackbarPos('V_high', 'image')

    # Define range of ball color in HSV
    lower_ball = np.array([h_low, s_low, v_low])
    upper_ball = np.array([h_high, s_high, v_high])

    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only ball colors
    mask = cv2.inRange(hsv, lower_ball, upper_ball)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image, image, mask=mask)

    cv2.imshow('image', res)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()