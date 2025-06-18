import cv2
import numpy as np
from mss import mss
import time

def enhance_drone_footage(frame, brightness, contrast):
    # Apply Reinhard's tone mapping for dynamic range compression
    frame_float = frame.astype(np.float32) / 255.0
    tone_map = cv2.createTonemapReinhard(gamma=1, intensity=0.5, light_adapt=0.5, color_adapt=0)
    frame_tone_mapped = tone_map.process(frame_float)
    frame = (frame_tone_mapped * 255).astype(np.uint8)
    
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_frame = clahe.apply(gray_frame)

    # Apply Gaussian blur for noise reduction
    enhanced_frame = cv2.GaussianBlur(enhanced_frame, (5, 5), 0)

    # Apply a bilateral filter for further noise reduction while keeping the edges sharp
    enhanced_frame = cv2.bilateralFilter(enhanced_frame, 9, 75, 75)

    # Convert the enhanced grayscale frame back to BGR color space
    enhanced_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_GRAY2BGR)

    # Apply sharpening filter
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    enhanced_frame = cv2.filter2D(enhanced_frame, -1, kernel)

    # Adjust brightness and contrast
    enhanced_frame = cv2.addWeighted(enhanced_frame, contrast, np.zeros_like(enhanced_frame), 0, brightness * 255)

    return enhanced_frame

def on_brightness_trackbar(val):
    global brightness
    brightness = val / 100.0 - 1.0

def on_contrast_trackbar(val):
    global contrast
    contrast = val / 100.0

# Create a window to display the enhanced drone footage
cv2.namedWindow("Enhanced Drone Footage")

# Create trackbars for brightness and contrast adjustment
cv2.createTrackbar("Brightness", "Enhanced Drone Footage", 100, 200, on_brightness_trackbar)
cv2.createTrackbar("Contrast", "Enhanced Drone Footage", 100, 300, on_contrast_trackbar)

brightness = 0.0
contrast = 1.0

mon = {'top': 1, 'left': 1, 'width': 2880, 'height': 900}
sct = mss()
frame_rate = 30
prev_frame_time = 0

while True:
    try:
        # Capture the current screen using MSS
        sct_img = sct.grab(mon)

        # Convert the screen image to a numpy array
        frame = cv2.cvtColor(
            cv2.cvtColor(
                np.array(sct_img), cv2.COLOR_BGR2RGB
            ),
            cv2.COLOR_RGB2BGR
        )

        # Enhance the drone footage
        enhanced_frame = enhance_drone_footage(frame, brightness, contrast)

        if enhanced_frame is not None:
            # Show the enhanced captured screen
            cv2.imshow("Enhanced Drone Footage", enhanced_frame)

        # Control the frame rate
        curr_frame_time = time.time()
        delay = max(1 / frame_rate - (curr_frame_time - prev_frame_time), 0)
        time.sleep(delay)
        prev_frame_time = curr_frame_time

        # Quit the script if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    except Exception as e:
        print(f"An error occurred: {e}")
        break
