import cv2
import numpy as np
from mss import mss

mon = {'top': 1, 'left': 1, 'width': 2880, 'height': 900}
sct = mss()

zoom_level = 5  # Initialize zoom level

# Create a window to show the live video feed
cv2.namedWindow("Live Video Feed", cv2.WINDOW_NORMAL)

# Create a window to show the zoomed-in region
cv2.namedWindow("Zoomed In", cv2.WINDOW_NORMAL)

# Create a sharpening filter kernel
kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])

# Initialize the X and Y coordinates of the zoomed-in region
zoomed_in_x = 0
zoomed_in_y = 0

# Callback function for the mouse click event
def on_mouse_click(event, x, y, flags, param):
    global zoomed_in_x, zoomed_in_y
    if event == cv2.EVENT_LBUTTONDOWN:
        zoomed_in_x = x
        zoomed_in_y = y

# Callback function for the zoom level trackbar (does nothing in this case)
def on_zoom_level_change(zoom):
    global zoom_level
    if zoom >= 5:
        zoom_level = zoom
    else:
        zoom_level = 5

# Set the mouse click event callback for the "Live Video Feed" window
cv2.setMouseCallback("Live Video Feed", on_mouse_click)

# Add slider bar to control the zoom level
cv2.createTrackbar("Zoom Level", "Live Video Feed", 5, 50, on_zoom_level_change)

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

        # Calculate the zoomed-in region
        zoomed_in_width, zoomed_in_height = frame.shape[1] // zoom_level, frame.shape[0] // zoom_level

        # Get the zoomed-in region
        zoomed_in = frame[zoomed_in_y:zoomed_in_y + zoomed_in_height, zoomed_in_x:zoomed_in_x + zoomed_in_width]

        # Resize the zoomed-in region
        zoomed_in = cv2.resize(zoomed_in, (zoomed_in_width * zoom_level, zoomed_in_height * zoom_level))

        # Apply histogram equalization to the zoomed-in image
        zoomed_in_yuv = cv2.cvtColor(zoomed_in, cv2.COLOR_BGR2YUV)
        zoomed_in_yuv[:,:,0] = cv2.equalizeHist(zoomed_in_yuv[:,:,0])
        zoomed_in = cv2.cvtColor(zoomed_in_yuv, cv2.COLOR_YUV2BGR)

        # Apply the sharpening filter to the zoomed-in image
        zoomed_in = cv2.filter2D(zoomed_in, -1, kernel)

        # Show the captured screen
        frame_with_box = frame.copy()
        cv2.rectangle(frame_with_box, (zoomed_in_x, zoomed_in_y), (zoomed_in_x + zoomed_in_width, zoomed_in_y + zoomed_in_height), (0,255, 0), 2)
        cv2.imshow("Live Video Feed", frame_with_box)

        # Show the zoomed-in region in a separate window
        cv2.imshow("Zoomed In", zoomed_in)

        # Quit the script if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    except Exception as e:
        print(f"An error occurred: {e}")
        break
 
