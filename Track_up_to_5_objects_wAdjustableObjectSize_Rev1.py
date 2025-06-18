import cv2
import numpy as np
from mss import mss

mon = {'top': 1, 'left': 1, 'width': 2880, 'height': 900}
sct = mss()

# Initialize previous frame and maximum number of objects to track
prev_frame = None
max_objects = 5

# Create a named window for the slider
cv2.namedWindow("Live Video Feed")

# Slider callback function (no action needed in this case)
def slider_callback(val):
    pass

# Create the slider control
cv2.createTrackbar("Size Range", "Live Video Feed", 1, 10, slider_callback)

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

        # Resize the frame for faster processing
        frame = cv2.resize(frame, (2880, 900))

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise and improve object detection
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Initialize the previous frame if not already done
        if prev_frame is None:
            prev_frame = gray
            continue

        # Calculate the absolute difference between the current and previous frame
        frame_delta = cv2.absdiff(prev_frame, gray)

        # Threshold the frame delta to create a binary image
        thresh = cv2.threshold(frame_delta, 30, 255, cv2.THRESH_BINARY)[1]

        # Dilate the thresholded image to fill in holes
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get the bounding boxes and areas of the contours
        boxes = [(cv2.boundingRect(c), cv2.contourArea(c)) for c in contours]

        # If there are moving objects detected
        if boxes:
            # Sort the boxes in descending order of area
            boxes.sort(key=lambda x: x[1], reverse=True)

            # Get the slider position and calculate the object size detection range
            size_range = cv2.getTrackbarPos("Size Range", "Live Video Feed") * 0.1

            # Calculate the median area of the boxes
            median_area = np.median([box[1] for box in boxes])

            # Filter objects based on size range
            filtered_boxes = [box for box in boxes if median_area * (1 - size_range) <= box[1] <= median_area * (1 + size_range)]

            # Draw a red cross over the detected objects
            for i in range(min(max_objects, len(filtered_boxes))):
                x, y, w, h = filtered_boxes[i][0]
                cv2.line(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.line(frame, (x + w, y), (x, y + h), (0, 255, 0), 2)

        # Show the captured screen
        cv2.imshow("Live Video Feed", frame)

        # Set the current frame as the previous frame for the next iteration
        prev_frame = gray

        # Quit the script if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    except Exception as e:
        print(f"An error occurred: {e}")
        break

