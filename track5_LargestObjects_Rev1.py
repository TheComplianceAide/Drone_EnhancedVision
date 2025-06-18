import cv2
import numpy as np
from mss import mss

mon = {'top': 1, 'left': 1, 'width': 2880, 'height': 900}
sct = mss()

# Initialize previous frame and maximum number of objects to track
prev_frame = None
max_objects = 5

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

        # Sort the boxes in descending order of area
        boxes.sort(key=lambda x: x[1], reverse=True)

        # Draw a red cross over the largest moving objects
        for i in range(min(max_objects, len(boxes))):
            x, y, w, h = boxes[i][0]
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
