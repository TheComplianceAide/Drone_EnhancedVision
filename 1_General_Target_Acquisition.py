import cv2
import numpy as np
from mss import mss
from PIL import Image
from time import time

def get_frame(sct, mon):
    sct_img = sct.grab(mon)
    img = Image.frombytes('RGB', (sct_img.size.width, sct_img.size.height), sct_img.rgb)
    return np.array(img)

def process_frame(frame, subtractor):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = subtractor.apply(frame)

    # Apply thresholding to the foreground mask
    _, thresh = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to remove noise
    kernel = np.ones((5,5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around the detected contours
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame

def main():
    mon = {'top':1, 'left':1, 'width':2880, 'height':900}
    sct = mss()

    frame = get_frame(sct, mon)
    subtractor = cv2.createBackgroundSubtractorMOG2(history=60, varThreshold=75, detectShadows=True)

    while True:
        try:
            frame = get_frame(sct, mon)
            processed_frame = process_frame(frame, subtractor)

            window_name = "DJI Mavic3 Motion Detection"
            cv2.imshow(window_name, processed_frame)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

if __name__ == '__main__':
    main()
