from mss import mss
import cv2
from PIL import Image
import numpy as np
from time import time

#SurfacePro resolution is 2880 x 1920
mon = {'top':1, 'left':1, 'width':1920, 'height':1080}

sct = mss()

first_frame = None

while 1:
        sct_img = sct.grab(mon)
        img = Image.frombytes('RGB', (sct_img.size.width, sct_img.size.height), sct_img.rgb)
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        img_bgr = cv2.GaussianBlur(img_bgr,(21,21),0)

        if first_frame is None:
            first_frame = img_bgr
            continue
        delta_frame = cv2.absdiff(first_frame,img_bgr)
        threshold_frame = cv2.threshold(delta_frame, 50, 255, cv2.THRESH_BINARY)[1]
        threshold_frame = cv2.dilate(threshold_frame, None, iterations=2)

        (cntr,_) = cv2.findContours(threshold_frame.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in cntr:
            #increase/decrease sensativity
            if cv2.contourArea(contour) < 10000:
                continue
            (x,y,w,h) = cv2.boundingRect(contour)
            cv2.rectangle(img_bgr,(x,y),(x+w,y+h),(0,255,0),3)
        
        
        cv2.imshow('test', img_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        
        
        
