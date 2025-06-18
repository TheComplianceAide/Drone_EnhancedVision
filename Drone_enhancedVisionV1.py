from mss import mss
import cv2
from PIL import Image
import numpy as np
from time import time
import os

#SurfacePro resolution is 2880 x 1920
#deaktop 1920x1080
mon = {'top':1, 'left':1, 'width':2880, 'height':900}

sct = mss()




while 1:
        sct_img = sct.grab(mon)
        img = Image.frombytes('RGB', (sct_img.size.width, sct_img.size.height), sct_img.rgb)
        frame = np.array(img)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.GaussianBlur(frame,(3,3),0)
        edges = cv2.Canny(frame,150,150)

        window_name = "DJI Mavic3 Motion Detection"
        cv2.imshow(window_name,edges)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

        

        

        #cv2.imshow("frame", mask)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        
        
        
