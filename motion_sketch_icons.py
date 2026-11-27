"""Small antialiased symbols for the field viewer's single control strip."""
import math

import cv2
import numpy as np


def draw_icon(canvas, key, cx, cy, color):
    def line(points, closed=False):
        vertices = np.array([(cx + x, cy + y) for x, y in points], np.int32)
        cv2.polylines(canvas, [vertices], closed, color, 1, cv2.LINE_AA)

    def circle(x, y, r, filled=False):
        cv2.circle(canvas, (cx+x, cy+y), r, color, -1 if filled else 1, cv2.LINE_AA)

    def rectangle(x, y, w, h):
        cv2.rectangle(canvas, (cx+x, cy+y), (cx+x+w, cy+y+h), color, 1, cv2.LINE_AA)

    if key == "view_all":
        rectangle(-8,-6,9,12); rectangle(3,-6,5,5); rectangle(3,1,5,5)
    elif key == "view_overview":
        rectangle(-9,-6,18,12)
    elif key == "view_zoom":
        circle(-2,-2,5); line([(2,2),(8,8)])
    elif key == "view_sketch":
        line([(-9,5),(-5,0),(-2,2),(3,-6),(5,-2),(9,-4)])
    elif key == "view_change":
        for x,y,r in [(-5,3,2),(-1,-1,2),(4,-3,2),(7,4,1)]:
            circle(x,y,r,True)
    elif key == "source":
        rectangle(-9,-7,18,14)
        line([(-8,4),(-3,-1),(1,3),(4,0),(8,4)])
        circle(4,-3,1,True)
    elif key == "landscape":
        circle(5,-5,3)
        line([(-10,5),(-4,-4),(2,4),(5,0),(10,6)])
        line([(-10,8),(10,8)])
    elif key == "sky":
        line([(-9,2),(-9,-1),(-6,-3),(-4,-3),(-2,-7),(2,-8),(5,-5),(6,-3),(9,-1),(9,2),(-9,2)],True)
        line([(-8,6),(3,6)])
        line([(-3,9),(8,9)])
    elif key == "shallows":
        for y in (-5,1,7):
            line([(-9,y),(-6,y-2),(-2,y-2),(2,y+1),(6,y+1),(9,y-1)])
    elif key == "auto":
        line([(-8,7),(-3,-7),(2,7)])
        line([(-6,2),(0,2)])
        line([(5,-9),(6,-5),(10,-4),(6,-3),(5,1),(4,-3),(1,-4),(4,-5)], True)
    elif key == "clear":
        line([(-10,7),(-4,-5),(0,0),(3,-4),(10,7)], True)
        line([(-6,-1),(-4,1),(-2,-2)])
    elif key == "daylight":
        circle(0,0,4)
        for angle in range(0, 360, 45):
            x,y = math.cos(math.radians(angle)),math.sin(math.radians(angle))
            line([(round(7*x),round(7*y)),(round(10*x),round(10*y))])
    elif key == "haze":
        cv2.ellipse(canvas,(cx,cy), (5,5), 0, 180,360, color,1,cv2.LINE_AA)
        for y, dx in ((1,0),(5,3),(9,0)):
            line([(-9+dx,y),(8-dx,y)])
    elif key == "neutral":
        circle(0,0,8)
        cv2.ellipse(canvas,(cx,cy),(7,7),0,90,270,color,-1,cv2.LINE_AA)
    elif key == "dusk":
        line([(4,-9),(-1,-8),(-6,-5),(-8,0),(-6,5),(-2,8),(3,8),(8,5),
              (3,5),(-1,2),(-2,-2),(0,-6)],True)
    elif key == "temporal":
        rectangle(-9,-4,13,10)
        line([(-6,-7),(7,-7),(7,3)])
        line([(-3,-10),(10,-10),(10,0)])
    elif key == "split":
        rectangle(-9,-7,18,14)
        cv2.rectangle(canvas,(cx-8,cy-6),(cx-1,cy+6),color,-1)
    elif key == "zebras":
        rectangle(-9,-7,18,14)
        for x in (-19,-13,-7,-1,5):
            a=max(x,-9); b=min(x+14,9)
            if a<b: line([(a,7-(a-x)),(b,7-(b-x))])
    elif key == "cues":
        circle(0,0,2,True)
        for start,end in ((-70,70),(110,250)):
            cv2.ellipse(canvas,(cx,cy),(8,8),0,start,end,color,1,cv2.LINE_AA)
    elif key == "capture":
        line([(-9,-5),(-4,-5),(-2,-8),(3,-8),(5,-5),(9,-5),(9,7),(-9,7)],True)
        circle(0,1,4)
    elif key == "back" or key == "forward":
        direction = -1 if key=="back" else 1
        line([(-5*direction,-6),(2*direction,0),(-5*direction,6)])
        line([(-direction,-6),(6*direction,0),(-direction,6)])
