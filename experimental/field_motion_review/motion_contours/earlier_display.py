"""Earlier cue thumbnails and a magnified view of their original pixels."""
import cv2 as cv
import numpy as np

from viewer import WIDTH, label

PANEL = (23,20,18)
ACCENT = (118,223,252)
TEXT = (225,229,235)
MUTED = (169,181,195)
CARD_BOX = (420,587,WIDTH,668)
TABS_X = WIDTH+29
TABS_Y = 598
TAB_SIZE = 284


def in_card(x, y):
    x0, y0, x1, y1 = CARD_BOX
    return x0 <= x < x1 and y0 <= y < y1


def tab_width(count):
    return min(TAB_SIZE, 3*TAB_SIZE//count)


def replay_tab(x, y, count=3):
    width = tab_width(count)
    if TABS_Y <= y < TABS_Y+48 and TABS_X <= x < TABS_X+count*width:
        return (x-TABS_X)//width
    return None


def focus(exposure, width_fraction, height_fraction):
    bgr = exposure.bgr
    h,w = bgr.shape[:2]
    cw,ch = max(1,round(w*width_fraction)),max(1,round(h*height_fraction))
    x,y = exposure.focus_xy
    u = int(np.clip(round(x-cw/2),0,w-cw))
    v = int(np.clip(round(y-ch/2),0,h-ch))
    return bgr[v:v+ch,u:u+cw],(x-u,y-v)


def offset_text(exposure, origin):
    return f'{round(1000*(exposure.stamp-origin)):+d} ms'


def replay_patch(exposures, index):
    """Keep the same camera rectangle when it contains all measured positions."""
    item = exposures[index]
    rectangles = np.asarray([image.rect for image in exposures])
    left,top = rectangles[:,:2].max(axis=0)
    right,bottom = (rectangles[:,:2]+rectangles[:,2:]).min(axis=0)
    cw = min(right-left,round(item.rect[2]*96/160))
    ch = min(bottom-top,round(item.rect[3]*60/104))
    points = rectangles[:,:2]+np.asarray([image.focus_xy for image in exposures])
    minimum,maximum = points.min(axis=0),points.max(axis=0)
    if cw>=36 and ch>=24:
        center = (minimum+maximum)*.5
        x = int(np.clip(round(center[0]-cw/2),left,right-cw))
        y = int(np.clip(round(center[1]-ch/2),top,bottom-ch))
        if (np.all(minimum >= (x+3,y+3)) and np.all(maximum < (x+cw-3,y+ch-3))):
            u,v = x-item.rect[0],y-item.rect[1]
            return item.bgr[v:v+ch,u:u+cw],points[index]-(x,y),True
    patch,relative = focus(item,96/160,60/104)
    return patch,relative,False


def paint_card(canvas, card, camera_stamp):
    x0,y0,x1,y1 = CARD_BOX
    canvas[y0:y1,x0:x1] = PANEL
    age_ms = max(0,round(1000*(camera_stamp-card.stamp)))
    age = f'{age_ms} ms ago' if age_ms<1000 else f'{age_ms/1000:.1f} s ago'
    label(canvas,f'Earlier motion   /   {age}',426,608,.48,ACCENT)
    label(canvas,'Click to inspect',821,608,.43,MUTED)
    for index,exposure in enumerate(card.exposures):
        x = 426+174*index
        crop,_ = focus(exposure,48/160,24/104)
        canvas[614:666,x:x+105] = cv.resize(crop,(105,52),interpolation=cv.INTER_NEAREST)
        when = 'Cue' if index==0 else offset_text(exposure,card.stamp)
        label(canvas,when,x+111,643,.42,MUTED)


def paint_replay(canvas, replay):
    exposure = replay.exposure
    canvas[:,WIDTH:] = PANEL
    label(canvas,replay.title,WIDTH+23,29,.62,TEXT)
    when = 'Cue exposure' if exposure.stamp==replay.card.stamp else offset_text(exposure,replay.card.stamp)
    label(canvas,when,WIDTH+23,59,.50,ACCENT)
    label(canvas,f'Source {exposure.stamp:.3f} s',WIDTH+710,59,.47,MUTED)

    patch,center,fixed = replay_patch(replay.card.exposures,replay.index)
    ih,iw = patch.shape[:2]
    scale = min(866/iw,488/ih)
    w,h = round(iw*scale),round(ih*scale)
    x,y = WIDTH+(WIDTH-w)//2,80+(488-h)//2
    canvas[y:y+h,x:x+w] = cv.resize(patch,(w,h),interpolation=cv.INTER_NEAREST)
    u = min(x+w-1,x+round(center[0]*w/iw))
    v = min(y+h-1,y+round(center[1]*h/ih))
    cv.line(canvas,(u,y-8),(u,y-2),ACCENT,2)
    cv.line(canvas,(u,y+h+2),(u,y+h+8),ACCENT,2)
    cv.line(canvas,(x-8,v),(x-2,v),ACCENT,2)
    cv.line(canvas,(x+w+2,v),(x+w+8,v),ACCENT,2)

    count = len(replay.card.exposures)
    width = tab_width(count)
    for index,item in enumerate(replay.card.exposures):
        left = TABS_X+width*index
        selected = index==replay.index
        color = ACCENT if selected else MUTED
        cv.rectangle(canvas,(left,TABS_Y),(left+width-min(12,width//7),TABS_Y+44),
                     (43,45,43) if selected else (32,29,27),-1)
        title = 'Cue exposure' if item.stamp==replay.card.stamp else f'Witness  {offset_text(item,replay.card.stamp)}'
        if count > 3:
            title = str(index+1)
        label(canvas,title,left+min(17,width//5),TABS_Y+28,.48,color)
    description = 'Fixed camera crop' if fixed else 'Cue-centered camera crop'
    label(canvas,f'{description}    Original pixels enlarged',WIDTH+29,675,.46,MUTED)
    action = 'pause' if replay.playing else 'play'
    label(canvas,f'Space {action}    Arrows step    B return',WIDTH+556,675,.46,MUTED)
