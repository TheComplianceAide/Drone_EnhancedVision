"""Draw changes supported by two earlier surface-aligned source crops."""

import cv2 as cv
import numpy as np


LUMA_COLOR=(56.,164.,246.)
CHROMA_COLOR=(179.,228.,77.)


def smoothed_channels(frame):
    samples=cv.cvtColor(frame,cv.COLOR_BGR2YCrCb).astype(np.float32)
    return cv.GaussianBlur(samples,(3,3),.68)


def gradient(channel):
    dx=cv.Sobel(channel,cv.CV_32F,1,0,scale=.125)
    dy=cv.Sobel(channel,cv.CV_32F,0,1,scale=.125)
    return cv.magnitude(dx,dy)


def chroma_edges(image):
    return cv.magnitude(gradient(image[:,:,1]),gradient(image[:,:,2]))


def edge_gain(current,past,kernel):
    observed=(current-past)[::3,::3]
    noise=max(1.,1.4826*float(np.median(abs(observed-np.median(observed)))))
    return np.maximum(0.,current-cv.dilate(past,kernel)-max(1.7,noise))/noise


def measurements(selected,earlier):
    now=smoothed_channels(selected)
    y=gradient(now[:,:,0])
    c=chroma_edges(now)
    brightness=[]
    color=[]
    for original,error in earlier:
        before=smoothed_channels(original)
        radius=int(np.clip(np.ceil(error),1,3))
        radius_size=2*radius+1
        kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE,(radius_size,radius_size))
        brightness.append(edge_gain(y,gradient(before[:,:,0]),kernel))
        color.append(edge_gain(c,chroma_edges(before),kernel))
    luma=np.minimum.reduce(brightness)
    chroma=np.minimum.reduce(color)
    # Exclude thin crop borders that do not have a complete local witness.
    for signal in (luma,chroma):
        signal[:8]=signal[-8:]=0.
        signal[:,:8]=signal[:,-8:]=0.
    return luma,chroma


def display_change(selected,earlier):
    luma,chroma=measurements(selected,earlier)
    a=np.clip(luma-1.5,0.,5.)*.11
    b=np.clip(chroma-1.5,0.,5.)*.11
    opacity=np.maximum(a,b)[:,:,None]
    color=np.where((b>a)[:,:,None],CHROMA_COLOR,LUMA_COLOR)
    painted=selected.astype(np.float32)*(1.-opacity)+color*opacity
    return np.uint8(np.rint(painted))


def references(stamps,checked,index):
    """Pick a separate verified reference close enough to limit extrapolation."""
    current=stamps[-1]
    if index>=len(stamps)-1 or checked[index] is None or not .04<=current-stamps[index]<=.29:
        return ()
    reference_lag=current-stamps[index]
    alternatives=[i for i,fit in enumerate(checked[:-1]) if fit is not None and i!=index
                  and .04<=current-stamps[i]<=.29 and abs(stamps[i]-stamps[index])>=.030]
    if not alternatives:
        return ()
    # Favor a genuinely different gap while using only measured exposures.
    desired=.20 if reference_lag<.15 else .10
    other=min(alternatives,key=lambda i:abs(current-stamps[i]-desired))
    return index,other
