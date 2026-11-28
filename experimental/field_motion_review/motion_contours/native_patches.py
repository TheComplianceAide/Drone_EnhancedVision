"""Check raw, camera-relative patch correspondences at two earlier times."""
import cv2
import numpy as np

from auto_tiles import anchors, checked_pose, grayscale, native_warp


def _detail(luma):
    values = luma.astype(np.float32)
    return values-cv2.blur(values,(19,19))


def _match(pattern, picture, x, y, reach):
    half = pattern.shape[0]//2
    target = picture[y-reach-half:y+reach+half+1,x-reach-half:x+reach+half+1]
    similarity = cv2.matchTemplate(target,pattern,cv2.TM_CCOEFF_NORMED)
    _,peak,_,(col,row) = cv2.minMaxLoc(similarity)
    return (x+col-reach,y+row-reach), float(peak), float(similarity[reach,reach])


def _error_at(now, prior, x,y,dx,dy,size,mask=None):
    a = now[y-size:y+size+1,x-size:x+size+1]
    b = prior[y+dy-size:y+dy+size+1,x+dx-size:x+dx+size+1]
    difference = np.abs(a-b)
    return float(np.mean(difference[mask] if mask is not None else difference))


def checked_patch(current, near, far, x, y, first_age, last_age, rim, energy):
    template=current[y-6:y+7,x-6:x+7]
    if np.std(template)<2.2:
        return None
    near_center,first_corr,first_static=_match(template,near,x,y,24)
    far_center,far_corr,far_static=_match(template,far,x,y,43)
    first = np.asarray(near_center)-[x,y]
    second = np.asarray(far_center)-[x,y]
    norm = np.linalg.norm(first)
    if min(first_corr,far_corr)<.78 or min(first_corr-first_static,far_corr-far_static)<.13:
        return None
    if norm<2. or np.linalg.norm(second-first*(last_age/first_age))>2.25+.19*np.linalg.norm(second):
        return None
    nx,ny=near_center
    expected,back_corr,_=_match(near[ny-6:ny+7,nx-6:nx+7],current,nx,ny,24)
    if np.linalg.norm(np.asarray(expected)-[x,y])>2.0 or back_corr<.76:
        return None
    dx,dy=map(int,first)
    gain=_error_at(current,near,x,y,0,0,6)-_error_at(current,near,x,y,dx,dy,6)
    background_gain=_error_at(current,near,x,y,0,0,26,rim)-_error_at(current,near,x,y,dx,dy,26,rim)
    if gain<1.35 or background_gain>gain*.52:
        return None
    return {'xy':(x,y),'box':[x-8,y-8,16,16],
            'near':near_center,'far':far_center,'score':gain,'peak_dn':energy}


def quieter_peak(current,near,far,raw,raw_proposals,regions,first_age,last_age,rim,camera_error):
    """Keep a small verification budget for peaks outside established clutter."""
    history=near-far
    clutter=cv2.blur(history*history,(31,31))
    gx=cv2.Sobel(near,cv2.CV_32F,1,0,scale=.125)
    gy=cv2.Sobel(near,cv2.CV_32F,0,1,scale=.125)
    slope=cv2.blur(cv2.magnitude(gx,gy),(7,7))*max(.2,2*camera_error)
    score=raw/np.sqrt(2.25+.75*clutter+slope*slope)
    score[raw<2.8]=0.
    for x,y in raw_proposals:
        score[y-1:y+2,x-1:x+2]=0.
    for region in regions:
        x,y=region['xy']
        score[y-12:y+13,x-12:x+13]=0.
    for _ in range(2):
        _,strength,_,(x,y)=cv2.minMaxLoc(score)
        if strength<=0.:
            return None
        score[y-11:y+12,x-11:x+12]=0.
        region=checked_patch(current,near,far,x,y,first_age,last_age,rim,float(raw[y,x]))
        if region is not None:
            return region
    return None


def native_patches(frame,past,old,first_age,last_age,*,clutter_reserve=False):
    gray = [grayscale(image) for image in (frame,past,old)]
    if np.quantile(gray[0][::9,::9],.90)<4:
        return [],{'camera':'No visible signal'}
    guide = [cv2.resize(image,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA) for image in gray]
    points = anchors(guide[0])
    camera = [checked_pose(guide[0],earlier,points) for earlier in guide[1:]]
    if any(m is None for m in camera):
        return [],{'camera':'Uncertain'}
    current = _detail(gray[0])
    near,far = [_detail(native_warp(basis,model[0])) for basis,model in zip(gray[1:],camera)]
    score = np.abs(cv2.blur(current-near,(3,3)))
    score[:60]=score[-60:]=0.
    score[:,:60]=score[:,-60:]=0.
    raw = score.copy() if clutter_reserve else None
    rim = np.ones((53,53),bool)
    rim[10:-10,10:-10]=False
    regions = []
    raw_proposals = []
    for _ in range(24):
        _,energy,_,(x,y)=cv2.minMaxLoc(score)
        if energy<2.8:
            break
        score[y-11:y+12,x-11:x+12]=0.
        if clutter_reserve:
            raw_proposals.append((x,y))
        region = checked_patch(current,near,far,x,y,first_age,last_age,rim,energy)
        if region is None:
            continue
        if any(np.linalg.norm(np.asarray(item['xy'])-[x,y])<=12 for item in regions):
            continue
        regions.append(region)
        if len(regions)>=6:
            break
    details = {'camera':camera[0][1]['camera']}
    if raw_proposals:
        reserve = quieter_peak(current,near,far,raw,raw_proposals,regions,first_age,last_age,rim,
                               camera[0][1]['p67_guide_px'])
        if reserve is not None:
            details['clutter_region'] = reserve
    return regions,details
