"""Measure motion against a checked camera model and the source appearance."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Support:
    regions: list
    mask: np.ndarray
    camera: dict


def magnitude(vectors):
    return cv2.magnitude(np.ascontiguousarray(vectors[..., 0]), np.ascontiguousarray(vectors[..., 1]))


def fit_camera(grid, destination, eligible):
    height, width = eligible.shape
    xs = np.arange(12, width - 12, 8)
    ys = np.arange(12, height - 12, 8)
    y, x = np.meshgrid(ys, xs, indexing='ij')
    x, y = x.ravel(), y.ravel()
    allowed = eligible[y, x]
    x, y = x[allowed], y[allowed]
    tiles = (x * 6 // width) + 6 * (y * 4 // height)
    occupied = np.unique(tiles)
    if len(x) < 140 or occupied.size < 12:
        return None
    checker = (((x - 12) // 8) + ((y - 12) // 8)) % 2 == 0
    current, old = grid[y, x], destination[y, x]
    fit_a, _ = cv2.estimateAffine2D(current[checker], old[checker], method=cv2.RANSAC,
                                    ransacReprojThreshold=1.6, maxIters=800, refineIters=5)
    fit_h, _ = cv2.findHomography(current[checker], old[checker], cv2.USAC_MAGSAC, 1.6,
                                  maxIters=750, confidence=.995)
    models = {}
    if fit_a is not None:
        models['affine'] = np.vstack([fit_a, [0., 0., 1.]])
    if fit_h is not None:
        models['projective'] = fit_h
    checked = {}
    for name, model in models.items():
        predicted = cv2.perspectiveTransform(current[~checker, None], model)[:, 0]
        error = magnitude(predicted - old[~checker]).ravel()
        tile_matches = np.bincount(tiles[~checker][error <= 1.8], minlength=24)
        supported = float(np.mean(error <= 1.8))
        if np.count_nonzero(tile_matches >= 3) < 10 or supported < .53:
            continue
        checked[name] = (float(np.median(error)), supported, model, error)
    if not checked:
        return None
    best = 'affine' if 'affine' in checked else 'projective'
    if best == 'affine' and 'projective' in checked and checked['projective'][0] + .12 < checked['affine'][0]:
        best = 'projective'
    median, fraction, model, error = checked[best]
    predicted = cv2.perspectiveTransform(grid.reshape(-1, 1, 2), model).reshape(grid.shape)
    details = {'geometry': best, 'held_pixels': len(error), 'held_near_model': fraction,
               'median_error': median, 'held_p90': float(np.quantile(error, .9))}
    return predicted, details


def source_residual(present, prior, destination):
    remapped = cv2.remap(prior, destination, None, cv2.INTER_LINEAR)
    return cv2.absdiff(present, remapped)


def movement(current, earlier, forward, backward, *, overlap=None, checked_reference_error=None,
             outlines=False, export_support=False):
    """Return support at current-image coordinates from two-direction flow."""
    h, w = current.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY).astype(np.float32)
    earlier_gray = cv2.cvtColor(earlier, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if max(np.percentile(current_gray[::8, ::8], 90), np.percentile(earlier_gray[::8, ::8], 90)) < 4:
        return Support([], mask, {'geometry': 'Insufficient visible signal'})
    if abs(float(np.median(current_gray[::8, ::8])-np.median(earlier_gray[::8, ::8]))) > 23:
        return Support([], mask, {'geometry': 'Exposure or scene change'})
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    grid = np.dstack((xx, yy))
    destination = grid + backward
    recovered = cv2.remap(forward, destination, None, cv2.INTER_LINEAR)
    cycle = magnitude(backward + recovered)
    good_cycle = cycle <= (.9 + .065 * magnitude(backward))
    good_cycle &= ((destination[..., 0] >= 5) & (destination[..., 1] >= 5)
                   & (destination[..., 0] < w - 5) & (destination[..., 1] < h - 5))
    if overlap is not None:
        before = cv2.remap(np.uint8(overlap), destination, None, cv2.INTER_NEAREST) > 0
        good_cycle &= before & overlap
    current_detail = current_gray - cv2.blur(current_gray, (15, 15))
    old_detail = earlier_gray - cv2.blur(earlier_gray, (15, 15))
    texture = cv2.blur(np.abs(current_detail), (7, 7))
    case = fit_camera(grid, destination, good_cycle & (texture >= 2.3))
    if case is None and overlap is not None and checked_reference_error is not None:
        case = (grid, {'geometry': 'independent descriptor reference',
                       'median_error': checked_reference_error})
    if case is None:
        return Support([], mask, {'geometry': 'Camera geometry uncertain',
                                  'cycle_valid': float(np.mean(good_cycle))})
    prediction, details = case
    residual = destination - prediction
    distance = magnitude(residual)
    residual_floor = max(1.7, details['median_error'] * 3.8)
    camera_error = cv2.blur(source_residual(current_detail, old_detail, prediction), (5, 5))
    motion_error = cv2.blur(source_residual(current_detail, old_detail, destination), (5, 5))
    improved = camera_error - motion_error
    valid_background = ((prediction[..., 0] > 4) & (prediction[..., 1] > 4)
                        & (prediction[..., 0] < w - 5) & (prediction[..., 1] < h - 5))
    if overlap is not None:
        valid_background &= cv2.remap(np.uint8(overlap), prediction, None, cv2.INTER_NEAREST) > 0
    tested = good_cycle & valid_background & (distance >= residual_floor) & (texture >= 2.0)
    tested &= (improved >= 1.15) & (motion_error <= camera_error * .76)
    bridge = cv2.morphologyEx(np.uint8(tested), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bridge)
    regions = []
    for ident in range(1, count):
        x, y, ww, hh, area = map(int, stats[ident])
        if area < 11 or area > h * w * .065 or min(ww, hh) < 3:
            continue
        roi = np.s_[y:y+hh, x:x+ww]
        selected = (labels[roi] == ident) & tested[roi]
        vectors = residual[roi][selected]
        if len(vectors) < 9:
            continue
        direction = np.median(vectors, axis=0)
        length = float(np.linalg.norm(direction))
        agreement = float(np.mean((vectors @ direction) >= .65 * length * np.linalg.norm(vectors, axis=1)))
        if length < residual_floor or agreement < .7:
            continue
        details_y, details_x = np.nonzero(selected)
        mask[roi] |= np.uint8(selected)
        rank = float(np.median(improved[roi][selected])) * np.sqrt(len(vectors))
        region={'center': [float(x + details_x.mean()), float(y + details_y.mean())],
                'bounds_xywh': [x, y, ww, hh], 'source_pixels': len(vectors), 'score': rank, 'motion_px': length}
        region['displacement'] = tuple(float(-v) for v in direction)
        if export_support:
            region['support_xy'] = np.column_stack((details_x+x,details_y+y))
        if outlines:
            contours,_=cv2.findContours(np.uint8(selected),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            most_supported=sorted(contours,key=cv2.contourArea,reverse=True)[:4]
            region['outlines']=[cv2.approxPolyDP(edge,.75,True)[:,0]+(x,y)
                                for edge in most_supported if cv2.contourArea(edge)>=2.]
        regions.append(region)
    regions.sort(key=lambda row: row['score'], reverse=True)
    details.update(cycle_valid=float(np.mean(good_cycle)), min_motion_px=residual_floor)
    return Support(regions, mask, details)
