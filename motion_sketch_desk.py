"""Shared Mac field-window renderer for passive motion review."""
from collections import deque
from functools import lru_cache
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from motion_sketch import W, H, measure, full_frame_change
from motion_sketch_history import Trail
from motion_sketch_zoom import enlarge
from motion_sketch_stack import HeldStack
from motion_sketch_cues import ChangeInk
from motion_sketch_light import OverviewLight
from motion_sketch_landscape import LandscapeGrade
from motion_sketch_icons import draw_icon

WIDTH, HEIGHT = 1440, 840
OVERVIEW = (0, 0, 936, 527)
LENS = (940, 0, 500, 281)
SKETCH = (940, 326, 500, 295)
CHANGE = (235, 531, 466, 262)
FOCUS_STAGE = (0, 0, 1440, 794)
PANE_NAMES = {"all": "All views", "overview": "Overview", "zoom": "Zoom",
              "sketch": "Sketch", "change": "Motion Map"}
LOOKS = {"landscape": "Landscape", "source": "Original", "auto": "Auto clear / night", "clear": "Clear",
         "daylight": "Daylight", "haze": "Haze", "neutral": "Neutral",
         "dusk": "Night", "temporal": "Temporal denoise"}


def picture_title(look):
    return LOOKS[look] if look in ("source", "landscape") else f"Landscape + {LOOKS[look]}"

BG=(24,21,19)
SURFACE=(41,36,32)
BORDER=(66,57,51)
FG=(242,240,233)
SECONDARY=(186,178,165)
ACCENT=(197,224,111)
AMBER=(101,170,253)


@lru_cache(maxsize=12)
def font(size, bold):
    directory=Path("/System/Library/Fonts/Supplemental")
    file=directory/("Arial Bold.ttf" if bold else "Arial.ttf")
    try:
        return ImageFont.truetype(str(file),size)
    except OSError:
        return ImageFont.load_default(size=size)


@lru_cache(maxsize=480)
def label_mask(value,size,bold):
    face=font(size,bold)
    box=face.getbbox(value)
    mask=Image.new("L",(max(1,box[2]-min(0,box[0])+2),size+11))
    ImageDraw.Draw(mask).text((0,0),value,font=face,fill=255,stroke_width=0)
    return np.array(mask)


def write(canvas,value,x,y,size=18,color=FG,bold=False):
    mask=label_mask(str(value),size,bold)
    h=min(mask.shape[0],canvas.shape[0]-y)
    w=min(mask.shape[1],canvas.shape[1]-x)
    if min(h,w)<=0:return
    target=canvas[y:y+h,x:x+w]
    a=mask[:h,:w,None].astype(np.float32)/255
    target[:]=(target*(1-a)+np.array(color)*a).astype(np.uint8)


def over_video(canvas, value, x, y, size=15, color=FG):
    # A one-pixel shadow keeps labels legible without an opaque backing.
    write(canvas, value, x+1, y+1, size, (22, 21, 18), True)
    write(canvas, value, x, y, size, color, True)


def box(canvas,rect,color):
    x,y,w,h=rect
    cv2.rectangle(canvas,(x,y),(x+w-1,y+h-1),color,-1)


def put(canvas,rect,image):
    x,y,w,h=rect
    canvas[y:y+h,x:x+w]=cv2.resize(image,(w,h),interpolation=cv2.INTER_AREA if image.shape[1]>w else cv2.INTER_LINEAR)


def in_rect(pt,rect):
    x,y,w,h=rect
    return x<=pt[0]<x+w and y<=pt[1]<y+h


def aspect_fit(bounds, width, height):
    x, y, available_w, available_h = bounds
    scale = min(available_w / width, available_h / height)
    w, h = max(1, round(width * scale)), max(1, round(height * scale))
    return (x + (available_w - w) // 2, y + (available_h - h) // 2, w, h)


@lru_cache(maxsize=400)
def fit_text(value, max_width, size, bold=False):
    face = font(size, bold)
    if face.getlength(value) <= max_width:
        return value
    low, high = 0, len(value)
    while low < high:
        mid = (low + high + 1) // 2
        if face.getlength(value[:mid] + "…") <= max_width:
            low = mid
        else:
            high = mid - 1
    return value[:low].rstrip() + "…"


class Desk:
    """Full-field change display and an explicitly selected source-pixel inspection lens."""
    def __init__(self,device="mps",trail_seconds=20, birds=None):
        self.device=device
        self.trail=Trail(keep_seconds=trail_seconds,max_gap_seconds=.55)
        self.times=deque(maxlen=4)
        self.previous=deque(maxlen=4)
        self.profiles=deque(maxlen=4)
        self.point=None
        self.lens_sizes=(160,96,48)
        self.lens_index=1
        self.clean=False
        self.gen=None
        self.last_t=None
        self.last_frame_key=None
        self.cached=None
        self.reference=None
        self.last_registration=None
        self.trail_reset_at=None
        self.scene_ok=True
        self.last_mark_time=None
        self.mark_session=1
        self.energy=deque(maxlen=1000)
        self.buttons={}
        self.pointer = None
        self.focus = "all"
        self.overview_rect = OVERVIEW
        self.stack_enabled = False
        self.show_best = False
        self.show_verified = False
        self.stack = HeldStack()
        self.birds = birds
        self.bird_state = None
        self.following_bird = False
        self.bird_lost = False
        self.bird_paused = False
        self.ink = ChangeInk()
        self.cues_enabled = True
        self.light = OverviewLight(device=device)
        self.landscape = LandscapeGrade(device=device)
        self._base_key = None
        self._base_picture = None
        self.picture_frame = None
        self.overview_clean_pixels = None
        self.look = "landscape"
        self.split = False
        self.zebras = False
        self.flash = ""
        self.flash_until = 0.
        self.last_warning = ""

    def notice(self, message):
        self.flash = message
        self.flash_until = time.monotonic() + 2.2

    def next_comparison(self):
        if self.stack_enabled:
            self.show_best, self.show_verified = (
                not self.show_best and not self.show_verified, self.show_best)

    def choose_look(self, key):
        if key not in LOOKS:
            raise ValueError(f"Unknown picture mode: {key}")
        self.look = key
        self.clean = False
        if key == "source":
            self.split = False
            self.zebras = False
        if self.focus == "change":
            self.set_focus("overview")
        self.light.reset()
        self.notice(picture_title(key))

    def stop_follow(self, *, lost=False, keep_point=False):
        was_following = self.following_bird
        if self.birds is not None:
            self.birds.clear()
        self.following_bird = False
        self.bird_lost = lost and (was_following or self.bird_lost)
        if was_following and not keep_point:
            self.point = None
            self.current_stroke = self.old_stroke = self.last_frame_key = None
            self.profiles.clear()
        if was_following:
            self.notice("Bird sighting lost" if lost else "")

    def select_bird(self, x, y, stamp):
        # A click belongs to the displayed observation; a stalled view cannot
        # transfer an old animal selection onto a newer frame.
        if (self.birds is None or not self.cues_enabled or self.clean or self.look == "source" or self.bird_paused
                or self.last_t is None or abs(stamp - self.last_t) > .35
                or not self.birds.select(x, y)):
            return False
        if self.stack_enabled:
            self.toggle_stack()
        self.following_bird = True
        self.bird_lost = False
        self.select(x, y)
        self.notice("Following visible bird")
        return True

    def toggle_stack(self):
        if self.following_bird:
            self.stop_follow(keep_point=True)
        self.stack_enabled = not self.stack_enabled
        self.stack.reset()
        self.show_best = False
        self.show_verified = False

    def select(self,x,y):
        point=(int(np.clip(round(x),W//2,1920-W//2)),int(np.clip(round(y),H//2,1080-H//2)))
        if point != self.point and self.stack_enabled:
            self.stack.reset()
        self.point = point

    def observe_mark(self,x,y,t):
        self.select(x,y)
        if self.last_mark_time is not None and (t-self.last_mark_time>.55 or t<self.last_mark_time):
            self.mark_session+=1
        self.trail.update([(self.mark_session,*self.point)],float(t),scene_registered=self.scene_ok)
        self.last_mark_time=t

    def end_stroke(self):
        self.mark_session+=1
        self.last_mark_time=None

    def clear(self, *, lost=False):
        self.trail.reset()
        self.ink.reset()
        self.stop_follow(lost=lost)
        self.last_mark_time=None
        self.end_stroke()

    def set_focus(self, name):
        if name not in PANE_NAMES:
            raise ValueError(f"Unknown view: {name}")
        self.focus = name
        # Coordinates become valid again once the new layout has been drawn.
        self.overview_rect = None
        self.end_stroke()

    def source_point(self, x, y):
        if self.overview_rect is None or not in_rect((x, y), self.overview_rect):
            return None
        left, top, width, height = self.overview_rect
        return ((x - left) * 1919 / max(1, width - 1),
                (y - top) * 1079 / max(1, height - 1))

    def _reset_images(self):
        self.stack.reset()
        self.light.reset()
        self.landscape.reset()
        self._base_key = self._base_picture = None
        self.picture_frame = self.overview_clean_pixels = None
        self.ink.reset()
        if self.birds is not None:
            self.birds.reset()
        self.bird_state = None
        self.following_bird = False
        self.times.clear()
        self.previous.clear()
        self.profiles.clear()
        self.reference=None
        self.last_registration=None
        self.energy.clear()
        self.trail_reset_at=None
        self.cached=None
        self.last_frame_key=None

    def _sample(self,frame,t,generation,paused=False):
        if self.gen!=generation or (self.last_t is not None and (t < self.last_t or t-self.last_t>0.75)):
            if self.gen is not None:self.clear(lost=True)
            self._reset_images()
        self.gen=generation
        fresh=(t!=self.last_t or self.cached is None)
        changed_point=self.last_frame_key!=self.point
        if fresh:
            display=cv2.resize(frame,(1920,1080),interpolation=cv2.INTER_AREA)
            gray=cv2.cvtColor(display,cv2.COLOR_BGR2GRAY)
            gpu=torch.from_numpy(gray).to(self.device)
            change=full_frame_change(gpu,self.previous[0]) if len(self.previous)==4 else None
            self.previous.append(gpu)
            self.times.append(t)
            self.cached=(display,gray,change)
            if change is not None and (not self.energy or t-self.energy[-1][0]>.09):
                self.energy.append((t,float(np.mean(change[:,:,0]))))
            while self.energy and t-self.energy[0][0]>30:self.energy.popleft()
            if self.last_registration is None or t-self.last_registration>.24:
                small=cv2.resize(gray,(240,135),interpolation=cv2.INTER_AREA).astype(np.float32)
                if self.reference is None:
                    self.reference=small
                    self.scene_ok=True
                else:
                    shift,response=cv2.phaseCorrelate(self.reference,small)
                    self.scene_ok=bool(response>.17 and np.hypot(*shift)<2.1)
                    if not self.scene_ok:
                        if self.trail.tracks:
                            self.trail_reset_at = t
                        self.clear(lost=True)
                        if self.birds is not None:
                            self.birds.reset()
                        self.bird_state = None
                        self.reference=small
                self.last_registration=t
            self.ink.update(change, t, self.scene_ok and self.cues_enabled)
        if self.birds is not None:
            if paused != self.bird_paused:
                self.stop_follow(lost=True)
                self.birds.reset()
                self.bird_state = None
            self.bird_paused = paused
            if self.cues_enabled and not paused and self.scene_ok:
                self.bird_state = self.birds.update(frame, t, generation)
                if self.following_bird:
                    if self.bird_state.selected is None:
                        if not self.bird_lost:
                            self.notice("Bird sighting lost")
                        self.bird_lost = True
                        self.point = None
                        self.current_stroke = self.old_stroke = None
                    else:
                        self.bird_lost = False
                        self.select(*self.bird_state.selected.center)
                changed_point = self.last_frame_key != self.point
        self.last_t=t
        display,gray,change=self.cached
        if self.point and (changed_point or fresh):
            x,y=self.point
            tile=gray[y-H//2:y+H//2,x-W//2:x+W//2]
            current=measure(tile,self.device)
            if changed_point and self.last_frame_key is not None and np.hypot(x-self.last_frame_key[0],y-self.last_frame_key[1])>W/2:
                self.profiles.clear()
            self.old_stroke=self.profiles[0] if len(self.profiles)==4 else None
            self.current_stroke=current
            if fresh:self.profiles.append(current)
        self.last_frame_key=self.point
        return display,change

    def _lens(self,frame):
        sh,sw=frame.shape[:2];x,y=self.point
        side=self.lens_sizes[self.lens_index]
        width=min(sw,round(side*sw/1920))
        height=min(sh,max(2,round(width*9/16)))
        a=int(np.clip(round(x*sw/1920-width/2),0,sw-width))
        b=int(np.clip(round(y*sh/1080-height/2),0,sh-height))
        patch=frame[b:b+height,a:a+width]
        return patch, width

    def _picture(self, frame, t, paused):
        """Prepare one full-scene photo for all visible photographic panes."""
        self.last_warning = ""
        if self.clean or self.look == "source":
            return frame
        # Pausing can replace a live observation with a JPEG replay of the
        # same timestamp. Repeated redraws of that replay share its cached grade.
        key = (self.gen, t, paused, frame.shape)
        if key != self._base_key:
            if self._base_key is not None and (paused != self._base_key[2] or frame.shape != self._base_key[3]):
                self.landscape.reset()
                self.light.reset()
            self._base_picture = self.landscape.apply(frame, t)
            self._base_key = key
        if self.look == "landscape":
            return self._base_picture
        result, self.last_warning = self.light.apply(self._base_picture, t, self.look)
        return result

    @property
    def active_look(self):
        if self.picture_frame is None:
            return None
        if self.clean or self.look == "source":
            return "source"
        return self.light.active if self.look == "auto" else self.look

    def _sketch_tile(self, picture):
        """Sample the photo beneath the sketch on the same 1080p measurement grid."""
        x, y = self.point
        h, w = picture.shape[:2]
        left = min(w-1,round((x-W//2)*w/1920))
        top = min(h-1,round((y-H//2)*h/1080))
        right = max(left+1, round((x+W//2)*w/1920))
        bottom = max(top+1, round((y+H//2)*h/1080))
        tile = picture[top:bottom,left:right]
        return tile if tile.shape[:2] == (H,W) else cv2.resize(tile, (W,H), interpolation=cv2.INTER_AREA)

    def _sketch(self, canvas, display, rect):
        raw = self._sketch_tile(display)
        w, h = rect[2:]
        back = cv2.resize((raw * .46).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        current, old = self.current_stroke, self.old_stroke

        def draw_series(points, color, width):
            xy = np.rint((points + .5) * [w / W, h / H]).astype(int)
            for a, b in zip(xy, xy[1:]):
                if abs(b[1] - a[1]) <= 3.1 * h / H:
                    cv2.line(back, tuple(a), tuple(b), color, width, cv2.LINE_AA)

        if current is not None:
            if old is not None:
                draw_series(old + np.mean(current, axis=0) - np.mean(old, axis=0),
                            AMBER, max(2, round(w / 245)))
            draw_series(current, ACCENT, max(3, round(w / 123)))
        left, top = rect[:2]
        canvas[top:top+h, left:left+w] = back

    def _button(self, canvas, key, title, rect, *, active=False, enabled=True, size=17):
        x, y, w, h = rect
        if enabled:
            self.buttons[key] = rect
        color = ((69, 75, 55) if active else SURFACE) if enabled else (33, 29, 26)
        box(canvas, rect, color)
        ink = (ACCENT if active else FG) if enabled else (134, 125, 111)
        mask = label_mask(title, size, active)
        write(canvas, title, x + max(3, (w - mask.shape[1]) // 2),
              y + (h - size) // 2 - 3, size, ink, active)

    def hover(self, x, y):
        self.pointer = (x, y)

    def _icon_button(self, canvas, key, rect, *, enabled=True, paused=False):
        x, y, w, h = rect
        active = key == "pause" and paused
        self._button(canvas, key, "", rect, enabled=enabled, active=active)
        cx, cy = x + w // 2, y + h // 2
        color = (ACCENT if active else FG) if enabled else (134, 125, 111)
        def path(points, *, closed=False):
            pts = np.int32([(cx+dx, cy+dy) for dx, dy in points])
            cv2.polylines(canvas, [pts], closed, color, 2, cv2.LINE_AA)
        if key == "pause" and not paused:
            for dx in (-6, 3):
                cv2.rectangle(canvas, (cx+dx,cy-8), (cx+dx+3,cy+8), color, -1)
        elif key in {"pause", "live"}:
            pts = np.int32([(cx-7,cy-8), (cx+5,cy), (cx-7,cy+8)])
            cv2.fillConvexPoly(canvas, pts, color, cv2.LINE_AA)
            if key == "live":
                path([(8,-8), (8,8)])
        elif key == "clear":
            path([(-9,2),(1,-9),(9,-2),(0,8),(-3,8)], closed=True)
            path([(-5,-2),(3,5)])
            path([(0,8),(10,8)])
        elif key == "close":
            path([(-7,-7),(7,7)])
            path([(-7,7),(7,-7)])

    def _controls(self, canvas, *, paused, has_frame, age=0, live=False, status=""):
        x = 5
        for name, width in zip(PANE_NAMES, (85, 91, 69, 75, 106)):
            key = "view_" + name
            active = self.focus == name
            self._button(canvas, key, "", (x,799,width,37), active=active)
            draw_icon(canvas, key, x+13, 817, ACCENT if active else FG)
            write(canvas, PANE_NAMES[name], x+26, 805, 14, ACCENT if active else FG, active)
            x += width + 2
        x += 5
        selected_look = "source" if self.clean else self.look
        for key in (*LOOKS, "split", "zebras", "cues"):
            action = "look_" + key if key in LOOKS else key
            active = key == selected_look if key in LOOKS else (
                selected_look != "source" and {"split": self.split, "zebras": self.zebras,
                                               "cues": self.cues_enabled}[key])
            self._button(canvas,action,"",(x,799,30,37),active=active)
            draw_icon(canvas,key,x+15,817,ACCENT if active else FG)
            x += 32
        x += 5
        options = (
            ("pause", None, 34, has_frame), ("back", None, 32, has_frame),
            ("forward", None, 32, has_frame), ("live", None, 34, has_frame),
            ("clear", None, 34, has_frame), ("capture", None, 36, has_frame),
            ("zoom", "Lens +", 58, True), ("trail", f"Trail {self.trail.keep_seconds}s", 77, True),
            ("stack", "End hold" if self.stack_enabled else "Hold + stack", 99, True),
            ("best", "Best stack" if self.show_best else "Live stack" if self.show_verified
             else "Best single", 88, self.stack_enabled),
        )
        for key, title, width, enabled in options:
            rect = (x, 799, width, 37)
            if key in ("back","forward","capture"):
                self._button(canvas,key,"",rect,enabled=enabled)
                draw_icon(canvas,key,x+width//2,817,FG if enabled else SECONDARY)
            elif title is None:
                self._icon_button(canvas, key, rect, enabled=enabled, paused=paused)
            else:
                active = (key == "stack" and self.stack_enabled) or (key == "best" and (self.show_best or self.show_verified))
                self._button(canvas,key,title,rect,active=active,enabled=enabled,size=14)
            x += width + 2
        self._icon_button(canvas, "close", (1390,799,42,37))
        hovered = None if self.pointer is None else next(
            (key for key, rect in self.buttons.items() if in_rect(self.pointer,rect)), None)
        tips = {"pause": "Resume (Space)" if paused else "Pause (Space)",
                "live": "Return to live (L)" if live else "Latest frame (L)",
                "clear": "Clear trail (C)", "close": "Close (Esc)",
                "back": "Back one second ([)", "forward":"Forward one second (])",
                "zoom":"Change zoom (Z)", "trail":"Change trail length (T)",
                "stack":"End hold (S)" if self.stack_enabled else "Hold this crop (S)",
                "best":"Compare live stack, best exposure and verified stack (B)",
                "capture":"Save source + viewer (P)", "split":"Compare original and adjusted picture",
                "zebras":"Stripes on clipped source highlights", "cues":"Motion highlights and bird cues"}
        tips.update({"look_"+name: picture_title(name) for name in LOOKS})
        for i, name in enumerate(PANE_NAMES):
            tips["view_"+name] = f"{PANE_NAMES[name]} ({i})"
        message = (self.flash if time.monotonic() < self.flash_until else tips.get(hovered)) or (
            f"Paused · {abs(age):.1f}s behind" if paused else status)
        if message and not ((self.clean or self.look == "source") and self.focus == "overview"):
            size = label_mask(message, 15, False).shape[1]
            center = (self.buttons[hovered][0] + self.buttons[hovered][2]//2) if hovered else 720
            min_left = WIDTH//2 + 2 if self.focus == "overview" and self.split else 4
            left = int(np.clip(center-size//2-8, min_left, WIDTH-size-20))
            box(canvas,(left,766,size+16,28),SURFACE)
            write(canvas,message,left+8,768,15,AMBER if paused else FG)

    @staticmethod
    def _empty(canvas, rect, title):
        box(canvas, rect, SURFACE)
        x, y, w, h = rect
        width = label_mask(title, 18, False).shape[1]
        write(canvas, title, x + max(4, (w - width) // 2), y + (h - 24) // 2, 18, SECONDARY)

    def _motion_trace(self, canvas, rect, t):
        if len(self.energy) < 2:
            return
        left, top, width, height = rect
        chart_w, chart_h = min(216, width // 3), min(40, height // 6)
        x, baseline = left + 10, top + height - 10
        pts = []
        for stamp, value in self.energy:
            if 0 <= t - stamp <= 20:
                px = x + chart_w - 1 - round((t - stamp) / 20 * (chart_w - 1))
                pts.append((px, baseline - round(min(1, value / 85) * chart_h)))
        if len(pts) >= 2:
            trace = [np.int32(pts)]
            cv2.polylines(canvas, trace, False, (34,35,31), 3, cv2.LINE_AA)
            cv2.polylines(canvas, trace, False, ACCENT, 1, cv2.LINE_AA)
            over_video(canvas, "20 s", x+chart_w+4, baseline-12, 12, ACCENT)

    def _overview(self, canvas, picture, rect, t, source):
        put(canvas, rect, picture)
        left,top,width,height = rect
        pixels = canvas[top:top+height,left:left+width]
        original_view = self.clean or self.look == "source"
        middle = width // 2 if self.split and not original_view else 0
        if middle:
            interp = cv2.INTER_AREA if source.shape[1] > width else cv2.INTER_LINEAR
            pixels[:,:middle] = cv2.resize(source,(width,height),interpolation=interp)[:,:middle]
        # The Retina presenter uses this logical photo to identify where HUD
        # pixels overlap the higher-resolution photographic layer.
        self.overview_clean_pixels = pixels.copy()
        if original_view:
            return canvas
        if self.zebras:
            # Find clipped channels on the native grid before downsampling can
            # average away a small bright patch in the operator's overview.
            mask = cv2.bitwise_not(cv2.inRange(source,(0,0,0),(254,254,254)))
            clipped = cv2.resize(mask,(width,height),interpolation=cv2.INTER_AREA) > 0
            yy,xx = np.ogrid[:height,:width]
            stripes = clipped & ((xx + yy) % 10 < 3)
        if middle:
            cv2.line(pixels,(middle,0),(middle,height-1),FG,1)
            over_video(pixels,"Original | " + picture_title(self.look),middle+6,5)
        if self.zebras:
            pixels[stripes] = (55,170,255)
        if self.cues_enabled:
            self.ink.draw(pixels)
        if self.last_warning:
            over_video(pixels,self.last_warning,middle+7,28 if middle else 5,14,AMBER)
        elif self.look == "auto":
            over_video(pixels,"Auto · " + LOOKS.get(self.light.active,self.light.active),
                       middle+7,28 if middle else 5,14)
        if self.cues_enabled and self.bird_state is not None and self.bird_state.status == "Bird model unavailable":
            over_video(pixels,"Bird cues unavailable",middle+7,52,14,AMBER)
        canvas = self.trail.draw(canvas, rect, (1920, 1080), t, scene_registered=self.scene_ok)
        if self.cues_enabled and self.birds is not None and self.bird_state is not None:
            self.birds.trail.keep_seconds = self.trail.keep_seconds
            canvas = self.birds.trail.draw(canvas, rect, (1920,1080), t, scene_registered=self.scene_ok)
            for cue in self.bird_state.cues:
                cx,cy = cue.center
                screen = (left+round(cx*(width-1)/1919), top+round(cy*(height-1)/1079))
                selected = self.bird_state.selected is not None and self.bird_state.selected.key == cue.key
                cv2.circle(canvas,screen,9 if selected else 7,ACCENT,2 if selected else 1,cv2.LINE_AA)
        self._motion_trace(canvas, (left+middle,top,width-middle,height), t)
        if self.point:
            x, y = self.point
            left, top, width, height = rect
            center = (left + round(x * (width - 1) / 1919), top + round(y * (height - 1) / 1079))
            cv2.drawMarker(canvas, center, AMBER, cv2.MARKER_CROSS, 13, 1, cv2.LINE_AA)
        if middle:
            # Retain a truly original comparison, including when the operator
            # enables clipping stripes, motion tint, marked paths or labels.
            canvas[top:top+height,left:left+middle] = self.overview_clean_pixels[:,:middle]
        return canvas

    def _zoom(self, canvas, rect, raw, fused):
        if not self.stack_enabled:
            put(canvas, rect, raw if self.clean or self.look == "source" else enlarge(raw))
            return
        best = self.show_best and self.stack.best is not None
        candidate = self.stack.best if best else fused
        if self.show_verified:
            candidate = self.stack.verified_best
            if candidate is not None:
                raw = self.stack.verified_source
        left, top, w, h = rect
        half = w // 2
        start = raw.shape[1] // 4
        stop = start + raw.shape[1] // 2
        for x, width, image in ((left, half, raw), (left + half, w - half, candidate)):
            if image is None:
                self._empty(canvas,(x,top,width,h),"Gathering independent frames")
                continue
            canvas[top:top+h, x:x+width] = cv2.resize(
                image[:, start:stop], (width, h), interpolation=cv2.INTER_CUBIC)
        cv2.line(canvas, (left + half, top), (left + half, top + h - 1), BORDER, 1)

    def _change(self, canvas, change, rect):
        if change is None:
            self._empty(canvas, rect, "Gathering frames")
        else:
            put(canvas, rect, change)

    def render(self, frame, t, mode="FILE REPLAY", source_name="", generation=1, paused=False,
               age=0, buffer_seconds=0, note=""):
        t = float(t)
        _, change = self._sample(frame, t, generation, paused)
        canvas = np.full((HEIGHT, WIDTH, 3), BG, np.uint8)
        self.buttons.clear()
        multi = self.focus == "all"
        self.overview_rect = None
        self.overview_clean_pixels = None
        show_overview = multi or self.focus == "overview"
        show_zoom = multi or self.focus == "zoom"
        show_sketch = multi or self.focus == "sketch"
        needs_picture = show_overview or (self.point is not None and (show_sketch or (show_zoom and not self.stack_enabled)))
        self.picture_frame = self._picture(frame,t,paused) if needs_picture else None

        if show_overview:
            stage = OVERVIEW if multi else FOCUS_STAGE
            self.overview_rect = aspect_fit(stage, frame.shape[1], frame.shape[0])
            canvas = self._overview(canvas, self.picture_frame, self.overview_rect, t, source=frame)

        raw = fused = None
        if self.point and (show_zoom or self.stack_enabled):
            # HeldStack always receives the decoded crop, including its paired
            # source and verified exposure. Ordinary zoom uses the chosen look.
            raw, source_width = self._lens(frame if self.stack_enabled else self.picture_frame)
            if self.stack_enabled:
                # A hold continues gathering while the operator looks at another pane.
                fused = self.stack.push(raw, t)
        if show_zoom:
            rect = LENS if multi else aspect_fit(FOCUS_STAGE, 16, 9)
            x, top, w, h = rect
            if raw is None:
                self._empty(canvas, rect, "Bird no longer visible" if self.bird_lost else "Choose a point in Overview")
                if multi:
                    write(canvas, "Zoom", x, top+h+2, 16, FG, True)
            else:
                self._zoom(canvas, rect, raw, fused)
                if self.stack_enabled:
                    elapsed = 0 if self.stack.started is None else max(0, t - self.stack.started)
                    detail = f"{elapsed:.0f}s held  ·  {self.stack.seen} examined  ·  {self.stack.status}"
                    if self.show_best and self.stack.best_stamp is not None:
                        detail = f"Best exposure  ·  {max(0, t - self.stack.best_stamp):.1f}s ago"
                    if self.show_verified and self.stack.verified_best_stamp is not None:
                        detail = f"Verified stack  ·  {max(0,t-self.stack.verified_best_stamp):.1f}s ago"
                else:
                    detail = "Following visible bird" if self.following_bird else f"{source_width} source pixels across"
                compare = "Verified stack" if self.show_verified else "Best exposure" if self.show_best else "Stack"
                source_label = "Paired source" if self.show_verified and self.stack.verified_best is not None else "Source"
                if not multi and self.stack_enabled:
                    over_video(canvas, source_label, x+8, top+h-25)
                    over_video(canvas, compare, x+w//2+8, top+h-25, color=ACCENT)
                if multi:
                    if self.stack_enabled:
                        write(canvas, source_label, x, top+h+2, 15, SECONDARY)
                        write(canvas, compare, x+w//2+2, top+h+2, 15, ACCENT)
                        write(canvas, detail, x, top+h+23, 15, SECONDARY)
                    else:
                        write(canvas, "Zoom", x, top+h+2, 16, FG, True)
                        write(canvas, detail, x+64, top+h+2, 15, SECONDARY)

        if show_sketch:
            rect = SKETCH if multi else aspect_fit(FOCUS_STAGE, W, H)
            x, y, w, h = rect
            if self.point is None:
                self._empty(canvas, rect, "Bird no longer visible" if self.bird_lost else "Choose a point in Overview")
                if multi:
                    write(canvas, "Shape sketch", x, y+h+2, 16, FG, True)
            else:
                self._sketch(canvas, self.picture_frame, rect)
                if multi:
                    write(canvas, "Shape sketch", x, y+h+2, 16, FG, True)
                    write(canvas, "Current", x+128, y+h+3, 15, ACCENT)
                    write(canvas, "Earlier", x+199, y+h+3, 15, AMBER)
                    if self.current_stroke is None:
                        write(canvas, "No clear stroke", x+282, y+h+3, 15, SECONDARY)

        if multi or self.focus == "change":
            rect = CHANGE if multi else aspect_fit(FOCUS_STAGE, 16, 9)
            self._change(canvas, change, rect)
            if multi:
                over_video(canvas, "Motion Map", rect[0]+7, rect[1]+4)
        status = ""
        if self.focus == "zoom" and raw is not None:
            status = detail
        if self.trail_reset_at is not None and 0 <= t - self.trail_reset_at <= 1.6:
            status = "View shifted · path cleared"
        self._controls(canvas, paused=paused, has_frame=True, age=age, live=mode=="LIVE", status=status)
        return canvas

    def render_waiting(self, state, source_name, *, mode="LIVE"):
        if self.last_t is not None:
            self.clear(lost=True)
            self._reset_images()
            self.point = None
            self.last_t = None
        self.buttons.clear()
        self.overview_rect = None
        canvas = np.full((HEIGHT, WIDTH, 3), BG, np.uint8)
        self._empty(canvas, FOCUS_STAGE, state.capitalize())
        self._controls(canvas, paused=False, has_frame=False)
        return canvas
