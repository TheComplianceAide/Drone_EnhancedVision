"""Show estimated animal joints on the exact source exposure used by Vision."""
from base64 import b64encode
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import select
import subprocess
from tempfile import TemporaryDirectory

import cv2 as cv
import numpy as np

from turning_bodies import footprints, source_guide
from viewer import WIDTH, label

SIZE = 300
IMAGE_WIDTH, IMAGE_HEIGHT = 1920,1080
MAJOR_JOINTS = ('neck', 'nose', 'tail_bottom')
POSE_AGE = .58
CADENCE = .18
EDGES = [('nose', 'neck'), ('neck', 'tail_bottom'), ('tail_bottom', 'tail_middle'),
         ('tail_middle', 'tail_top')]
for side in ('left', 'right'):
    EDGES.extend((('nose', f'{side}_eye'), (f'{side}_eye', f'{side}_ear_bottom')))
    for leg, joint in (('front', 'neck'), ('back', 'tail_bottom')):
        start = f'{side}_{leg}'
        EDGES.extend(((joint, f'{start}_elbow'), (f'{start}_elbow', f'{start}_knee'),
                      (f'{start}_knee', f'{start}_paw')))


@dataclass(frozen=True)
class Pose:
    joints: dict
    center: np.ndarray
    size: float


@dataclass(frozen=True)
class Sample:
    stamp: float
    epoch: int
    frame: np.ndarray
    poses: tuple


class NativePose:
    def __init__(self, binary):
        self.child = subprocess.Popen([str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL)

    def stop(self):
        if self.child.poll() is None:
            self.child.kill()

    def close(self):
        try:
            self.child.stdin.close()
        except BrokenPipeError:
            pass
        finally:
            self.child.wait()
            self.child.stdout.close()

    def estimate(self, images):
        if not images:
            return ()
        encoded = []
        for image in images:
            valid, data = cv.imencode('.jpg', image, (cv.IMWRITE_JPEG_QUALITY, 99))
            if not valid:
                raise RuntimeError('Could not encode the camera crop')
            encoded.append(b64encode(memoryview(data)).decode('ascii'))
        self.child.stdin.write(json.dumps(encoded).encode('utf-8') + b'\n')
        self.child.stdin.flush()
        if not select.select([self.child.stdout], [], [], 4)[0]:
            self.child.kill()
            raise TimeoutError('Apple Vision pose helper timed out')
        reply = self.child.stdout.readline()
        if not reply:
            raise RuntimeError('Apple Vision pose helper stopped')
        rows = json.loads(reply)
        if rows is None or len(rows) != len(images):
            raise RuntimeError('Apple Vision pose request failed')
        return rows


def compile_pose(folder):
    source = Path(__file__).with_name('AnimalPose.swift')
    binary = Path(folder) / 'animal-pose'
    subprocess.run(['xcrun', 'swiftc', '-O', '-module-cache-path', str(Path(folder) / 'swift-cache'),
                    str(source), '-o', str(binary)], check=True, timeout=20)
    return binary


def source_poses(native, frame):
    height,width = frame.shape[:2]
    scale_x,scale_y = width/IMAGE_WIDTH,height/IMAGE_HEIGHT
    crop_w,crop_h = max(1,round(SIZE*scale_x)),max(1,round(SIZE*scale_y))
    anchors = []
    for body in footprints(source_guide(frame), budget=12):
        u,v = body.xy
        left = int(np.clip(round(2*u*scale_x-crop_w/2),0,width-crop_w))
        top = int(np.clip(round(2*v*scale_y-crop_h/2),0,height-crop_h))
        anchors.append((left,top))
    crops = [frame[y:y+crop_h,x:x+crop_w] for x,y in anchors]
    proposed = []
    for (x,y),group in zip(anchors,native.estimate(crops)):
        for row in group:
            major_count = sum(row[key][2]>=.35 for key in MAJOR_JOINTS if key in row)
            if len(row)>=10 and major_count>=2:
                joints = {key: ((x+crop_w*u)/scale_x, (y+crop_h*v)/scale_y, c)
                          for key,(u,v,c) in row.items()}
                points = np.float32([p[:2] for p in joints.values()])
                center = np.median(points, axis=0)
                size = float(np.linalg.norm(np.ptp(points, axis=0)))
                if 18. <= size <= 275.:
                    proposed.append(Pose(joints, center, size))
    kept = []
    proposed.sort(key=lambda p: sum(v[2] for v in p.joints.values()), reverse=True)
    for pose in proposed:
        duplicate = False
        for other in kept:
            keys = pose.joints.keys() & other.joints.keys()
            differences = [np.linalg.norm(np.subtract(pose.joints[k][:2],other.joints[k][:2]))
                           for k in keys]
            duplicate = (np.linalg.norm(pose.center-other.center)<max(14.,.24*min(pose.size,other.size))
                         or (len(differences)>=5 and np.median(differences)<6.))
            if duplicate:
                break
        if not duplicate:
            kept.append(pose)
    return tuple(kept[:4])


class PoseLane:
    def __init__(self):
        self.tmp = TemporaryDirectory(prefix='motion-review-pose-')
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='animal-pose')
        self.native = None
        self.latest = None
        self.failed = None
        self.pending = None
        self.epoch = None
        self.submitted = None
        try:
            self.native = NativePose(compile_pose(self.tmp.name))
        except (OSError,subprocess.SubprocessError) as exc:
            self.failed = exc
        else:
            self.pending = self.pool.submit(self.warm)

    def warm(self):
        self.native.estimate([np.zeros((SIZE,SIZE,3), np.uint8)])

    def measure(self, frame, stamp, epoch):
        return Sample(stamp,epoch,frame,source_poses(self.native,frame))

    def prepare(self):
        # Warm the replay reader before it consumes the first recorded exposure.
        if self.pending is None:
            return
        try:
            self.pending.result()
        except Exception as exc:
            self.failed = exc
        finally:
            self.pending = None

    def service(self, frame, stamp, epoch):
        changed = False
        if epoch!=self.epoch or stamp is None:
            changed = self.latest is not None
            self.latest = self.submitted = None
            self.epoch = epoch
        if self.pending is not None and self.pending.done():
            work, self.pending = self.pending, None
            try:
                value = work.result()
            except Exception as exc:
                self.failed = exc
                self.latest = None
                changed = True
            else:
                if value is not None and value.epoch == epoch and stamp is not None:
                    self.latest = value if value.poses and 0<=stamp-value.stamp<=POSE_AGE else None
                    changed = True
        if self.latest is not None and stamp-self.latest.stamp>POSE_AGE:
            self.latest = None
            changed = True
        if (frame is not None and self.failed is None and self.pending is None and
                (self.submitted is None or stamp-self.submitted>=CADENCE)):
            self.submitted = stamp
            self.pending = self.pool.submit(self.measure,frame,stamp,epoch)
        return changed

    def close(self):
        if self.native is not None:
            self.native.stop()
        self.pool.shutdown(wait=True,cancel_futures=True)
        try:
            if self.native is not None:
                self.native.close()
        finally:
            self.tmp.cleanup()


def paint_pose(canvas, evidence, stamp):
    cyan = (234,215,94)
    canvas[:,WIDTH:] = (23,20,18)
    picture = cv.resize(evidence.frame,(WIDTH,540),interpolation=cv.INTER_AREA)
    height,width = evidence.frame.shape[:2]
    scale_x,scale_y = width/IMAGE_WIDTH,height/IMAGE_HEIGHT
    label(canvas,'Animal pose review  |  Estimated joints [P]',WIDTH+14,28,.55)
    label(canvas,f'{1000*(stamp-evidence.stamp):.0f} ms behind camera',WIDTH+748,28,.47,(169,181,195))
    for i,pose in enumerate(evidence.poses):
        def point(key):
            x,y,_ = pose.joints[key]
            return round(x/2),round(y/2)
        for a,b in EDGES:
            if a in pose.joints and b in pose.joints:
                cv.line(picture,point(a),point(b),cyan,1,cv.LINE_AA)
        for key in pose.joints:
            cv.circle(picture,point(key),2,cyan,-1,cv.LINE_AA)
        x,y = pose.center
        size = int(np.clip(round(pose.size*1.4),92,284))
        wide,high = max(1,round(size*scale_x)),max(1,round(size*.43*scale_y))
        dx = int(np.clip(round(x*scale_x)-wide//2,0,width-wide))
        dy = int(np.clip(round(y*scale_y)-high//2,0,height-high))
        left = WIDTH+6+i*236
        crop = evidence.frame[dy:dy+high,dx:dx+wide]
        canvas[593:691,left:left+228] = cv.resize(crop,(228,98),interpolation=cv.INTER_NEAREST)
        cv.line(canvas,(left,592),(left+227,592),cyan,2)
    canvas[43:583,WIDTH:] = picture
