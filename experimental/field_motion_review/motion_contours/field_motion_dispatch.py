"""Schedule the existing field motion lanes within a two-job budget."""

from color_motion import next_source


class FieldMotionDispatch:
    _routes = {None: (), 'day': ('rapid', 'sustained'), 'night': ('night',)}

    def __init__(self, review):
        self._review = review
        self._lanes = {}
        self._retiring = set()
        self._mode = self._epoch = self._route_start = None
        self._gate_key = None
        self._night_eligible = False
        self._closed = False

    @property
    def mode(self):
        return self._mode

    def _visible_lane(self, name):
        if name in self._routes[self._mode] and name not in self._retiring:
            return self._lanes.get(name)
        return None

    @property
    def rapid(self):
        return self._visible_lane('rapid')

    @property
    def night(self):
        return self._visible_lane('night')

    @property
    def sustained(self):
        return self._visible_lane('sustained')

    def _route_for(self, frame, stamp, epoch):
        if frame is None or stamp is None:
            self._gate_key = None
            return None
        key = epoch, stamp
        if key != self._gate_key:
            self._night_eligible = self._review.night_source_eligible(frame)
            self._gate_key = key
        return 'night' if self._night_eligible else 'day'

    def _drain_retired(self, epoch):
        for name in tuple(self._retiring):
            lane = self._lanes.get(name)
            if lane is not None:
                # A missing camera stamp discards the completed result and
                # prevents submission. Wait to reset until its worker is idle.
                lane.service(None, None, epoch)
                if lane.pending is not None:
                    continue
                lane.scout_epoch = None
                lane.submitted = None
            self._retiring.remove(name)

    def _start_lane(self, name):
        options = {name: True}
        if name == 'night':
            options['night_bridge'] = True
        lane = self._review.CrossingLane(**options)
        self._lanes[name] = lane
        return lane

    def warm_day(self):
        """Prepare the daylight backends before a feed arrives when possible."""
        if not self._lanes and not self._closed:
            for name in self._routes['day']:
                self._start_lane(name)

    def service(self, frame, stamp, epoch, history):
        if self._closed:
            return False
        mode = self._route_for(frame, stamp, epoch)
        changed = (mode, epoch) != (self._mode, self._epoch)
        if changed:
            self._retiring.update(self._routes[self._mode])
            self._mode, self._epoch, self._route_start = mode, epoch, stamp
        self._drain_retired(epoch)

        active = self._routes[mode]
        for name, lane in self._lanes.items():
            if name not in active and name not in self._retiring:
                lane.service(None, None, epoch)
        for name in active:
            lane = self._visible_lane(name)
            if lane is not None:
                changed |= lane.service(None, stamp, epoch)

        # Count both preparing and analyzing lanes, including any worker
        # draining a different route. Constructing a lane can start a job.
        free = 2 - sum(lane.pending is not None for lane in self._lanes.values())
        for name in active:
            if not free:
                break
            if name in self._retiring:
                continue
            lane = self._lanes.get(name)
            if lane is None:
                lane = self._start_lane(name)
                if lane.pending is not None:
                    free -= 1
                    continue
            if lane.pending is not None:
                continue

            submission = None
            if name != 'night' and history is not None:
                recent = ((pts, image) for pts, image in history if pts >= self._route_start)
                submission = next_source(recent, epoch, lane.submitted, stamp)
            if submission is None and lane.submitted is not None:
                last_epoch, last_stamp = lane.submitted
                if last_epoch == epoch and stamp <= last_stamp:
                    continue
            changed |= lane.service(frame, stamp, epoch, submission=submission)
            free -= lane.pending is not None
        return changed

    def close(self):
        self._closed = True
        self._mode = None
        for lane in self._lanes.values():
            lane.close()
        self._lanes.clear()
        self._retiring.clear()
