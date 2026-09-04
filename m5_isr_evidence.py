"""Bounded, asynchronous ISR track history with explicit persistence failures."""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import queue
import sys
import threading
import time
import cv2


class EvidenceLog:
    # Operator-history policy: one position record/second plus every state
    # transition. Queue 32 bounds memory/disk lag; overflow is visible, never
    # represented as recorded evidence. This does not limit detector inputs.
    def __init__(self, snap_dir: Path, tag="fable_motion_isr_rev3"):
        self.snap_dir = Path(snap_dir)
        self.snap_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.path = self.snap_dir / f'{tag}_tracks_{stamp}.jsonl'
        self._tag = tag
        self._states = {}
        self._last_update = {}
        self._last_ts = None
        self._last_snap = -float('inf')
        self._queue = queue.Queue(maxsize=32)
        self.error = ''
        self.written = 0
        self.dropped = 0
        self._stopping = threading.Event()
        self._worker = threading.Thread(target=self._write, name='ISR evidence writer', daemon=True)
        self._worker.start()

    @property
    def status(self):
        return f'EVIDENCE ERROR {self.error}' if self.error else f'evidence written {self.written} pending {self._queue.qsize()}'

    def _fail(self, text):
        self.error = text
        print(f'[ISR evidence] {text}', file=sys.stderr, flush=True)

    def _write(self):
        while not self._stopping.is_set() or not self._queue.empty():
            try:
                records, snapshot = self._queue.get(timeout=.1)
            except queue.Empty:
                continue
            try:
                if snapshot is not None:
                    path, frame = snapshot
                    try:
                        if not cv2.imwrite(str(path), frame):
                            raise OSError(f'PNG write returned false: {path.name}')
                        records.append({'event': 'snapshot_written', 'snapshot': path.name,
                                        'bytes': path.stat().st_size})
                    except Exception as exc:
                        self._fail(f'{type(exc).__name__}: {exc}')
                        records.append({'event': 'snapshot_failed', 'snapshot': path.name,
                                        'error': str(exc)})
                with self.path.open('a', encoding='utf-8') as stream:
                    for record in records:
                        stream.write(json.dumps(record, sort_keys=True) + '\n')
                    stream.flush()
                self.written += len(records)
            except Exception as exc:
                self._fail(f'{type(exc).__name__}: {exc}')
            finally:
                self._queue.task_done()

    def observe(self, result, frame):
        ts = float(result.ts)
        records = []
        if self._last_ts is not None and ts <= self._last_ts:
            if ts == self._last_ts:
                return None
            self._states.clear()
            self._last_update.clear()
            self._last_snap = -float('inf')
            records.append({'event': 'source_timeline_reset', 'source_ts': ts})
        self._last_ts = ts
        current = {t.tid: t for t in result.tracks if t.state == 'CONF'}
        new = [t for tid, t in current.items() if tid not in self._states]
        snapshot = None
        snap_name = None
        if new and ts - self._last_snap >= 2:
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            snap_name = f'{self._tag}_source_{stamp}.png'
            snapshot = (self.snap_dir / snap_name, frame.copy())
        for tid, track in current.items():
            if tid not in self._states or ts - self._last_update.get(tid, -float('inf')) >= 1:
                records.append({'event': 'confirmed' if tid not in self._states else 'position',
                                'source_ts': ts, 'recorded_at': datetime.now().astimezone().isoformat(),
                                'track': asdict(track), 'snapshot_requested': snap_name,
                                'coordinates': 'source image pixels; speed is anchor pixels/second',
                                'semantic_identity': 'unknown'})
        for tid in self._states.keys() - current.keys():
            records.append({'event': 'lost', 'source_ts': ts, 'tid': tid})
        if not records:
            return None
        try:
            self._queue.put_nowait((records, snapshot))
        except queue.Full:
            self.dropped += len(records)
            self._fail(f'queue full; {self.dropped} history records not persisted')
            return 'EVIDENCE ERROR: history queue full'
        # Advance only after admission, so a rejected transition can be retried.
        self._states = {tid: True for tid in current}
        for record in records:
            if 'track' in record:
                self._last_update[record['track']['tid']] = ts
        self._last_update = {tid: value for tid, value in self._last_update.items() if tid in current}
        if snapshot is not None:
            self._last_snap = ts
        return ('CONFIRMED ' + ','.join(f'#{t.tid}' for t in new) + ' | evidence queued') if new else None

    def close(self):
        self._stopping.set()
        self._worker.join(timeout=5)
        if self._worker.is_alive():
            self._fail('writer missed 5-second shutdown deadline; evidence may be incomplete')
        return not self.error
