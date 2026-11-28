"""Fast behavioural tests for Fable View.

They run the real engine on small synthetic scenes on CPU, so they need no
recording and no GPU.  The full-resolution numerical checks live in
``_FableView.py --selftest``.
"""
import json
import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import fable_view_gpu as G
from fable_view_app import Controls, Saver, Session, View, build_parser, fire, load_script
from fable_view_engine import FableConfig, FableEngine, Stabiliser
from fable_view_registration import EgoMotion, corner_shift
from fable_view_motion import TrackView
from fable_view_selftest import add_blobs, add_blocks, low_light, make_world, psnr, raised_blocks, render
from fable_view_zoom import LiveSuperZoom
from m5_field_launch import FABLE_VIEW, mission_arguments

SMALL = (640, 360)
WORLD = make_world(seed=4, width=1100, height=760)


def estimate(engine):
    return engine.last_estimate[0].permute(1, 2, 0).cpu().numpy()


class LaunchContract(unittest.TestCase):
    def test_launcher_requires_the_gpu(self):
        self.assertEqual(mission_arguments(FABLE_VIEW), ['--device', 'mps', '--require-mps'])

    def test_required_gpu_is_never_replaced_by_cpu(self):
        with self.assertRaises(G.BackendError):
            G.select_device('cpu', require_mps=True)

    def test_entry_script_imports_without_relaunching(self):
        import _FableView
        self.assertTrue(callable(_FableView.main))

    def test_parser_defaults(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.source, 'rtmp://127.0.0.1:1935/live/mavic3')
        self.assertEqual(args.night, 'auto')
        self.assertFalse(args.headless)


class Registration(unittest.TestCase):
    def test_known_motion_is_recovered(self):
        ego = EgoMotion(work_width=640)
        a = render(WORLD, 0, 'pan', SMALL)
        b = render(WORLD, 3, 'pan', SMALL)
        self.assertFalse(ego.update(a).ok)                      # first frame has no reference
        reg = ego.update(b)
        self.assertTrue(reg.ok)
        from fable_view_selftest import camera
        truth = np.linalg.inv(camera(3, 'pan')) @ camera(0, 'pan')
        delta = np.linalg.inv(truth) @ reg.H
        self.assertLess(corner_shift(delta / delta[2, 2], *SMALL), 0.3)

    def test_raised_structure_shows_as_slide_and_flat_ground_does_not(self):
        # A block a quarter of the way up to a camera travelling 2.2 px/frame
        # slides 0.55 px/frame against the ground; the tracked points on it
        # must report that, and the points on the ground must not.
        ego = EgoMotion()
        world = make_world(seed=4, width=2200, height=1300)
        blocks = raised_blocks(count=1)
        on_block, on_ground = [], []
        for i in range(24):
            frame, boxes = add_blocks(render(world, i, 'pan', (1280, 720)), blocks, i, 'pan', 0.25)
            reg = ego.update(np.clip(frame, 0, 255).astype(np.uint8))
            if reg.ok and reg.field is not None and i >= 6:
                where, rate = reg.field
                a, b, c, e = boxes[0]
                inside = (where[:, 0] > a + 6) & (where[:, 0] < c - 6) & (where[:, 1] > b + 6) & (where[:, 1] < e - 6)
                outside = (where[:, 0] < a - 20) | (where[:, 0] > c + 20) | (where[:, 1] < b - 20) | (where[:, 1] > e + 20)
                if inside.sum() >= 3:
                    on_block.append(float(np.median(rate[inside])))
                on_ground.append(float(np.percentile(rate[outside], 90)))
        self.assertGreater(len(on_block), 5)
        self.assertAlmostEqual(float(np.median(on_block)), 0.55, delta=0.15)
        self.assertLess(float(np.median(on_ground)), 0.10)

    def test_blank_frames_report_failure(self):
        ego = EgoMotion(work_width=640)
        blank = np.full((360, 640, 3), 90, np.uint8)
        ego.update(blank)
        self.assertFalse(ego.update(blank).ok)


class Canvas(unittest.TestCase):
    def test_translation_rekey_is_a_whole_pixel_move(self):
        stab = Stabiliser((360, 640))
        stab.advance(G.translation(-37.3, 9.6))                 # scene content moves left: camera pans right
        move, exact = stab.rekey()
        self.assertTrue(exact)
        self.assertEqual(float(move[0, 2]) % 16.0, 0.0)
        self.assertEqual(float(move[1, 2]) % 16.0, 0.0)
        self.assertTrue(stab.inside())

    def test_rotation_rekey_resamples(self):
        stab = Stabiliser((360, 640))
        angle = math.radians(7.0)
        turn = np.array([[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0],
                         [0.0, 0.0, 1.0]])
        stab.advance(turn)
        _, exact = stab.rekey()
        self.assertFalse(exact)


class Night(unittest.TestCase):
    def test_accumulation_reduces_noise_without_blur(self):
        rng = np.random.default_rng(2)
        engine = FableEngine(FableConfig(device='cpu', night='max', detect=False))
        for i in range(18):
            noisy, clean = low_light(render(WORLD, i, 'hover', SMALL), rng, scale=0.1)
            result = engine.process(noisy, i / 30.0)
        gain = psnr(estimate(engine), clean) - psnr(noisy.astype(np.float32) / 255.0, clean)
        self.assertGreater(gain, 6.0)
        self.assertGreater(result.night.exposure_gain, 1.5)
        self.assertGreater(result.night.n_mean, 8.0)

    def test_night_off_returns_the_source(self):
        engine = FableEngine(FableConfig(device='cpu', night='off', detect=False))
        frame = render(WORLD, 0, 'hover', SMALL)
        for i in range(3):
            result = engine.process(frame, i / 30.0)
        self.assertLessEqual(int(np.abs(result.display.astype(int) - frame.astype(int)).max()), 1)

    def test_repeated_timestamp_restarts_instead_of_averaging(self):
        engine = FableEngine(FableConfig(device='cpu', night='max', detect=False))
        frame = render(WORLD, 0, 'hover', SMALL)
        engine.process(frame, 1.0)
        engine.process(frame, 1.033)
        result = engine.process(frame, 1.033)
        self.assertEqual(result.reset, 'source discontinuity')


class Motion(unittest.TestCase):
    def run_scene(self, movers, frames=70):
        rng = np.random.default_rng(6)
        engine = FableEngine(FableConfig(device='cpu', night='off', detect=True))
        hits, false_ids = 0, set()
        for i in range(frames):
            frame, truth = add_blobs(render(WORLD, i, 'hover', SMALL), movers, i, 'hover')
            frame = np.clip(frame + rng.normal(0, 1.5, frame.shape), 0, 255).astype(np.uint8)
            result = engine.process(frame, i / 30.0)
            for t in result.tracks:
                if truth and min(math.hypot(t.fx - x, t.fy - y) for (x, y) in truth) < 12:
                    hits += 1
                else:
                    false_ids.add(t.id)
        return hits, false_ids

    def test_moving_object_is_confirmed_and_tracked(self):
        hits, false_ids = self.run_scene([(620.0, 380.0, 1.6, 0.5, 80.0, 1.5)])
        self.assertGreater(hits, 25)
        self.assertEqual(false_ids, set())

    def test_static_scene_raises_nothing(self):
        hits, false_ids = self.run_scene([])
        self.assertEqual(false_ids, set())


class Zoom(unittest.TestCase):
    def test_point_zoom_stacks_frames(self):
        rng = np.random.default_rng(8)
        engine = FableEngine(FableConfig(device='cpu', night='off', detect=False))
        self.assertIsNone(engine.process(render(WORLD, 0, 'hover', SMALL), 0.0).zoom)
        engine.zoom.roi = 96.0
        engine.zoom.set_point(320.0, 180.0, engine.stab.to_canvas)
        for i in range(1, 12):
            frame = np.clip(render(WORLD, i, 'hover', SMALL) + rng.normal(0, 3.0, (360, 640, 3)), 0, 255).astype(np.uint8)
            result = engine.process(frame, i / 30.0)
        z = result.zoom
        self.assertEqual(z.image.shape, (engine.zoom.out, engine.zoom.out, 3))
        self.assertEqual(z.raw.shape, z.image.shape)
        self.assertEqual(z.mode, 'point')
        self.assertGreater(z.frames, 6.0)
        self.assertFalse(z.lost)
        self.assertAlmostEqual(z.centre[0], 320.0, delta=2.5)

    def test_lost_target_holds_the_ground_it_was_last_seen_on(self):
        # While the camera pans, a followed target that disappears must leave
        # the window on the ground where it was, showing the live picture.
        engine = FableEngine(FableConfig(device='cpu', night='off', detect=False))
        zoom = LiveSuperZoom(engine.device, engine.grids, out=240)
        zoom.follow(7)
        shown = None
        for i in range(16):
            frame = render(WORLD, i, 'pan', SMALL)
            result = engine.process(frame, i / 30.0)
            # A target fixed on the ground: it moves through the frame with the pan.
            target = [] if i >= 8 else [TrackView(7, 400.0 - 2.2 * i, 180.0, 30.0, 30.0, 0.0, 2.0, 9, False, 1.0)]
            if i >= 1:
                shown = zoom.update(frame, 1.0, engine.stab.to_canvas, result.registration, target,
                                    engine.acc.noise, i / 30.0)
        self.assertIsNotNone(shown)
        self.assertTrue(shown.lost)
        self.assertAlmostEqual(shown.centre[0], 400.0 - 2.2 * 15, delta=2.0)
        self.assertAlmostEqual(shown.centre[1], 180.0, delta=2.0)
        self.assertGreater(shown.frames, 3.0)                    # a live stack of that ground, not a frozen picture

    def test_following_a_missing_track_reports_lost(self):
        engine = FableEngine(FableConfig(device='cpu', night='off', detect=True))
        engine.process(render(WORLD, 0, 'hover', SMALL), 0.0)
        engine.zoom.follow(12345)
        result = engine.process(render(WORLD, 1, 'hover', SMALL), 1 / 30.0)
        self.assertIsNone(result.zoom)                          # never seen: nothing to show
        self.assertTrue(engine.zoom._lost)


class Operator(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.session = Session('cpu', False, 'auto', captures=Path(self.tmp.name) / 'cap', log=None)
        self.view = View()
        self.controls = Controls(self.session, self.view)

    def tearDown(self):
        self.session.close()
        self.tmp.cleanup()

    def test_waiting_screen_and_first_frame_render(self):
        canvas = self.view.render(self.session, (800, 480), None)
        self.assertEqual(canvas.shape, (480, 800, 3))
        self.session.process(render(WORLD, 0, 'hover', SMALL), 0.0)
        canvas = self.view.render(self.session, (800, 480), None)
        self.assertEqual(canvas.shape, (480, 800, 3))
        self.assertEqual(set(self.view.buttons), {'follow', 'night', 'look', 'zoom_in', 'zoom_out', 'compare',
                                                 'save', 'more'})
        self.assertIsNotNone(self.view.to_frame(400, 200))

    def test_click_inspects_a_point_and_keys_change_state(self):
        self.session.process(render(WORLD, 0, 'hover', SMALL), 0.0)
        self.view.render(self.session, (800, 480), None)
        self.controls.click(400, 200)
        self.assertEqual(self.session.engine.zoom.mode, 'point')
        self.session.process(render(WORLD, 1, 'hover', SMALL), 1 / 30.0)
        self.assertIsNotNone(self.session.result.zoom)
        self.controls.key(ord('v'))
        self.assertEqual(self.session.engine.config.night, 'max')
        self.controls.key(ord('c'))
        self.assertTrue(self.session.compare)
        self.controls.key(ord('i'))
        self.assertTrue(self.session.inspect)
        canvas = self.view.render(self.session, (800, 480), None)
        self.assertTrue(self.view.main_is_zoom)
        self.assertEqual(canvas.shape, (480, 800, 3))
        self.controls.key(27)
        self.assertTrue(self.controls.quit)

    @unittest.skipUnless(G.torch.backends.mps.is_available(), 'the dusk picture runs on the Apple GPU only')
    def test_landscape_looks_carry_the_dusk_picture_and_natural_does_not(self):
        # A bright warm sky over cool, dim ground: the scene the dusk picture acts on.
        from motion_sketch_landscape import LandscapeGrade
        camera = np.empty((180, 320, 3), dtype=np.uint8)
        camera[:72] = [106, 177, 221]
        camera[72:] = [53, 35, 29]
        camera[73::3, ::3, 0] += 10
        landscape_only = LandscapeGrade('mps').apply(camera, 0.0)
        session = Session('mps', True, 'off', captures=Path(self.tmp.name) / 'gpu', log=None)
        try:
            session.process(camera, 0.0)
            np.testing.assert_array_equal(session.display, camera)           # Natural: the decoded picture
            self.assertNotIn('dusk', session._looks)
            session.look = 'landscape'
            session.process(camera, 1 / 30.0)
            self.assertGreater(session._looks['dusk'].strength, 0.99)
            self.assertFalse(np.array_equal(session.display[90:], landscape_only[90:]))
            np.testing.assert_array_equal(session.display[:65], landscape_only[:65])
        finally:
            session.close()

    def test_follow_without_a_target_waits_honestly(self):
        self.session.process(render(WORLD, 0, 'hover', SMALL), 0.0)
        self.controls.act('follow')
        self.assertTrue(self.session.auto_follow)
        self.assertEqual(self.session.follow_id, 0)
        self.controls.act('follow')
        self.assertFalse(self.session.auto_follow)

    def test_save_writes_source_processed_and_receipt(self):
        frame = render(WORLD, 0, 'hover', SMALL)
        self.session.process(frame, 0.0)
        view = self.view.render(self.session, (800, 480), None)
        self.assertTrue(self.session.save(view, 'unit-test', 'synthetic'))
        self.session.saver.close()
        self.assertIsNone(self.session.saver.error)
        folder = Path(self.session.saver.saved[0])
        self.assertTrue(np.array_equal(cv2.imread(str(folder / 'source.png')), frame))
        details = json.loads((folder / 'frame.json').read_text())
        self.assertEqual(set(details['sha256']), {'source.png', 'processed.png', 'view.png'})
        self.assertIn('unmodified', details['statement'])

    def test_rehearsal_script_fires_actions(self):
        path = Path(self.tmp.name) / 'script.json'
        path.write_text(json.dumps([{'t': 2.0, 'key': 'c'}, {'t': 1.0, 'action': 'more'}]))
        events = load_script(path)
        self.assertEqual([e['t'] for e in events], [1.0, 2.0])
        self.session.process(render(WORLD, 0, 'hover', SMALL), 0.0)
        for event in events:
            fire(event, self.controls, self.session)
        self.assertTrue(self.session.more)
        self.assertTrue(self.session.compare)


class SaverTests(unittest.TestCase):
    def test_one_save_at_a_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            saver = Saver(tmp)
            image = np.zeros((8, 8, 3), np.uint8)
            self.assertTrue(saver.request({'source': image}, {}))
            saver.close()
            self.assertEqual(len(saver.saved), 1)
            self.assertFalse(any(p.name.startswith('.saving-') for p in Path(tmp).iterdir()))


if __name__ == '__main__':
    unittest.main()
