"""Catch regressions in the default Mac photographic path without flight media."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from motion_sketch_desk import Desk
from motion_sketch_landscape import LandscapeGrade


@unittest.skipUnless(torch.backends.mps.is_available(), "requires an Apple GPU")
class EmbeddedDuskTests(unittest.TestCase):
    def test_default_uses_the_dusk_picture_once_and_original_bypasses_it(self):
        camera = np.empty((180, 320, 3), dtype=np.uint8)
        camera[:72] = [106, 177, 221]
        camera[72:] = [53, 35, 29]
        camera[73::3, ::3, 0] += 10
        untouched = camera.copy()
        previous_base = LandscapeGrade("mps").apply(camera, 0.)
        desk = Desk("mps")
        self.assertEqual(desk.look, "shallows")
        with patch.object(desk.dusk_picture, "apply", wraps=desk.dusk_picture.apply) as apply:
            adjusted = desk._picture(camera, 0., False)
            self.assertGreater(desk.dusk_picture.strength, .99)
            self.assertFalse(np.array_equal(adjusted[90:], previous_base[90:]))
            np.testing.assert_array_equal(adjusted[:65], previous_base[:65])
            desk.choose_look("landscape")
            np.testing.assert_array_equal(desk._picture(camera, 0., False), adjusted)
            desk.choose_look("source")
            self.assertIs(desk._picture(camera, 0., False), camera)
            apply.assert_called_once()
        np.testing.assert_array_equal(camera, untouched)

    def test_a_cut_to_daylight_stops_dusk_processing_on_the_first_frame(self):
        camera = np.empty((180, 320, 3), dtype=np.uint8)
        camera[:72] = [106, 177, 221]
        camera[72:] = [53, 35, 29]
        daylight = np.full_like(camera, 145)
        desk = Desk("mps")
        desk.choose_look("landscape")
        previous = LandscapeGrade("mps")
        previous.apply(camera, 0.)
        desk._picture(camera, 0., False)
        result = desk._picture(daylight, 1./30, False)
        self.assertEqual(desk.dusk_picture.strength, 0.)
        np.testing.assert_array_equal(result, previous.apply(daylight, 1./30))
