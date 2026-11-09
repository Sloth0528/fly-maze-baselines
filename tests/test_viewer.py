"""Desktop smoke test; CI runs the headless test module separately."""

import unittest

from maze_baselines.environment import from_config
from maze_baselines.models import KINDS, default_checkpoint, load
from maze_baselines.viewer import Viewer


class ViewerTests(unittest.TestCase):
    def test_all_architectures_draw_step_and_reset(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                path = default_checkpoint(kind)
                brain, checkpoint = load(path, kind)
                viewer = Viewer(from_config(checkpoint['environment']), brain, checkpoint, path)
                try:
                    viewer.root.update()
                    self.assertIn('UNTRAINED', viewer.label)
                    self.assertIn('updates=0', viewer.label)
                    viewer.step()
                    viewer.root.update()
                    self.assertEqual(viewer.env.steps, 1)
                    viewer.reset()
                    self.assertEqual(viewer.env.steps, 0)
                finally:
                    viewer.root.destroy()


if __name__ == '__main__':
    unittest.main()
