import unittest
from unittest.mock import patch

import numpy as np

from gaze_camera import (
    Config,
    Corrector,
    extract_eye,
    iris_lateral_angle,
    project_angles,
)


class GazeTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((100, 140, 3), 120, np.uint8)
        self.points = np.array([[30, 50], [40, 44], [50, 44], [60, 50], [50, 56], [40, 56]], np.float32)

    def test_calibration_and_invalid_config(self):
        self.assertEqual(Config().camera_above_cm, 25.0)
        self.assertAlmostEqual(float(Config(camera_above_cm=10, viewing_distance_cm=100, strength=1).angles()[0, 0]), 5.7106, places=3)
        self.assertTrue(np.all(np.abs(Config(camera_above_cm=500).angles()) <= [30, 40]))
        target = project_angles(Config(camera_above_cm=25, viewing_distance_cm=60, strength=1),
                                [(640, 360), (640, 360)], (1280, 720))
        self.assertAlmostEqual(float(target[0]), 22.6199, places=3)
        self.assertAlmostEqual(float(target[1]), 0.0, places=3)
        for config in [Config(source='/dev/video10'), Config(fps=0), Config(width=641),
                       Config(strength=float('nan')), Config(viewing_distance_cm=0)]:
            with self.assertRaises(ValueError):
                config.validate()

    def test_border_and_blink_bypass(self):
        self.assertIsNone(extract_eye(self.frame, self.points - [40, 0], 'L'))
        closed = self.points.copy()
        closed[:, 1] = 50
        self.assertIsNone(extract_eye(self.frame, closed, 'R'))
        self.assertIsNone(extract_eye(self.frame, self.points * float('nan'), 'L'))

    def test_anchor_order_and_normalization(self):
        image, anchors, box = extract_eye(self.frame, self.points, 'L')
        self.assertEqual(image.shape, (1, 48, 64, 3))
        np.testing.assert_allclose(image, 120/255, atol=1e-7)
        x0, y0, x1, y1 = box
        expected = ((self.points[[3, 2, 1, 0, 5, 4]] - [x0, y0]) * [64/(x1-x0), 48/(y1-y0)]).astype(int)
        np.testing.assert_array_equal(anchors[0, 0, 0], -expected.flatten())

    def test_project_angles_follow_eye_position(self):
        config = Config(camera_above_cm=25, viewing_distance_cm=60, strength=1)
        above = project_angles(config, [(640, 300), (640, 300)], (1280, 720))
        below = project_angles(config, [(640, 420), (640, 420)], (1280, 720))
        self.assertLess(float(above[0]), float(below[0]))
        self.assertAlmostEqual(float(above[1]), 0.0, places=3)

    def test_iris_lateral_angle_moves_toward_center(self):
        config = Config(iris_correction_gain_deg=100, strength=1)
        points = {'L': self.points, 'R': self.points.copy()}
        centers = {
            'L': self.points.mean(axis=0) - [8, 0],
            'R': self.points.mean(axis=0) - [8, 0],
        }
        self.assertAlmostEqual(iris_lateral_angle(config, points, centers), 26.6667, places=3)
        centers['L'][0] += 16
        centers['R'][0] += 16
        self.assertAlmostEqual(iris_lateral_angle(config, points, centers), -26.6667, places=3)

    def test_correction_fades_at_blink_boundary(self):
        engine = Corrector(Config())
        try:
            self.assertEqual(engine.correction_mix, 0.0)
            engine._fade_correction(True)
            self.assertAlmostEqual(engine.correction_mix, 0.35)
            engine._fade_correction(False)
            self.assertLess(engine.correction_mix, 0.35)
        finally:
            engine.close()

    def test_no_face_and_disabled_are_exact_passthrough(self):
        engine = Corrector(Config())
        try:
            blank = np.zeros((240, 320, 3), np.uint8)
            output, eyes = engine.process(blank)
            self.assertEqual(eyes, 0)
            np.testing.assert_array_equal(output, blank)
            engine.config.strength = 0
            with patch.object(engine.detector, 'detect_for_video', side_effect=AssertionError('must bypass inference')):
                output, eyes = engine.process(blank)
            np.testing.assert_array_equal(output, blank)
        finally:
            engine.close()

    def test_pretrained_models_respond_to_angle(self):
        engine = Corrector(Config())
        try:
            rng = np.random.default_rng(7)
            for model in engine.models.values():
                outputs = []
                image, anchors, _ = extract_eye(self.frame, self.points, 'R')
                image[:] = rng.random(image.shape)
                for angle in (0, 10):
                    inputs = {'image': image, 'anchors': anchors, 'angles': np.array([[angle, 0]], np.float32)}
                    for detail in model.get_input_details():
                        model.set_tensor(detail['index'], inputs[detail['name']])
                    model.invoke()
                    output = model.get_tensor(model.get_output_details()[0]['index'])
                    self.assertEqual(output.shape, (1, 48, 64, 3))
                    self.assertTrue(np.isfinite(output).all())
                    outputs.append(output)
                self.assertGreater(np.mean(np.abs(outputs[1]-outputs[0])), 0.001)
        finally:
            engine.close()


if __name__ == '__main__':
    unittest.main()
