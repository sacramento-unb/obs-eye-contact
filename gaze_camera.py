"""Headless webcam -> MediaPipe + neural eye warp -> V4L2 loopback."""
import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import logging
import math
import os
from pathlib import Path
import signal
import socket
import threading
import time
import tomllib

import cv2
import mediapipe as mp
import numpy as np
import pyvirtualcam
from ai_edge_litert.interpreter import Interpreter

ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger("obs-gaze")
EYES = {"L": [362, 385, 387, 263, 373, 380], "R": [33, 160, 158, 133, 153, 144]}
IRIS = {"L": [474, 475, 476, 477], "R": [469, 470, 471, 472]}


@dataclass
class Config:
    source: str = "/dev/video0"
    sink: str = "/dev/video10"
    width: int = 1280
    height: int = 720
    fps: int = 30
    camera_above_cm: float = 25.0
    camera_right_cm: float = 0.0
    viewing_distance_cm: float = 60.0
    strength: float = 0.75
    iris_correction_gain_deg: float = 45.0
    iris_smoothing: float = 0.25
    focal_length_px: float = 650.0

    def validate(self):
        if not all(type(x) is int for x in (self.width, self.height, self.fps)):
            raise ValueError("width, height and fps must be integers")
        if not (320 <= self.width <= 1920 and 240 <= self.height <= 1080
                and self.width % 2 == 0 and self.height % 2 == 0 and 1 <= self.fps <= 60):
            raise ValueError("Use even dimensions in 320x240..1920x1080 and 1..60 fps")
        values = (self.camera_above_cm, self.camera_right_cm, self.viewing_distance_cm,
                  self.strength, self.iris_correction_gain_deg, self.iris_smoothing,
                  self.focal_length_px)
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in values):
            raise ValueError("Calibration values must be finite numbers")
        if not (10 <= self.viewing_distance_cm <= 300 and 0 <= self.strength <= 1
                and 100 <= self.focal_length_px <= 2000):
            raise ValueError("Distance must be 10..300 cm; strength must be 0..1")
        if not (0 <= self.iris_correction_gain_deg <= 240 and 0 < self.iris_smoothing <= 1):
            raise ValueError("Iris gain must be 0..240 degrees and smoothing must be in (0, 1]")
        if not all(isinstance(x, str) and x.startswith('/dev/') for x in (self.source, self.sink)):
            raise ValueError("source and sink must be /dev/ device paths")
        if Path(self.source).resolve() == Path(self.sink).resolve():
            raise ValueError("Source and sink must be different devices")

    def angles(self):
        # shortcut: assumes gaze at the calibrated screen point, not arbitrary gaze tracking.
        angles = np.degrees(np.arctan2(
            [self.camera_above_cm, self.camera_right_cm], self.viewing_distance_cm
        )) * self.strength
        return np.clip(angles, [-30, -40], [30, 40]).astype(np.float32)[None]


def project_angles(config, eye_centers, video_size):
    """Use WangWilly's camera geometry to estimate the correction target angle."""
    left, right = eye_centers
    width, height = video_size
    center_x, center_y = (np.asarray(left) + np.asarray(right)) * 0.5
    eye_z = -config.viewing_distance_cm
    camera_offset = np.array([
        config.camera_right_cm,
        -config.camera_above_cm,
        0.0,
    ], dtype=np.float32)
    eye_x = (-abs(eye_z) * (center_x - width / 2) / config.focal_length_px
             + camera_offset[0])
    eye_y = (abs(eye_z) * (center_y - height / 2) / config.focal_length_px
             + camera_offset[1])
    vertical = math.degrees(math.atan2(-eye_y, -eye_z))
    horizontal = math.degrees(math.atan2(-eye_x, -eye_z))
    vertical += math.degrees(math.atan2(
        eye_y - camera_offset[1], camera_offset[2] - eye_z
    ))
    horizontal += math.degrees(math.atan2(
        eye_x - camera_offset[0], camera_offset[2] - eye_z
    ))
    # Match the vertical/horizontal ranges represented by the referenced
    # model's training data instead of saturating the 25 cm vertical offset.
    return np.clip(np.array([vertical, horizontal]) * config.strength,
                   [-30, -40], [30, 40]).astype(np.float32)


def iris_lateral_angle(config, eye_points, iris_centers):
    """Convert normalized iris displacement into a horizontal warp angle.

    The upstream camera geometry describes where the camera is relative to the
    screen, but with a centered camera its horizontal terms cancel. The iris
    displacement supplies the missing gaze direction while the eyelid contour
    supplies a stable anatomical reference.
    """
    offsets = []
    for side in EYES:
        points = np.asarray(eye_points[side], dtype=np.float32)
        center = points.mean(axis=0)
        width = float(np.linalg.norm(points[3] - points[0]))
        if width < 1 or not np.isfinite(width):
            return 0.0
        iris = np.asarray(iris_centers[side], dtype=np.float32)
        offsets.append(float((iris[0] - center[0]) / width))
    normalized_offset = float(np.mean(offsets))
    # Positive model angles move the synthesized pupil toward image-right;
    # invert the observed offset to bring it back toward the eye center.
    return float(np.clip(-normalized_offset * config.iris_correction_gain_deg,
                         -40.0, 40.0) * config.strength)


def extract_eye(frame, points, side, center=None):
    """Match the pretrained model's BGR crop and six-anchor ordering."""
    if not np.isfinite(points).all():
        return None
    width = float(np.linalg.norm(points[3] - points[0]))
    if width < 12:
        return None
    openness = (np.linalg.norm(points[1] - points[5]) + np.linalg.norm(points[2] - points[4])) / (2 * width)
    if openness < 0.16:
        return None
    # WangWilly's MediaPipe predictor uses the iris center both for the
    # geometry and for the crop origin; the six eyelid landmarks are only
    # anchors. Keep the optional argument for the standalone crop tests.
    cx, cy = points.mean(axis=0) if center is None else np.asarray(center)
    half_w = abs(float(points[3, 0] - points[0, 0])) * 0.75
    crop_h = half_w * 1.5
    x0, x1 = int(cx - half_w), int(cx + half_w)
    y0, y1 = int(cy - crop_h * 7 / 12), int(cy + crop_h * 5 / 12)
    h, w = frame.shape[:2]
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h) or min(x1-x0, y1-y0) < 8:
        return None
    crop = frame[y0:y1, x0:x1]
    image = cv2.resize(crop, (64, 48)).astype(np.float32)[None] / 255.0
    order = [3, 2, 1, 0, 5, 4] if side == "L" else [0, 1, 2, 3, 4, 5]
    anchors = ((points[order] - [x0, y0]) * [64/(x1-x0), 48/(y1-y0)]).astype(np.int32)
    yy, xx = np.mgrid[:48, :64]
    grid = np.stack((xx, yy), axis=-1)
    feature = (grid[:, :, None, :] - anchors).reshape(1, 48, 64, 12).astype(np.float32)
    return image, feature, (x0, y0, x1, y1)


class Corrector:
    def __init__(self, config):
        self.config = config
        self.models = {}
        for side in EYES:
            model = Interpreter(model_path=str(ROOT / f"models/gaze_{side}.tflite"), num_threads=2)
            model.allocate_tensors()
            self.models[side] = model
        self.detector = mp.tasks.vision.FaceLandmarker.create_from_options(
            mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(ROOT / "models/face_landmarker.task"),
                    delegate=mp.tasks.BaseOptions.Delegate.CPU,
                ),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.6,
                min_face_presence_confidence=0.6,
                min_tracking_confidence=0.6,
            )
        )
        self.timestamp = -1
        self.gaze_angles = np.zeros(2, dtype=np.float32)
        self.correction_mix = 0.0

    def close(self):
        self.detector.close()

    def process(self, frame):
        if self.config.strength == 0:
            return frame, 0
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (640, round(h * 640 / w)))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        self.timestamp = max(self.timestamp + 1, time.monotonic_ns() // 1_000_000)
        result = self.detector.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), self.timestamp
        )
        if not result.face_landmarks:
            self._fade_correction(False)
            return frame, 0
        face = result.face_landmarks[0]
        points = {side: np.array([(face[i].x*w, face[i].y*h) for i in ids], np.float32)
                  for side, ids in EYES.items()}
        iris_points = {side: np.array([(face[i].x*w, face[i].y*h) for i in ids], np.float32)
                       for side, ids in IRIS.items()}
        widths = [np.linalg.norm(p[3] - p[0]) for p in points.values()]
        axis = points["L"].mean(axis=0) - points["R"].mean(axis=0)
        # The model was trained on frontal faces; bypass strong yaw/roll and blinks.
        if min(widths) < 12 or min(widths) / max(widths) < 0.65 or abs(axis[1]) > abs(axis[0]) * 0.36:
            self._fade_correction(False)
            return frame, 0
        iris_centers = [iris_points[side].mean(axis=0) for side in EYES]
        if not all(np.isfinite(center).all() for center in iris_centers):
            self._fade_correction(False)
            return frame, 0
        centers = dict(zip(EYES, iris_centers))
        patches = {side: extract_eye(frame, p, side, centers[side])
                   for side, p in points.items()}
        if any(p is None for p in patches.values()):
            self._fade_correction(False)
            return frame, 0
        target_angles = project_angles(self.config, iris_centers, (w, h))
        target_angles[1] += iris_lateral_angle(self.config, points, centers)
        target_angles = np.clip(target_angles, [-30, -40], [30, 40]).astype(np.float32)
        smoothing = self.config.iris_smoothing
        self.gaze_angles = ((1 - smoothing) * self.gaze_angles
                            + smoothing * target_angles)
        self._fade_correction(True)
        output = frame.copy()
        for side, (image, anchors, box) in patches.items():
            model = self.models[side]
            angles = self.gaze_angles[None]
            inputs = {"image": image, "anchors": anchors, "angles": angles}
            for detail in model.get_input_details():
                model.set_tensor(detail["index"], inputs[detail["name"]])
            model.invoke()
            corrected = model.get_tensor(model.get_output_details()[0]["index"])[0]
            if not np.isfinite(corrected).all():
                raise RuntimeError("Non-finite gaze model output")
            x0, y0, x1, y1 = box
            corrected = cv2.resize(np.clip(corrected * 255, 0, 255), (x1-x0, y1-y0))
            original = frame[y0:y1, x0:x1]
            corrected = (corrected * self.correction_mix
                         + original * (1 - self.correction_mix)).astype(np.uint8)
            cut_y, cut_x = 3, 4
            output[y0 + cut_y:y1 - cut_y, x0 + cut_x:x1 - cut_x] = \
                corrected[cut_y:-cut_y, cut_x:-cut_x]
        return output, 2

    def _fade_correction(self, active):
        """Ramp correction over a few frames so blink boundaries do not jump."""
        target = 1.0 if active else 0.0
        self.correction_mix += (target - self.correction_mix) * 0.35


class LatestCamera:
    """Drain capture continuously so slow inference cannot accumulate stale frames."""
    def __init__(self, config):
        self.cap = cv2.VideoCapture(config.source, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"Cannot open {config.source}; check permissions or another camera consumer")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
        self.cap.set(cv2.CAP_PROP_FPS, config.fps)
        # UVC needs a queued buffer while the previous MJPEG frame is decoded.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        LOG.info("Capture negotiated %dx%d at %.1f fps", self.cap.get(3), self.cap.get(4), self.cap.get(5))
        self.condition = threading.Condition()
        self.frame = None
        self.sequence = 0
        self.error = None
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._capture, daemon=True)
        self.thread.start()

    def _capture(self):
        try:
            while not self.stopped.is_set():
                ok, frame = self.cap.read()
                if not ok:
                    raise RuntimeError("Webcam disconnected or stopped delivering frames")
                with self.condition:
                    self.frame = frame
                    self.sequence += 1
                    self.condition.notify_all()
        except Exception as error:
            with self.condition:
                self.error = error
                self.condition.notify_all()
        finally:
            self.cap.release()

    def read(self, previous):
        with self.condition:
            if not self.condition.wait_for(lambda: self.sequence > previous or self.error, timeout=5):
                raise RuntimeError("No camera frame for 5 seconds")
            if self.error:
                raise self.error
            return self.frame, self.sequence

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=2)


def notify(message):
    address = os.environ.get("NOTIFY_SOCKET")
    if address:
        if address.startswith("@"):
            address = "\0" + address[1:]
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as client:
            client.sendto(message.encode(), address)


def run(config, frames=0):
    config.validate()
    cv2.setNumThreads(2)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    if not Path(config.sink).exists():
        raise RuntimeError(f"Missing {config.sink}; run install-system.sh")
    with ExitStack() as stack:
        corrector = Corrector(config)
        stack.callback(corrector.close)
        camera = LatestCamera(config)
        stack.callback(camera.close)
        sink = stack.enter_context(pyvirtualcam.Camera(
            width=config.width, height=config.height, fps=config.fps,
            fmt=pyvirtualcam.PixelFormat.BGR, device=config.sink, backend="v4l2loopback",
        ))
        LOG.info("Streaming %s -> %s; base correction [vertical, horizontal]=%s degrees; lateral iris gain=%.1f",
                 config.source, sink.device, config.angles().tolist()[0],
                 config.iris_correction_gain_deg)
        started = last_log = time.monotonic()
        count = corrected = sequence = 0
        while not stop.is_set() and (frames == 0 or count < frames):
            frame, sequence = camera.read(sequence)
            if frame.shape[:2] != (config.height, config.width):
                frame = cv2.resize(frame, (config.width, config.height))
            frame, eyes = corrector.process(frame)
            sink.send(frame)
            count += 1
            corrected += bool(eyes)
            if count == 1:
                notify("READY=1")
            notify("WATCHDOG=1")
            sink.sleep_until_next_frame()
            if time.monotonic() - last_log >= 30:
                LOG.info("%.1f fps; %d/%d frames corrected", count/(time.monotonic()-started), corrected, count)
                last_log = time.monotonic()
        LOG.info("Stopped: %d frames, %.1f fps, %d corrected", count, count/max(time.monotonic()-started, .001), corrected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.toml")
    parser.add_argument("--frames", type=int, default=0, help="Exit after N frames (0 = continuous)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if args.frames < 0:
            raise ValueError("--frames must be nonnegative")
        with args.config.open("rb") as stream:
            config = Config(**tomllib.load(stream))
        run(config, args.frames)
    except Exception:
        LOG.exception("Pipeline failed; systemd will retry when running as a service")
        raise SystemExit(1)
