"""One-time conversion; run with TensorFlow CPU 2.19, outside the runtime venv."""
import hashlib
import io
from pathlib import Path
import tempfile
import urllib.request
import zipfile

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    ),
    "weights.zip": (
        "https://github.com/WangWilly/gaze-correction-cam/releases/download/v0.1.1/weights.zip",
        "07279dd9072f784e32c26e0c40bdf10270f98c6f3e9b3effc3254c4ce05fa76a",
    ),
}


def download(name):
    url, digest = ASSETS[name]
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"Checksum mismatch for {name}")
    return data


def convert(checkpoint, target):
    output_name = "tower_0/warping_model/apply_lcm/Add"
    config = tf.compat.v1.ConfigProto(
        device_count={"GPU": 0}, intra_op_parallelism_threads=2,
        inter_op_parallelism_threads=1, allow_soft_placement=True,
    )
    with tf.Graph().as_default(), tf.compat.v1.Session(config=config) as session:
        saver = tf.compat.v1.train.import_meta_graph(str(checkpoint) + ".meta", clear_devices=True)
        saver.restore(session, str(checkpoint))
        frozen = tf.compat.v1.graph_util.convert_variables_to_constants(
            session, session.graph_def, [output_name]
        )

    # Bypass training's per-GPU batch slices, keeping the original inference graph.
    with tf.Graph().as_default(), tf.compat.v1.Session(config=config) as session:
        inputs = [
            tf.compat.v1.placeholder(tf.float32, [1, 48, 64, 3], name="image"),
            tf.compat.v1.placeholder(tf.float32, [1, 48, 64, 12], name="anchors"),
            tf.compat.v1.placeholder(tf.float32, [1, 2], name="angles"),
        ]
        output, = tf.import_graph_def(
            frozen,
            input_map={
                "tower_0/strided_slice:0": inputs[0],
                "tower_0/strided_slice_1:0": inputs[1],
                "tower_0/strided_slice_2:0": inputs[2],
                "model_control/phase_train:0": tf.constant(False),
            },
            return_elements=[output_name + ":0"], name="",
        )
        converter = tf.compat.v1.lite.TFLiteConverter.from_session(session, inputs, [output])
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        model = converter.convert()
        interpreter = tf.lite.Interpreter(model_content=model, num_threads=2)
        interpreter.allocate_tensors()
        rng = np.random.default_rng(42)
        yy, xx = np.mgrid[:48, :64]
        anchor = np.stack([xx - 20, yy - 24] * 6, axis=-1)[None].astype(np.float32)
        for angle in ([0, 0], [10, 0], [-10, 5]):
            sample = [rng.random((1, 48, 64, 3), dtype=np.float32), anchor,
                      np.array([angle], dtype=np.float32)]
            expected = session.run(output, dict(zip(inputs, sample)))
            for detail in interpreter.get_input_details():
                index = ["image", "anchors", "angles"].index(detail["name"])
                interpreter.set_tensor(detail["index"], sample[index])
            interpreter.invoke()
            actual = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
            np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)
        target.write_bytes(model)
        print(f"Verified {target.name}: {len(model):,} bytes", flush=True)


if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "models/face_landmarker.task").write_bytes(download("face_landmarker.task"))
    with tempfile.TemporaryDirectory(prefix="obs-gaze-models-") as directory:
        with zipfile.ZipFile(io.BytesIO(download("weights.zip"))) as archive:
            archive.extractall(directory)
        for side in ("L", "R"):
            checkpoint = Path(directory) / "weights/warping_model/flx/12" / side / side
            convert(checkpoint, ROOT / "models" / f"gaze_{side}.tflite")
