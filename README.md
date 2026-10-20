# OBS Eye Contact

Local neural gaze redirection for Arch/Omarchy on an AMD CPU:

`Lenovo webcam (/dev/video0) → MediaPipe landmarks → two pretrained eye-warp networks → /dev/video10 → OBS`

The runtime uses CPU/XNNPACK, not CUDA or ROCm. The converted eye models are about 1 MB each; MediaPipe's face model is 3.6 MB. Frames stay in memory; the pipeline does not record or upload video. Network access is needed only to install dependencies and prepare models.

## Use the installed camera

In OBS, add **Video Capture Device (V4L2)** and select **OBS Eye Contact**, `/dev/video10`. Use **1280×720**, **30 FPS**, and **YU12 / I420** if a pixel format must be selected. Start the service before opening the device selector; `exclusive_caps=1` exposes capture capability only while a producer is attached.

This is an OBS input. OBS's **Start Virtual Camera** button publishes OBS output and must not write to the same device. Disable any OBS source that opens the physical `/dev/video0`, since this pipeline owns that webcam.

```bash
systemctl --user status obs-gaze.service
systemctl --user restart obs-gaze.service
journalctl --user -u obs-gaze.service -f
# Release the physical webcam:
systemctl --user stop obs-gaze.service
# Stop now and disable automatic startup:
systemctl --user disable --now obs-gaze.service
# Restore automatic startup:
systemctl --user enable --now obs-gaze.service
```

The kernel device is created at boot. The inference service starts at graphical login and stops at logout. It retries failures every five seconds, drains capture into a single latest-frame slot, and has a 20-second systemd watchdog. The loopback sends uniform NULL frames after three seconds without input (these appear green in the negotiated YUV format, rather than retaining the last camera image). No video-group membership or world-writable device permissions are required on an active local desktop session: logind/udev supplies the device ACLs.

## Calibration

Edit `~/.config/obs-gaze/config.toml`, then restart the service. The repository's `config.toml` supplies defaults only; reinstalling the user service preserves the installed configuration.

| Setting | Meaning |
| --- | --- |
| `camera_above_cm` | Vertical distance from the screen point you normally look at to the lens; positive above |
| `camera_right_cm` | Horizontal distance from that point to the lens; positive right |
| `viewing_distance_cm` | Approximate distance from your eyes to the screen/camera |
| `focal_length_px` | Approximate webcam focal length used by the WangWilly camera geometry |
| `strength` | 0 = unchanged video; 1 = full calibrated correction |
| `iris_correction_gain_deg` | Horizontal correction gain for iris displacement normalized by eye width; default 200 is intentionally aggressive |
| `iris_smoothing` | Temporal smoothing for that offset; lower values reduce jitter |

The default values assume the lens is 25 cm above the screen fixation point and you sit 60 cm away. The runtime follows WangWilly's camera geometry for the vertical target and adds a lateral iris-displacement correction when the camera is centered. It smooths that target before applying the same target to both pretrained eye models. `strength` scales the resulting angle, which is limited to the model's training domain: ±30° vertically and ±40° horizontally.

This is a pretrained neural image warp, not a model that synthesizes arbitrary head poses. MediaPipe's iris landmarks locate the eyes for the same geometry-based target calculation used by the referenced project; they do not provide a full 3D gaze vector. The corrected 64×48 eye crops replace the interior of the original crop while a small border is preserved. Looking far away, changing distance substantially, glasses reflections, or strongly tilted faces can still produce artifacts. Blinks, missing faces, eyes outside the image, and strongly asymmetric/rolled eye geometry bypass correction. The head-pose guard is a heuristic, not full 3D pose estimation. The original weights were trained on frontal head poses. Visual calibration is still needed for your camera placement.

For persistent physical-camera identity, `source` can be changed to a `/dev/v4l/by-id/...-video-index0` path. The installed default is `/dev/video0`. Keep `sink` at `/dev/video10`; the system module and service timeout control are configured for it.

## Reproduce the setup

Run these from the project directory. Keep the directory in place because the user service refers to its absolute path.

```bash
# In a terminal; installs headers for installed kernel pkgbase names, including linux-omarchy.
sudo ./install-system.sh

# Isolated Python 3.12; the system Python is not modified.
uv sync --frozen --python 3.12

# One-time TensorFlow CPU environment; TensorFlow is not a runtime dependency.
uv run --isolated --no-project --python 3.12 --with tensorflow-cpu==2.19.0 python tools/prepare_models.py

.venv/bin/python -m unittest -v test_gaze_camera.py
.venv/bin/python install-user.py
```

`install-system.sh` installs `v4l2loopback-dkms`, matching headers, and `v4l-utils`; verifies the active kernel's headers; creates `/etc/modules-load.d/obs-gaze.conf` and `/etc/modprobe.d/obs-gaze.conf`; loads the module; and rebuilds the boot images through `limine-mkinitcpio` on Omarchy. Other mkinitcpio systems use `mkinitcpio -P`. It refuses an occupied `/dev/video10` or conflicting module options instead of unloading another application's camera. A kernel updated beyond the running version may require rebooting before installation can finish. It does not refresh package databases or perform a full system upgrade.

The user installer creates `~/.config/systemd/user/obs-gaze.service`, checks the unit, and enables/starts it. It preserves calibration and backs up changed unit files.

The model preparation script verifies source SHA-256 hashes, imports the original pretrained TensorFlow graph, removes the training batch slicing, converts to built-in LiteRT operators, and compares outputs against the frozen original at three correction angles before saving each model. It does not initialize an untrained network or substitute geometric iris translation.

## Validation and troubleshooting

```bash
dkms status
cat /sys/module/v4l2loopback/parameters/exclusive_caps
v4l2-ctl -d /dev/video10 --all
timeout 15 v4l2-ctl -d /dev/video10 --stream-mmap=3 --stream-count=150 --stream-to=/dev/null
systemctl --user is-enabled obs-gaze.service
```

For a bounded foreground test, first stop the service, run `.venv/bin/python gaze_camera.py --config ~/.config/obs-gaze/config.toml --frames 150`, then restart the service. Two producers cannot share the sink.

- Missing `/dev/video10`: check `journalctl -b -u systemd-modules-load`, `dkms status`, and the headers for `uname -r`.
- Device busy: close other applications consuming the physical webcam. OBS should consume the loopback.
- Permission denied: check `getfacl /dev/video0 /dev/video10` from the active desktop login. This configuration is for a graphical user session, not a root or lingering headless service.
- A black stream after failure: inspect the service journal. Disconnects and capture stalls restart the worker automatically; it retries if the webcam is absent.
- Low FPS: improve lighting first. This webcam needs at least two queued capture buffers; using one was measured to halve delivery to about 15 FPS. The latest-frame thread prevents those driver buffers from growing an application backlog.
- `VIDIOC_CREATE_BUFS ... Inappropriate ioctl for device` from `v4l2-ctl`: this driver's optional buffer-creation ioctl is unsupported; the tool falls back to its normal streaming buffers. Check the subsequent frame counts and exit status.

Unit tests cover calibration validation, crop boundaries, blink bypass, anchor ordering, no-face/disabled passthrough, and pretrained-model response to angles. See `VALIDATION.md` for measurements on this machine. No reboot is performed automatically.

## Remove persistence

```bash
systemctl --user disable --now obs-gaze.service
rm ~/.config/systemd/user/obs-gaze.service
systemctl --user daemon-reload
sudo rm /etc/modules-load.d/obs-gaze.conf /etc/modprobe.d/obs-gaze.conf
sudo limine-mkinitcpio
```

Keep or remove `~/.config/obs-gaze/` separately according to whether you want to retain calibration. Once all consumers close the loopback, `sudo modprobe -r v4l2loopback` removes the current device. Installed DKMS packages can be retained for other virtual cameras.

## Model provenance

- [Hsu et al. gaze correction](https://github.com/chihfanhsu/gaze_correction) and [WangWilly/gaze-correction-cam v0.1.1 weights](https://github.com/WangWilly/gaze-correction-cam/releases/tag/v0.1.1). The crop geometry and anchor ordering follow this implementation. BSD terms are retained in `THIRD_PARTY_LICENSE.txt`.
- [MediaPipe Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python), float16 model version 1. MediaPipe landmarks locate the eyes; they do not themselves redirect gaze.
- [v4l2loopback persistence and exclusive capabilities](https://github.com/v4l2loopback/v4l2loopback/blob/main/README.md).
- [pyvirtualcam Linux output](https://github.com/letmaik/pyvirtualcam).

Source download checksums and immutable model URLs are in `tools/prepare_models.py`; Python dependency versions and hashes are in `uv.lock`.
