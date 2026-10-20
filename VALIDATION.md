# Validation on the target machine

Date: 2026-10-16. Host: GMKtec / AMD Ryzen 5 PRO 6650H, Omarchy 4.0.4, running kernel `7.2.5-3-omarchy`.

| Check | Result |
| --- | --- |
| DKMS | v4l2loopback 0.15.4 installed for `7.2.5-3-omarchy` and `7.2.3-arch1-3` |
| Headers | Matching `linux-omarchy-headers` already present; `linux-headers` installed for the alternate kernel |
| Persistence | Modules-load and modprobe files installed; both Omarchy UKIs rebuilt; active kernel UKI listing contains `etc/modprobe.d/obs-gaze.conf` |
| Device | `/dev/video10`, label `OBS Eye Contact`, exclusive capture capabilities while producing |
| Permissions | User `vitor` has read/write ACLs on both devices, without adding video-group membership |
| Conversion | Both pretrained models compared against the original frozen TensorFlow graph at three angles; tolerance `atol=2e-4, rtol=2e-4` passed |
| Unit tests | All 6 tests passed |
| Eye-quality fix | Iris horizontal error is smoothed at runtime; model output is composited through the six-point eye-opening polygon instead of the full rectangular crop |
| Dependency consistency | `uv pip check` passed; LiteRT's backports-strenum dependency constrained to the Python 3.12-compatible 1.2.8 release |
| Neural inference | Two eye networks measured at about 8.5 ms combined, with two CPU threads each |
| V4L2 consumer | 150 frames read successfully at 1280×720, YU12; measured 29.88–30.23 FPS |
| Capture fix | One UVC capture buffer delivered ~15 FPS; two buffers restored ~30 FPS |
| Runtime memory | Approximately 128–131 MiB in systemd's service accounting |
| Crash recovery | Main process killed with SIGKILL; automatically active again after 6.1 seconds |
| Stale-frame timeout | Suspended producer; consumer received uniform NULL frames, decoded as green `[0, 154, 0]` in BGR |
| Watchdog recovery | Suspended process terminated by systemd after watchdog expiry, then automatically restarted; total 26.3 seconds |
| Final service state | Enabled for `graphical-session.target` and active after the fault tests |

The restart counter reflects the two deliberate fault tests. Runtime logs also showed hundreds of frames taking the neural correction path, rather than only no-face passthrough. No camera images were saved during validation.

Not tested: an actual reboot, physical unplug/replug, or the OBS GUI (OBS was not running). V4L2 capture compatibility was verified with an independent consumer. Boot persistence is configured and the initramfs options were inspected, but a reboot was intentionally not performed. The initial screen/camera distances remain estimates; the user still needs to judge visual quality and adjust calibration in the OBS preview. The model assumes a frontal face looking at a fixed screen point.

Model SHA-256 values produced by this conversion:

```text
64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff  face_landmarker.task
5c3578bf04bb89d4e22478ec8f58535e9d52fe75c850733debfc77e2676228d8  gaze_L.tflite
416c624c0675c7e001ce4d8f502b357860fc2f3bf984d8468d76e5e0b65c94d3  gaze_R.tflite
```
