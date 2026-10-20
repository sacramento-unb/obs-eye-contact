#!/usr/bin/env python3
"""Install and enable the service for the current desktop user."""
from pathlib import Path
import os
import shutil
import subprocess
import time

project = Path(__file__).resolve().parent
if os.geteuid() == 0:
    raise SystemExit("Run this as your desktop user, without sudo.")
for name in (".venv/bin/python", "models/face_landmarker.task", "models/gaze_L.tflite", "models/gaze_R.tflite"):
    if not (project / name).exists():
        raise SystemExit(f"Missing {name}; follow README setup first.")
config_dir = Path.home() / ".config/obs-gaze"
config_dir.mkdir(parents=True, exist_ok=True)
config = config_dir / "config.toml"
if not config.exists():
    shutil.copyfile(project / "config.toml", config)
unit_dir = Path.home() / ".config/systemd/user"
unit_dir.mkdir(parents=True, exist_ok=True)
unit = unit_dir / "obs-gaze.service"
escaped = str(project).replace('\\', '\\\\').replace('"', '\\"').replace('%', '%%')
if '\n' in escaped or '\r' in escaped:
    raise SystemExit("Project path must not contain newlines")
content = (project / "obs-gaze.service.in").read_text().replace("@PROJECT@", escaped)
if unit.exists() and unit.read_text() != content:
    shutil.copyfile(unit, unit.with_suffix(f".service.bak.{time.time_ns()}"))
unit.write_text(content)
subprocess.run(["systemd-analyze", "--user", "verify", str(unit)], check=True)
subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
subprocess.run(["systemctl", "--user", "enable", "obs-gaze.service"], check=True)
subprocess.run(["systemctl", "--user", "restart", "obs-gaze.service"], check=True)
print(f"Enabled obs-gaze.service. Calibration: {config}")
