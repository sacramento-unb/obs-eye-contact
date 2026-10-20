#!/usr/bin/env bash
set -euo pipefail
if (( EUID != 0 )); then
  echo "Run with sudo in a terminal, or pkexec from the desktop." >&2
  exit 1
fi
active_kernel=$(uname -r)
test -f "/usr/lib/modules/$active_kernel/pkgbase" || {
  echo "Missing module tree for running kernel $active_kernel. Reboot into an installed kernel." >&2
  exit 1
}
headers=()
for base in /usr/lib/modules/*/pkgbase; do
  read -r package < "$base"
  headers+=("$package-headers")
done
pacman -S --needed --noconfirm v4l2loopback-dkms v4l-utils "${headers[@]}"
test "$(cat "/usr/lib/modules/$active_kernel/build/include/config/kernel.release")" = "$active_kernel" || {
  echo "Installed headers do not match running kernel; reboot and run again." >&2
  exit 1
}
dkms autoinstall -k "$active_kernel"
modinfo -k "$active_kernel" v4l2loopback >/dev/null
if test -e /dev/video10; then
  test "$(cat /sys/class/video4linux/video10/name)" = 'OBS Eye Contact' || {
    echo '/dev/video10 is already in use by another device; leaving it untouched.' >&2
    exit 1
  }
elif test -d /sys/module/v4l2loopback; then
  echo 'v4l2loopback is already loaded without our camera; stop its users and unload it before retrying.' >&2
  exit 1
fi
for file in /etc/modprobe.d/*.conf; do
  [[ "$file" == /etc/modprobe.d/obs-gaze.conf ]] && continue
  if grep -Eq '^[[:space:]]*(options|install)[[:space:]]+v4l2loopback\b' "$file"; then
    echo "Existing v4l2loopback configuration in $file; reconcile it first." >&2
    exit 1
  fi
done
install -d /etc/modules-load.d /etc/modprobe.d
printf '%s\n' v4l2loopback > /etc/modules-load.d/obs-gaze.conf
printf '%s\n' 'options v4l2loopback devices=1 video_nr=10 card_label="OBS Eye Contact" exclusive_caps=1' > /etc/modprobe.d/obs-gaze.conf
modprobe v4l2loopback
udevadm settle
test "$(cat /sys/class/video4linux/video10/name)" = 'OBS Eye Contact'
# Omarchy includes modconf in its UKIs; keep boot-time options consistent.
if command -v limine-mkinitcpio >/dev/null; then
  limine-mkinitcpio
else
  mkinitcpio -P
fi
v4l2-ctl --list-devices
