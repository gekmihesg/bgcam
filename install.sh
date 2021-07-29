#!/bin/sh
set -eu

use_venv=""
device=""

usage() {
    cat <<EOF
$0 [-v] [-d device]

  -v          use python virtualenv for package installation
  -d device   use specified v4l2loopback device as output

EOF
    exit ${1:-0}
}

while getopts 'vd:h' OPT; do
    case "$OPT" in
        v) use_venv="y";;
        d) device="$OPTARG";;
        h) usage;;
        *) usage 2;;
    esac
done

if ! lsmod | grep -q '^v4l2loopback\s'; then
    modinfo v4l2loopback >/dev/null 2>&1 &&
    echo 'v4l2loopback needs to be loaded. Please do something like this:' ||
    cat <<'EOF'
The v4l2loopback module was not found.
Please install it (the package is probably named v4l2loopback-dkms).
After the installation, please do something like this:
EOF
    cat <<'EOF'

echo 'options v4l2loopback devices=1 exclusive_caps=1 video_nr=100 card_label="bgcam"' | sudo tee /etc/modprobe.d/v4l2loopback.conf
echo v4l2loopback | sudo tee /etc/modules-load.d/v4l2loopback.conf
sudo modprobe v4l2loopback

EOF
    exit 1
fi

test -n "$device" ||
device="$(
    { command -v v4l2-ctl >/dev/null 2>/dev/null &&
        v4l2-ctl --list-devices |
            awk 'BEGIN{RS="\n\n"} /platform:v4l2loopback-/{print $NF}' ||
        ls /dev/video[0-9]* | grep -vFx "$(readlink -e /dev/v4l/by-path/*)"
    } | sed -rn 's|^/dev/video([0-9]+)|\1\t\0|p' |
        sort -n | head -1 | cut -f2)" 2>/dev/null

if [ -z "$device" ]; then
    echo "Could auto-detect v4l2loopback device."
    echo "Try to specify it manually with -d."
    exit 1
fi

if [ ! -c "$device" ]; then
    echo "'$device' is not a character device."
    exit 1
fi
if [ ! -w "$device" ]; then
    echo "'$device' is not writeable."
    echo "Try adding yourself to the '$(stat -c%G "$device")' group."
    exit 1
fi

dir="$(dirname "$(readlink -f "$0")")"
cd "$dir"

if [ "${1:-}" = venv ]; then
    test -d venv || python3 -m virtualenv venv
    source venv/bin/activate
fi
pip3 install --disable-pip-version-check --user -r requirements.txt

mkdir -p ~/.config/systemd/user ~/.config/bgcam ~/.local/bin
ln -srf "$dir/bgcam" ~/.local/bin/bgcam
ln -srf "$dir/bgcam-set-background" ~/.local/bin/bgcam-set-background
ln -srf "$dir/bgcam.service" ~/.config/systemd/user/bgcam.service

touch ~/.config/bgcam/config
sed -i -e '/^\s*BGCAM_LOOPBACK_DEVICE\s*=/d' ~/.config/bgcam/config
echo "BGCAM_LOOPBACK_DEVICE=\"$device\"" >> ~/.config/bgcam/config
grep -q '^\s*BGCAM_CAMERA\s*=' ~/.config/bgcam/config ||
    echo "BGCAM_CAMERA=\"$(ls /dev/v4l/by-id/*-index0)\"" >> \
        ~/.config/bgcam/config

systemctl --user daemon-reload
systemctl --user enable bgcam.service
systemctl --user restart bgcam.service
sleep 1
systemctl --user status bgcam.service
