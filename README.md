# bgcam

Virtual Linux webcam replacing or blurring the background using [MediaPipe][1].
Highly inspired by [fangfufu/Linux-Fake-Background-Webcam][2].

## Overview

This project is intended to be run as a systemd user service. It needs
[v4l2loopback][3] device (or similar) to output the image. The `install.sh`
script will provide hints for setting this up.

The virtual webcam is watched with inotify to keep track whether there are
consumers. By default, only once the virtual webcam is opened, the real camera
should turn on and image processing starts. Sending `SIGUSR2` to the process
toggles a always-on mode.

For the real camera, multiple input devices can be given and the first available
device will be used. If the device becomes unavailable, it will automatically
switch to the next in list. This can be useful for docking station setups.
If the preferred camera is reconnected while a fallback camera is active, send
`SIGHUP` to the process to switch back.

The background can be either a predefined color, an image or a video. If none of
those are given, the background will be blurred instead. Image or video can be
reloaded during runtime by sending `SIGUSR1` to the process.

## Installation

Use `./install.sh` to check for v4l2loopback setup, `pip install` all python
requirements, link all files to the correct location (all within `$HOME`) and
start the service. (Add `-v` to use a python virtual environment.)

## Usage

```
usage: bgcam [-h] [-b BACKGROUND] [-c FALLBACK_COLOR] [-B BLUR_RADIUS] [-a]
             [-F FPS] [-W WIDTH] [-H HEIGHT] [-C CODEC] [-v LOOPBACK_DEVICE]
             [-t THRESHOLD] [-m {0,1}]
             [-l {critical,error,warning,info,debug}]
             [camera ...]

All options can be passed by using environment variables, i.e.
BGCAM_LOOPBACK_DEVICE).

positional arguments:
  camera                Real cameras in preferred order. (default:
                        /dev/video0)

optional arguments:
  -h, --help            show this help message and exit
  -b BACKGROUND, --background BACKGROUND
                        Background image or video. If the file doesn't exists,
                        a fallback color or the blurred camera image will be
                        used instead. (default:
                        $HOME/.config/bgcam/background)
  -c FALLBACK_COLOR, --fallback-color FALLBACK_COLOR
                        Color to use as background if background image does
                        not exist. (default: -1)
  -B BLUR_RADIUS, --blur-radius BLUR_RADIUS
                        Blur radius for background if background image does
                        not exist and no fallback color is set. (default: 55)
  -a, --always-on       Do not wait for consumers before producing images.
                        (default: False)
  -F FPS, --fps FPS     Frame rate limit. Use the camera's default frame rate
                        if set to 0. If the camera is slower, this option as
                        no effect. (default: 0)
  -W WIDTH, --width WIDTH
                        Output width. (default: 640)
  -H HEIGHT, --height HEIGHT
                        Output height. (default: 480)
  -C CODEC, --codec CODEC
                        Real cameras codec to use. See `v4l2-ctl -d
                        /dev/video0 --list-formats-ext' to get the supported
                        codecs. (default: )
  -v LOOPBACK_DEVICE, --loopback-device LOOPBACK_DEVICE
                        V4L2 loopback device to use as output. (default:
                        /dev/video100)
  -t THRESHOLD, --threshold THRESHOLD
                        Certainty threshold in percent for splitting the input
                        image into foreground and background. (default: 50)
  -m {0,1}, --model {0,1}
                        Model to use for segmentation, use 0 to select the
                        general model, and 1 to select the landscape model
                        (default: 1)
  -l {critical,error,warning,info,debug}, --log-level {critical,error,warning,info,debug}
                        Log level. (default: info)

SIGHUP re-opens the camera (in preferred order), SIGUSR1 reloads the
background image, SIGUSR2 toggles between on-demand and always-on state.
```

`bgcam-set-background` manages the background image as a symlink. Passing a
filename will set the symlink to it and send the reload signal to the systemd
service. If no filename is given the symlink will be removed which causes the
application to fall back to a background color or blurring. It can be used with
[filemanager-actions][4] for example.

## License

GNU General Public License v3.0 (see https://www.gnu.org/licenses/gpl-3.0.txt)

[1]: https://google.github.io/mediapipe/solutions/selfie_segmentation.html
[2]: https://github.com/fangfufu/Linux-Fake-Background-Webcam
[3]: https://github.com/umlaeute/v4l2loopback
[4]: https://gitlab.gnome.org/GNOME/filemanager-actions
