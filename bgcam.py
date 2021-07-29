#!/usr/bin/env python
import sys
import os
import math
import time
import signal
import types
import cv2
import collections
import logging
import mediapipe as mp
import numpy as np
import argparse
import pyfakewebcam
import inotify_simple as inotify


APPLICATION = 'bgcam'


def resize_image(image, shape, dst=None, keep_aspect=True):
    if image.shape[:2] == shape[:2]:
        # shape is already equal, just return the image without modification
        if not dst is None:
            np.copyto(dst, image)
            return dst
        return image
    if keep_aspect:
        ih, iw = shape[:2]
        bh, bw = image.shape[:2]
        scale = max(iw / bw, ih / bh)
        h, w = int(ih / scale), int(iw / scale)
        if w == 0 or h == 0:
            # image is too small, ignore the aspect ratio
            return cv2.resize(image, (iw, ih), dst)
        else:
            y, x = int(bh / 2 - h / 2), int(bw / 2 - w / 2)
            return cv2.resize(image[y:y+h, x:x+w, :], (iw, ih), dst)
    else:
        return cv2.resize(image, shape[:2], dst)


class TimedVideoCapture():
    """
    cv2.VideoCapture variant that skips or repeats frames depending on the video's fps and the time
    passed since the last read.
    @param loop  Whether to restart the video once the end is reached.
    """
    # Should just inherit from cv2.VideoCapture but this causes GC issues.
    # See https://github.com/microsoft/debugpy/issues/208
    def __getattr__(self, name): return getattr(self._vc, name)

    def __init__(self, filename, loop=True, *args, **kwargs):
        #super().__init__(filename, *args, **kwargs)
        self._vc = cv2.VideoCapture(filename, *args, **kwargs)
        self._loop = loop
        self.reset()

    def reset(self):
        self._image = None
        self._frame_time = None
        self.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def read(self, dst=None):
        """
        returns (success, image, new) where new tells whether the frame was repeated or not
        """
        if self._image is None:
            #success, image = super().read()
            success, image = self._vc.read()
            if success:
                self._last_frame_time = time.monotonic()
                self._image = image
            new = success
        else:
            if self._frame_time is None:
                self._frame_time = 1 / self.get(cv2.CAP_PROP_FPS)
            missed = int((time.monotonic() - self._last_frame_time) / self._frame_time)
            if missed:
                for i in range(missed):
                    if not self.grab():
                        if self._loop:
                            self.reset()
                            return self.read(dst)
                        return (False, dst, False)
                self.retrieve(self._image)
                self._last_frame_time = time.monotonic()
                success, new = True, True
            else:
                success, new = True, False
        if not dst is None:
            np.copyto(dst, self._image)
            return (success, dst, new)
        return (success, self._image, new)


class Background():
    """
    Delivers a constant stream of background images.
    @param filename  Image or video to used as source.
    @param width     Fixed width of the output image.
    @param height    Fixed height of the output image.
    @param fallback  An integer describing a color.
    """
    def __init__(self, filename, width, height, fallback=None):
        self._filename = filename
        self._target_shape = (height, width, 3)
        if isinstance(fallback, int) and fallback >= 0:
            # convert to rgb color
            self._fallback = (fallback >> 16) & 0xff, (fallback >> 8) & 0xff, fallback & 0xff
        else:
            self._fallback = None
        self._image = np.zeros(self._target_shape, dtype=np.uint8)
        self.reload()

    def _set_image(self, image):
        # resize the image to fit frame and convert it to RGB
        resize_image(image, self._target_shape, self._image)
        cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB, self._image)

    def reload(self):
        filename = self._filename if self._filename and os.path.exists(self._filename) else None
        if filename and (image := cv2.imread(filename)) is not None:
            self._set_image(image)
            self._next = lambda: (True, self._image, False)
        elif filename and (video := TimedVideoCapture(filename)) and video.isOpened():
            self._next = video.read
        elif not self._fallback is None:
            self._image[:] = self._fallback
            self._next = lambda: (True, self._image, False)
        else:
            self._next = lambda: (False, None, False)

    def read(self, dst=None):
        success, image, new = self._next()
        if not success:
            return (False, dst)
        if new:
            self._set_image(image)
        if not dst is None:
            np.copyto(dst, self._image)
            return (True, dst)
        return (True, self._image)


class FakeCam(pyfakewebcam.FakeWebcam):
    """
    FakeWebcam wrapper with consumer counter using inotify.
    """
    def __init__(self, video_device, width, height, *args, **kwargs):
        super().__init__(video_device, width, height, *args, **kwargs)
        self._blank_frame = np.zeros((height, width, 3), dtype=np.uint8)
        self.blank()
        self._consumers = 0
        self._inotify = inotify.INotify(nonblocking=True)
        self._inotify.add_watch(os.path.realpath(video_device),
            inotify.flags.OPEN | inotify.flags.CLOSE_NOWRITE| inotify.flags.CLOSE_WRITE)

    def blank(self):
        self.schedule_frame(self._blank_frame)

    def has_consumers(self):
        for event in self._inotify.read(0):
            if event.mask & (inotify.flags.CLOSE_NOWRITE | inotify.flags.CLOSE_WRITE):
                # if the device was already opened when we started watching, there could be
                # more close than open events, so never go below zero
                self._consumers = max(0, self._consumers - 1)
            if event.mask & inotify.flags.OPEN:
                self._consumers += 1
        return self._consumers > 0


class ComposedCam():
    # Should just inherit from cv2.VideoCapture but this causes GC issues.
    # See https://github.com/microsoft/debugpy/issues/208
    def __getattr__(self, name): return getattr(self._vc, name)

    def __init__(self, video_device, width, height, bg_stream,
            threshold=25, blur=55, *args, **kwargs):
        #super().__init__(video_device, *args, **kwargs)
        self._vc = cv2.VideoCapture(video_device, *args, **kwargs)
        if not self.isOpened():
            return
        self._target_shape = (height, width, 3)
        self._bg_stream = bg_stream
        self._threshold = max(0, min(100, threshold)) / 100
        self._blur = (max(3, blur | 1),) * 2
        self._mask_blur = (max(3, int(math.sqrt(width**2 + height**2) / 200) | 1),) * 2
        self._segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

        self._image = np.zeros(self._target_shape, dtype=np.uint8)
        self._bg_image = None

    def read(self, dst=None):
        # success, raw_image = super().read()
        success, raw_image = self._vc.read()
        if not success:
            return (False, None)

        image = self._image if dst is None else dst
        resize_image(raw_image, self._target_shape, image)  # scale image to target size

        cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)  # convert to rgb to get better results
        image.flags.writeable = False  # from the docs, set to False for better performance
        mask = self._segmenter.process(image).segmentation_mask
        image.flags.writeable = True

        # improve the mask
        cv2.threshold(mask, self._threshold, 1, cv2.THRESH_BINARY, mask)
        cv2.blur(mask, self._mask_blur, mask)
        cv2.threshold(mask, 0.25, 1, cv2.THRESH_TOZERO, mask)

        success, bg_image = self._bg_stream.read() # already rgb
        if not success:  # no background image available, use blurred version of the original
            self._bg_image = cv2.blur(image, self._blur, self._bg_image)
            bg_image = self._bg_image
        else: # free the buffer
            self._bg_image = None

        cv2.blendLinear(image, bg_image, mask, 1 - mask, image)
        return (True, image)


class SignalCollector():
    """
    Collects signals for later processing.
    x.add_signal('reload', signal.SIGHUP, persist=False)
    # kill -HUP $pid
    x.reload # True
    x.reload # False
    """
    def __init__(self):
        self._handlers = dict()

    def add_signal(self, name, *signums, persist=False):
        """
        Add a signal handler.
        @param name      Name for the attribute to check whether a signal was received.
        @param *signums  Signals to listen for.
        @param persist   Whether to keep the state at True or reset to False on read.
        """
        if name in self._handlers:
            raise ArgumentError('signal for %s already defined' %(name))
        self._handlers[name] = types.SimpleNamespace(state=False, persist=persist)
        def handler(signum, frame):
            self._handlers[name].state = True
        for signum in signums:
            signal.signal(signum, handler)

    def __getattr__(self, name):
        """
        Returns the signal state and resets it if it was added with persist=False
        """
        if name in self._handlers:
            if self._handlers[name].state:
                if not self._handlers[name].persist:
                    self._handlers[name].state = False
                return True
            return False
        raise AttributeError('no handler for %s' %(name))


class EnvArgumentParser(argparse.ArgumentParser):
    """
    Automatically sets the default to a corresponding envirnment variable (upper case).
    @param prefix  Common environment variable prefix.
    """
    def __init__(self, prefix='', *args, **kwargs):
        self._prefix = prefix
        super().__init__(*args, **kwargs)

    def add_argument(self, *args, **kwargs):
        arg = super().add_argument(*args, **kwargs)
        arg.default = os.environ.get(self._prefix + arg.dest.upper(), arg.default)
        return arg


if __name__ == '__main__':
    env_prefix = APPLICATION.upper() + '_'
    log_levels = (logging.CRITICAL, logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG)
    parser = EnvArgumentParser(env_prefix, prog=APPLICATION,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description=f'''
                All options can be passed by using environment variables,
                i.e. {env_prefix}LOOPBACK_DEVICE).''',
            epilog='''
                SIGHUP re-opens the camera (in preferred order), SIGUSR1 reloads the background
                image.''')
    parser.add_argument('-b', '--background', default=os.path.join(os.path.expanduser('~'),
                '.config', APPLICATION, 'background'), help='''
            Background image or video. If the file doesn\'t exists, a fallback color or the
            blurred camera image will be used instead.''')
    parser.add_argument('-c', '--fallback-color', default='-1', help='''
            Color to use as background if background image does not exist.''')
    parser.add_argument('-B', '--blur-radius', type=int, default=55, help='''
            Blur radius for background if background image does not exist and no fallback color is
            set.''')
    parser.add_argument('-F', '--fps', type=int, default=0, help='''
            Frame rate limit. Use the camera\'s default frame rate if set to 0.
            If the camera is slower, this option as no effect.''')
    parser.add_argument('-W', '--width', type=int, default=640, help='''
            Output width.''')
    parser.add_argument('-H', '--height', type=int, default=480, help='''
            Output height.''')
    parser.add_argument('-C', '--codec', default='', help='''
            Real cameras codec to use. See `v4l2-ctl -d /dev/video0 --list-formats-ext\' to get the
            supported codecs.''')
    parser.add_argument('-v', '--loopback-device', default='/dev/video100', help='''
            V4L2 loopback device to use as output.''')
    parser.add_argument('-t', '--threshold', type=int, default=50, help='''
            Certainty threshold in percent for splitting the input image into foreground and
            background.''')
    parser.add_argument('-l', '--log-level', default='info',
            choices=[logging.getLevelName(x).lower() for x in log_levels], help='''
            Log level.''')
    parser.add_argument('camera', nargs='*', default='/dev/video0', help='''
            Real cameras in preferred order.''')
    config = parser.parse_args()

    logging.basicConfig(level=logging.getLevelName(config.log_level.upper()),
            format='%(levelname)s: %(message)s')
    logger = logging.getLogger(APPLICATION)

    if isinstance(config.camera, str):
        # default (or environment variable) is a string, split it to a list
        config.camera = config.camera.split(' ')
    if len(config.camera) == 0:
        logger.error('No cameras specified.')
        sys.exit(2)

    try:
        # type is string so that we can parse for 0xFFFFFF
        config.fallback_color = int(config.fallback_color, 0)
    except ValueError:
        logger.error('Cannot parse fallback color %s', config.fallback_color)
        config.fallback_color = None
        sys.exit(2)

    if config.codec:
        # convert FOURCC codec to int
        if len(config.codec) == 4:
            config.codec = cv2.VideoWriter_fourcc(*config.codec)
        else:
            logger.error('Cannot parse codec %s', config.codec)
            sys.exit(2)
    else:
        config.codec = 0

    try:
        fake_cam = FakeCam(config.loopback_device, config.width, config.height)
    except Exception as e:
        logger.error('%s - Close all consumers and try again', str(e).strip())
        sys.exit(1)

    show_fps = sys.stdout.isatty() and logger.isEnabledFor(logging.INFO)
    bg_stream = Background(config.background, config.width, config.height, config.fallback_color)
    max_failed_frames = 5  # how many failed camera reads in a row before considering the cam lost

    # start listening for signals
    state = SignalCollector()
    state.add_signal('reload_camera', signal.SIGHUP)
    state.add_signal('reload_background', signal.SIGUSR1)
    state.add_signal('should_stop', signal.SIGINT, signal.SIGQUIT, signal.SIGTERM, persist=True)

    logger.debug('Starting main loop')
    while not state.should_stop:
        logger.debug('Waiting for consumers')
        while not state.should_stop:
            if fake_cam.has_consumers():
                logger.debug('Got consumers')
                break
            time.sleep(0.5)
        else:
            continue

        logger.debug('Opening camera')
        while not state.should_stop and fake_cam.has_consumers():
            for filename in config.camera:
                if not os.path.exists(filename):
                    logger.debug('Did not find camera %s', filename)
                else:
                    cam = ComposedCam(filename, config.width, config.height, bg_stream,
                            config.threshold, config.blur_radius)
                    if cam.isOpened():
                        logger.info('Opened camera %s', filename)
                        break
                    else:
                        logger.debug('Failed to open %s', filename)
            else:
                logger.debug('No valid camera found')
                time.sleep(1)
                continue
            break
        else:
            continue

        camera_config = argparse.Namespace()
        for do_set in (True, False):
            # the properties have an impact on each other, so in the first pass, set all of them
            # and read them in the second pass
            for prop, key in [
                        (cv2.CAP_PROP_FOURCC,       'codec'),
                        (cv2.CAP_PROP_FRAME_WIDTH,  'width'),
                        (cv2.CAP_PROP_FRAME_HEIGHT, 'height'),
                        (cv2.CAP_PROP_FPS,          'fps'),
                    ]:
                value = getattr(config, key)
                if do_set:
                    if value > 0:
                        cam.set(prop, value)
                else:
                    effective_value = int(cam.get(prop))
                    setattr(camera_config, key, effective_value)
                    logger.debug('Set %s to %s (effective: %s)', key, value, effective_value)

        bg_stream.reload()
        failed_frames = 0

        # if the camera is running faster than requested, limit fps manually
        fps = config.fps if config.fps > 0 else camera_config.fps
        limit_fps = fps < camera_config.fps

        if show_fps:
            fps_last_display = time.monotonic()
            frame_time = time.monotonic()
            frame_timings = collections.deque(maxlen=fps)

        logger.debug('Start producing frames')
        while cam.isOpened() and not state.should_stop:
            frame_process_start = time.monotonic()

            if not fake_cam.has_consumers():
                logger.info('No more consumers')
                fake_cam.blank()
                break
            if state.reload_camera:
                logger.info('Reloading camera')
                break
            if state.reload_background:
                logger.info('Reloading background')
                bg_stream.reload()

            success, image = cam.read()
            if not success:
                failed_frames += 1
                logger.debug('Failed to get frame from camera %d/%d',
                        failed_frames, max_failed_frames)
                if failed_frames >= max_failed_frames:
                    logger.warning('Lost camera')
                    break
            else:
                failed_frames = 0
                fake_cam.schedule_frame(image)

            if show_fps:
                now = time.monotonic()
                frame_timings.append(now - frame_time)
                frame_time = now
                if now - fps_last_display >= 0.5: # update every 500ms
                    # average fps over the last second
                    print(' % 6.2f FPS' %(1 / np.mean(frame_timings)), end='\r')
                    fps_last_display = now

            if limit_fps:
                time.sleep(max(0, 1 / fps - time.monotonic() + frame_process_start))

        # end frame loop

        cam.release()
        logger.info('Closed camera')

    # end main loop

    logger.debug('Stopping')
    fake_cam.blank()
