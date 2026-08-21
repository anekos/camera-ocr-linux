import logging

from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image

from ocr_app.camera import Camera, bgr_frame_to_rgb_bytes

logger = logging.getLogger(__name__)

TARGET_FPS = 30
CAMERA_DEVICE_INDEX = 0


class OcrApp(App):
    def build(self) -> Image:
        self.image_widget = Image()
        self.camera = Camera(device_index=CAMERA_DEVICE_INDEX)
        Clock.schedule_interval(self._update, 1.0 / TARGET_FPS)
        return self.image_widget

    def _update(self, dt: float) -> None:
        frame = self.camera.read_frame()
        if frame is None:
            logger.warning("Failed to read frame from camera; keeping last frame")
            return

        height, width = frame.shape[:2]
        texture = Texture.create(size=(width, height), colorfmt="rgb")
        texture.blit_buffer(
            bgr_frame_to_rgb_bytes(frame),
            colorfmt="rgb",
            bufferfmt="ubyte",
        )
        self.image_widget.texture = texture

    def on_stop(self) -> None:
        self.camera.release()
