import logging
import threading

import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from yomitoku import DocumentAnalyzer

from ocr_app.camera import Camera, bgr_frame_to_rgb_array, bgr_frame_to_rgb_bytes

logger = logging.getLogger(__name__)

TARGET_FPS = 30
CAMERA_DEVICE_INDEX = 0
BUTTON_HEIGHT = 50


class OcrApp(App):
    last_frame: np.ndarray | None = None

    def build(self) -> BoxLayout:
        self.image_widget = Image()
        self.ocr_button = Button(text="OCR実行", size_hint_y=None, height=BUTTON_HEIGHT)
        self.ocr_button.bind(on_press=self._on_ocr_button_press)

        layout = BoxLayout(orientation="vertical")
        layout.add_widget(self.image_widget)
        layout.add_widget(self.ocr_button)

        self.camera = Camera(device_index=CAMERA_DEVICE_INDEX)
        self.analyzer = DocumentAnalyzer(device="cuda")
        Clock.schedule_interval(self._update, 1.0 / TARGET_FPS)
        return layout

    def _update(self, dt: float) -> None:
        frame = self.camera.read_frame()
        if frame is None:
            logger.warning("Failed to read frame from camera; keeping last frame")
            return

        self.last_frame = frame

        height, width = frame.shape[:2]
        texture = Texture.create(size=(width, height), colorfmt="rgb")
        texture.blit_buffer(
            bgr_frame_to_rgb_bytes(frame),
            colorfmt="rgb",
            bufferfmt="ubyte",
        )
        self.image_widget.texture = texture

    def _on_ocr_button_press(self, instance: Button) -> None:
        if self.last_frame is None:
            logger.warning("No frame available yet; skipping OCR")
            return

        frame = self.last_frame.copy()
        self.ocr_button.disabled = True
        threading.Thread(target=self._run_ocr, args=(frame,), daemon=True).start()

    def _run_ocr(self, frame: np.ndarray) -> None:
        try:
            img = bgr_frame_to_rgb_array(frame)
            analyzed, _ocr_vis, _layout_vis = self.analyzer(img)
            print(analyzed.model_dump_json(), flush=True)
        finally:
            Clock.schedule_once(lambda dt: setattr(self.ocr_button, "disabled", False))

    def on_stop(self) -> None:
        self.camera.release()
