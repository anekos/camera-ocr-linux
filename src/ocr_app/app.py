import logging
import threading

import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.core.clipboard import Clipboard
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from yomitoku import DocumentAnalyzer

from ocr_app.camera import Camera, bgr_frame_to_rgb_array, bgr_frame_to_rgb_bytes
from ocr_app.ocr_result import extract_recognized_text

logger = logging.getLogger(__name__)

TARGET_FPS = 30
CAMERA_DEVICE_INDEX = 0
CONTROL_ROW_HEIGHT = 50
RESULT_TEXT_HEIGHT = 150
JAPANESE_FONT_PATH = (
    "/home/anekos/.nix-profile/share/fonts/truetype/migu/migu-1m-regular.ttf"
)


class OcrApp(App):
    last_frame: np.ndarray | None = None

    def build(self) -> BoxLayout:
        self.image_widget = Image()

        self.result_text_input = TextInput(
            readonly=True,
            font_name=JAPANESE_FONT_PATH,
            size_hint_y=None,
            height=RESULT_TEXT_HEIGHT,
        )

        self.copy_checkbox = CheckBox(
            active=True, size_hint_x=None, width=CONTROL_ROW_HEIGHT
        )
        copy_label = Label(text="クリップボードにコピー", font_name=JAPANESE_FONT_PATH)

        self.ocr_button = Button(
            text="OCR実行",
            size_hint_x=None,
            width=CONTROL_ROW_HEIGHT * 2,
            font_name=JAPANESE_FONT_PATH,
        )
        self.ocr_button.bind(on_press=self._on_ocr_button_press)

        control_row = BoxLayout(
            orientation="horizontal", size_hint_y=None, height=CONTROL_ROW_HEIGHT
        )
        control_row.add_widget(self.copy_checkbox)
        control_row.add_widget(copy_label)
        control_row.add_widget(self.ocr_button)

        layout = BoxLayout(orientation="vertical")
        layout.add_widget(self.image_widget)
        layout.add_widget(self.result_text_input)
        layout.add_widget(control_row)

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
            text = extract_recognized_text(analyzed)
            Clock.schedule_once(lambda dt: self._apply_ocr_result(text))
        finally:
            Clock.schedule_once(lambda dt: setattr(self.ocr_button, "disabled", False))

    def _apply_ocr_result(self, text: str) -> None:
        self.result_text_input.text = text
        if self.copy_checkbox.active:
            Clipboard.copy(text)

    def on_stop(self) -> None:
        self.camera.release()
