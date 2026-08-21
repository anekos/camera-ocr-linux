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
from kivy.uix.scrollview import ScrollView
from kivy.uix.splitter import Splitter
from kivy.uix.textinput import TextInput
from yomitoku import DocumentAnalyzer

from ocr_app.camera import Camera, bgr_frame_to_rgb_array, bgr_frame_to_rgb_bytes
from ocr_app.ocr_result import extract_recognized_text

logger = logging.getLogger(__name__)

TARGET_FPS = 30
CAMERA_DEVICE_INDEX = 0
CONTROL_ROW_HEIGHT = 60
RESULT_TEXT_HEIGHT = 150
RESULT_TEXT_MIN_HEIGHT = 60
RESULT_TEXT_MAX_HEIGHT = 600
RESOLUTION_LABEL_WIDTH = 120
FONT_SIZE = 24
SCROLLBAR_WIDTH = 12
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
            font_size=FONT_SIZE,
            size_hint_y=None,
        )

        result_scroll = ScrollView(
            bar_width=SCROLLBAR_WIDTH, scroll_type=["bars", "content"]
        )
        result_scroll.add_widget(self.result_text_input)

        def _fit_result_text_height(*_args: object) -> None:
            self.result_text_input.height = max(
                self.result_text_input.minimum_height, result_scroll.height
            )

        self.result_text_input.bind(minimum_height=_fit_result_text_height)
        result_scroll.bind(height=_fit_result_text_height)

        result_splitter = Splitter(
            sizable_from="top",
            size_hint=(1, None),
            height=RESULT_TEXT_HEIGHT,
            min_size=RESULT_TEXT_MIN_HEIGHT,
            max_size=RESULT_TEXT_MAX_HEIGHT,
        )
        result_splitter.add_widget(result_scroll)

        self.copy_checkbox = CheckBox(
            active=True, size_hint_x=None, width=CONTROL_ROW_HEIGHT
        )
        copy_label = Label(
            text="クリップボードにコピー",
            font_name=JAPANESE_FONT_PATH,
            font_size=FONT_SIZE,
            size_hint_x=None,
        )
        copy_label.bind(
            texture_size=lambda instance, value: setattr(instance, "width", value[0])
        )

        self.ocr_button = Button(
            text="OCR実行", font_name=JAPANESE_FONT_PATH, font_size=FONT_SIZE
        )
        self.ocr_button.bind(on_press=self._on_ocr_button_press)

        self.resolution_label = Label(
            text="",
            font_name=JAPANESE_FONT_PATH,
            font_size=FONT_SIZE,
            size_hint_x=None,
            width=RESOLUTION_LABEL_WIDTH,
        )

        control_row = BoxLayout(
            orientation="horizontal", size_hint_y=None, height=CONTROL_ROW_HEIGHT
        )
        control_row.add_widget(self.copy_checkbox)
        control_row.add_widget(copy_label)
        control_row.add_widget(self.ocr_button)
        control_row.add_widget(self.resolution_label)

        layout = BoxLayout(orientation="vertical")
        layout.add_widget(self.image_widget)
        layout.add_widget(result_splitter)
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
        self.resolution_label.text = f"{width}x{height}"

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
