import logging
import tempfile
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.core.clipboard import Clipboard
from kivy.graphics.texture import Texture
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.splitter import Splitter
from kivy.uix.textinput import TextInput
from yomitoku import DocumentAnalyzer

from ocr_app.camera import (
    Camera,
    bgr_frame_to_rgb_array,
    bgr_frame_to_rgb_bytes,
    flip_horizontal,
    flip_vertical,
    save_frame_as_png,
)
from ocr_app.ocr_result import extract_page_number, extract_recognized_text

logger = logging.getLogger(__name__)

TARGET_FPS = 30
CAMERA_DEVICE_INDEX = 0
CONTROL_ROW_HEIGHT = 60
CONTROL_ROW_SPACING = 10
RESULT_TEXT_HEIGHT = 150
RESULT_TEXT_MIN_HEIGHT = 60
RESULT_TEXT_MAX_HEIGHT = 600
RESOLUTION_LABEL_WIDTH = 120
PAGE_NUMBER_LABEL_WIDTH = 120
FONT_SIZE = 24
SCROLLBAR_WIDTH = 12
JAPANESE_FONT_PATH = (
    "/home/anekos/.nix-profile/share/fonts/truetype/migu/migu-1m-regular.ttf"
)


class ClickableLabel(ButtonBehavior, Label):
    pass


class OcrApp(App):
    last_frame: np.ndarray | None = None

    def _build_labeled_checkbox(
        self, text: str, active: bool
    ) -> tuple[CheckBox, ClickableLabel]:
        checkbox = CheckBox(active=active, size_hint_x=None, width=CONTROL_ROW_HEIGHT)
        label = ClickableLabel(
            text=text,
            font_name=JAPANESE_FONT_PATH,
            font_size=FONT_SIZE,
            size_hint_x=None,
        )
        label.bind(
            texture_size=lambda instance, value: setattr(instance, "width", value[0])
        )
        label.bind(
            on_release=lambda instance: setattr(checkbox, "active", not checkbox.active)
        )
        return checkbox, label

    def build(self) -> BoxLayout:
        self.image_widget = Image()

        self.result_text_input = TextInput(
            readonly=True,
            font_name=JAPANESE_FONT_PATH,
            font_size=FONT_SIZE,
            size_hint_y=None,
        )

        # scroll_typeに"content"を含めると、テキストエリア上のドラッグ操作が
        # 常にスクロール判定に取られ、TextInputのドラッグによる範囲選択が
        # 効かなくなる。"bars"のみにすることで、スクロールバー以外への
        # タッチはTextInputへそのまま渡され、選択操作が機能する。
        result_scroll = ScrollView(bar_width=SCROLLBAR_WIDTH, scroll_type=["bars"])
        result_scroll.add_widget(self.result_text_input)

        def _fit_result_text_height(*_args: object) -> None:
            self.result_text_input.height = max(
                self.result_text_input.minimum_height, result_scroll.height
            )

        self.result_text_input.bind(minimum_height=_fit_result_text_height)
        result_scroll.bind(height=_fit_result_text_height)
        self.result_text_input.bind(
            selection_text=self._on_result_selection_text_changed
        )

        result_splitter = Splitter(
            sizable_from="top",
            size_hint=(1, None),
            height=RESULT_TEXT_HEIGHT,
            min_size=RESULT_TEXT_MIN_HEIGHT,
            max_size=RESULT_TEXT_MAX_HEIGHT,
        )
        result_splitter.add_widget(result_scroll)

        self.copy_checkbox, copy_label = self._build_labeled_checkbox(
            "クリップボードにコピー", active=True
        )
        self.flip_checkbox, flip_label = self._build_labeled_checkbox(
            "反転", active=False
        )

        self.ocr_button = Button(
            text="OCR実行", font_name=JAPANESE_FONT_PATH, font_size=FONT_SIZE
        )
        self.ocr_button.bind(on_press=self._on_ocr_button_press)

        self.save_button = Button(
            text="画像を保存", font_name=JAPANESE_FONT_PATH, font_size=FONT_SIZE
        )
        self.save_button.bind(on_press=self._on_save_button_press)

        self.resolution_label = Label(
            text="",
            font_name=JAPANESE_FONT_PATH,
            font_size=FONT_SIZE,
            size_hint_x=None,
            width=RESOLUTION_LABEL_WIDTH,
        )

        self.page_number_label = Label(
            text="",
            font_name=JAPANESE_FONT_PATH,
            font_size=FONT_SIZE,
            size_hint_x=None,
            width=PAGE_NUMBER_LABEL_WIDTH,
        )

        control_row = BoxLayout(
            orientation="horizontal",
            size_hint_y=None,
            height=CONTROL_ROW_HEIGHT,
            spacing=CONTROL_ROW_SPACING,
        )
        control_row.add_widget(self.copy_checkbox)
        control_row.add_widget(copy_label)
        control_row.add_widget(self.flip_checkbox)
        control_row.add_widget(flip_label)
        control_row.add_widget(self.ocr_button)
        control_row.add_widget(self.save_button)
        control_row.add_widget(self.page_number_label)
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

        if self.flip_checkbox.active:
            frame = flip_vertical(frame)
            frame = flip_horizontal(frame)

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
            image_height, image_width = img.shape[:2]
            page_number = extract_page_number(analyzed, image_width, image_height)
            Clock.schedule_once(lambda dt: self._apply_ocr_result(text, page_number))
        finally:
            Clock.schedule_once(lambda dt: setattr(self.ocr_button, "disabled", False))

    def _apply_ocr_result(self, text: str, page_number: int | None) -> None:
        self.result_text_input.text = text
        self.page_number_label.text = (
            f"ページ: {page_number}" if page_number is not None else ""
        )
        if self.copy_checkbox.active:
            Clipboard.copy(text)

    def _on_save_button_press(self, instance: Button) -> None:
        if self.last_frame is None:
            logger.warning("No frame available yet; skipping save")
            return

        timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
        output_path = save_frame_as_png(
            self.last_frame, Path(tempfile.gettempdir()), timestamp
        )
        Clipboard.copy(str(output_path))

    def _on_result_selection_text_changed(
        self, instance: TextInput, value: str
    ) -> None:
        if value:
            Clipboard.copy(value)

    def on_stop(self) -> None:
        self.camera.release()
