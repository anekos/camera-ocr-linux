import json
import logging
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
from app_paths import AppPaths
from kivy.app import App
from kivy.clock import Clock
from kivy.core.clipboard import Clipboard
from kivy.graphics import Color, Line
from kivy.graphics.texture import Texture
from kivy.input.motionevent import MotionEvent
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.spinner import Spinner
from kivy.uix.splitter import Splitter
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget
from yomitoku import DocumentAnalyzer
from yomitoku.document_analyzer import DocumentAnalyzerSchema

from ocr_app.camera import (
    Camera,
    bgr_frame_to_rgb_array,
    bgr_frame_to_rgb_bytes,
    crop_frame,
    encode_frame_as_png,
    flip_horizontal,
    flip_vertical,
    save_frame_as_png,
)
from ocr_app.formatting import format_selection_as_quote
from ocr_app.notifications import send_notification
from ocr_app.ocr import google_vision
from ocr_app.ocr import openai as openai_ocr
from ocr_app.ocr.yomitoku import extract_page_number, extract_recognized_text
from ocr_app.selection import (
    normalize_box,
    spread_guide_line_points,
    touch_to_image_fraction,
)
from ocr_app.settings import (
    get_bool,
    get_float,
    load_settings,
    resolve_save_directory,
    save_settings,
)

logger = logging.getLogger(__name__)

TARGET_FPS = 30
CAMERA_DEVICE_INDEX = 0
# 連続してフレーム取得に失敗した回数がこれを超えたら、切断とみなして再接続を試みる。
CAMERA_READ_FAILURE_LIMIT = 5
CAMERA_RECONNECT_INTERVAL_SEC = 1.0
CAMERA_STATUS_TEXT_WAITING = "待機中"
CAMERA_STATUS_TEXT_DISCONNECTED = "切断中"
CONTROL_ROW_HEIGHT = 60
CONTROL_ROW_SPACING = 10
RESULT_TEXT_HEIGHT = 150
RESULT_TEXT_MIN_HEIGHT = 60
RESULT_TEXT_MAX_HEIGHT = 600
FONT_SIZE = 24
SCROLLBAR_WIDTH = 12
# Kivyのデフォルト(20sp)はホイール1ノッチでの移動量が小さすぎるため広げる。
RESULT_SCROLL_WHEEL_DISTANCE = 100
JAPANESE_FONT_PATH = (
    "/home/anekos/.nix-profile/share/fonts/truetype/migu/migu-1m-regular.ttf"
)
SETTINGS_PATH = AppPaths.get_paths("ocr-app", "anekos").user_config / "settings.json"
SELECTION_CLICK_THRESHOLD = 10
ENGINE_SPINNER_WIDTH = 260

# OCRエンジンのID(設定ファイルに保存する値)と、Spinnerに表示するラベルの対応。
# 他のOCR API(GCV以外)を足すときはここに追加すればよい。
OCR_ENGINE_YOMITOKU = "yomitoku"
OCR_ENGINE_GOOGLE_VISION = "google_vision"
OPENAI_ENGINE_PREFIX = "openai:"
OPENAI_MODELS = ["gpt-5", "gpt-5-mini", "gpt-4o"]
OCR_ENGINE_LABELS = {
    OCR_ENGINE_YOMITOKU: "yomitoku",
    OCR_ENGINE_GOOGLE_VISION: "Google Cloud Vision",
    **{f"{OPENAI_ENGINE_PREFIX}{model}": f"OpenAI: {model}" for model in OPENAI_MODELS},
}
DEFAULT_OCR_ENGINE = OCR_ENGINE_YOMITOKU


class ClickableLabel(ButtonBehavior, Label):
    pass


class OcrApp(App):
    last_frame: np.ndarray | None = None
    selection_box: tuple[float, float, float, float] | None = None
    _selection_start: tuple[float, float] | None = None
    current_page_number: int | None = None
    last_yomitoku_analyzed: list[DocumentAnalyzerSchema] | None = None

    def _build_status_label(self) -> Label:
        """ステータス行用のラベルを作る。

        固定幅+text_sizeで折り返す方式は、内容によって2行になってしまう
        (例: FPS表示が90px幅に収まらず折り返される)ため使わない。
        代わりにtexture_sizeに合わせてwidthを追従させ、常に1行で
        表示されるようにする。
        """
        label = Label(
            text="",
            font_name=JAPANESE_FONT_PATH,
            font_size=FONT_SIZE,
            size_hint_x=None,
            valign="middle",
        )
        label.bind(
            texture_size=lambda instance, value: setattr(instance, "width", value[0])
        )
        label.bind(
            height=lambda instance, value: setattr(instance, "text_size", (None, value))
        )
        return label

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
        self.image_widget.bind(on_touch_down=self._on_preview_touch_down)
        self.image_widget.bind(on_touch_move=self._on_preview_touch_move)
        self.image_widget.bind(on_touch_up=self._on_preview_touch_up)
        with self.image_widget.canvas.after:
            Color(1, 0, 0, 1)
            self.selection_line = Line(width=2)
            Color(0, 1, 1, 1)
            self.spread_guide_line = Line(width=4)

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
        result_scroll = ScrollView(
            bar_width=SCROLLBAR_WIDTH,
            scroll_type=["bars"],
            scroll_wheel_distance=RESULT_SCROLL_WHEEL_DISTANCE,
        )
        result_scroll.add_widget(self.result_text_input)

        def _fit_result_text_height(*_args: object) -> None:
            self.result_text_input.height = max(
                self.result_text_input.minimum_height, result_scroll.height
            )

        self.result_text_input.bind(minimum_height=_fit_result_text_height)
        result_scroll.bind(height=_fit_result_text_height)
        self.result_text_input.bind(on_touch_up=self._on_result_text_touch_up)

        settings = load_settings(SETTINGS_PATH)

        self.result_splitter = Splitter(
            sizable_from="top",
            size_hint=(1, None),
            height=get_float(settings, "result_height", RESULT_TEXT_HEIGHT),
            min_size=RESULT_TEXT_MIN_HEIGHT,
            max_size=RESULT_TEXT_MAX_HEIGHT,
        )
        self.result_splitter.add_widget(result_scroll)
        self.result_splitter.bind(height=lambda instance, value: self._save_settings())

        self.save_directory = resolve_save_directory(
            settings, AppPaths.get_paths("ocr-app", "anekos").user_data
        )
        self.copy_checkbox, copy_label = self._build_labeled_checkbox(
            "クリップボードコピー",
            active=get_bool(settings, "copy_to_clipboard", True),
        )
        self.flip_checkbox, flip_label = self._build_labeled_checkbox(
            "反転", active=get_bool(settings, "flip", False)
        )
        self.raw_order_checkbox, raw_order_label = self._build_labeled_checkbox(
            "元の順で出力", active=get_bool(settings, "raw_order", False)
        )
        self.save_to_fixed_directory_checkbox, save_to_fixed_directory_label = (
            self._build_labeled_checkbox(
                "指定保存先",
                active=get_bool(settings, "save_to_fixed_directory", False),
            )
        )
        self.spread_checkbox, spread_label = self._build_labeled_checkbox(
            "見開き(右→左)", active=get_bool(settings, "spread", False)
        )
        for checkbox in (
            self.copy_checkbox,
            self.flip_checkbox,
            self.raw_order_checkbox,
            self.save_to_fixed_directory_checkbox,
            self.spread_checkbox,
        ):
            checkbox.bind(active=lambda instance, value: self._save_settings())
        self.raw_order_checkbox.bind(active=self._on_raw_order_toggled)

        saved_engine = settings.get("ocr_engine")
        if saved_engine not in OCR_ENGINE_LABELS:
            saved_engine = DEFAULT_OCR_ENGINE
        self.engine_spinner = Spinner(
            text=OCR_ENGINE_LABELS[saved_engine],
            values=list(OCR_ENGINE_LABELS.values()),
            font_name=JAPANESE_FONT_PATH,
            font_size=FONT_SIZE,
            size_hint_x=None,
            width=ENGINE_SPINNER_WIDTH,
        )
        self.engine_spinner.bind(text=self._on_engine_spinner_text)

        # 保存先ディレクトリが設定ファイルに無かった場合の補完値を含め、
        # 起動時点の設定を正規の形で書き戻しておく。
        self._save_settings()

        self.ocr_button = Button(
            text="OCR実行", font_name=JAPANESE_FONT_PATH, font_size=FONT_SIZE
        )
        self.ocr_button.bind(on_press=self._on_ocr_button_press)

        self.save_button = Button(
            text="画像を保存", font_name=JAPANESE_FONT_PATH, font_size=FONT_SIZE
        )
        self.save_button.bind(on_press=self._on_save_button_press)

        self.ocr_status_label = self._build_status_label()
        self.ocr_status_label.text = "OCR: --"
        self.status_separator_label = self._build_status_label()
        self.status_separator_label.text = "|"
        self.camera_status_label = self._build_status_label()

        toggle_row = BoxLayout(
            orientation="horizontal",
            size_hint_y=None,
            height=CONTROL_ROW_HEIGHT,
            spacing=CONTROL_ROW_SPACING,
        )
        toggle_row.add_widget(self.copy_checkbox)
        toggle_row.add_widget(copy_label)
        toggle_row.add_widget(self.flip_checkbox)
        toggle_row.add_widget(flip_label)
        toggle_row.add_widget(self.raw_order_checkbox)
        toggle_row.add_widget(raw_order_label)
        toggle_row.add_widget(self.save_to_fixed_directory_checkbox)
        toggle_row.add_widget(save_to_fixed_directory_label)
        toggle_row.add_widget(self.spread_checkbox)
        toggle_row.add_widget(spread_label)
        toggle_row.add_widget(self.engine_spinner)

        button_row = BoxLayout(
            orientation="horizontal",
            size_hint_y=None,
            height=CONTROL_ROW_HEIGHT,
            spacing=CONTROL_ROW_SPACING,
        )
        button_row.add_widget(self.ocr_button)
        button_row.add_widget(self.save_button)

        status_row = BoxLayout(
            orientation="horizontal",
            size_hint_y=None,
            height=CONTROL_ROW_HEIGHT,
            spacing=CONTROL_ROW_SPACING,
            padding=[CONTROL_ROW_SPACING, 0],
        )
        status_row.add_widget(Widget())  # 残り幅を埋めて、以降を右寄せにするスペーサー
        status_row.add_widget(self.ocr_status_label)
        status_row.add_widget(self.status_separator_label)
        status_row.add_widget(self.camera_status_label)

        layout = BoxLayout(orientation="vertical")
        layout.add_widget(self.image_widget)
        layout.add_widget(self.result_splitter)
        layout.add_widget(toggle_row)
        layout.add_widget(button_row)
        layout.add_widget(status_row)

        try:
            self.camera: Camera | None = Camera(device_index=CAMERA_DEVICE_INDEX)
        except RuntimeError as error:
            logger.warning("Camera not available at startup: %s", error)
            self.camera = None
        self.fps_smoothed: float | None = None
        self._capture_lock = threading.Lock()
        self._latest_captured_frame: np.ndarray | None = None
        self._capture_running = True
        self._camera_status_text: str | None = (
            None if self.camera is not None else CAMERA_STATUS_TEXT_WAITING
        )
        self.analyzer: DocumentAnalyzer | None = None
        self._update_ocr_button_state()
        threading.Thread(target=self._load_yomitoku_analyzer, daemon=True).start()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        Clock.schedule_interval(self._update, 1.0 / TARGET_FPS)
        return layout

    def _load_yomitoku_analyzer(self) -> None:
        analyzer = DocumentAnalyzer(device="cuda")

        def _on_loaded(dt: float) -> None:
            self.analyzer = analyzer
            self._update_ocr_button_state()

        Clock.schedule_once(_on_loaded)

    def _capture_loop(self) -> None:
        """UIの描画レート(TARGET_FPS)に律速されない、カメラの実キャプチャループ。

        `cv2.VideoCapture.read()` はカメラから次のフレームが届くまでブロックするため、
        ここで律速せずに回し続けることで、UIの描画キャップとは独立にカメラ自体の
        実効FPSを計測できる。
        """
        last_time = time.perf_counter()
        consecutive_failures = 0
        while self._capture_running:
            if self.camera is None:
                self._wait_for_camera_reconnect()
                consecutive_failures = 0
                last_time = time.perf_counter()
                continue

            frame = self.camera.read_frame()
            if frame is None:
                consecutive_failures += 1
                if consecutive_failures >= CAMERA_READ_FAILURE_LIMIT:
                    self._wait_for_camera_reconnect()
                    consecutive_failures = 0
                    last_time = time.perf_counter()
                continue
            consecutive_failures = 0

            now = time.perf_counter()
            elapsed, last_time = now - last_time, now
            current_fps = 1.0 / elapsed if elapsed > 0 else None

            with self._capture_lock:
                self._latest_captured_frame = frame
                self._camera_status_text = None
                if current_fps is not None:
                    self.fps_smoothed = (
                        current_fps
                        if self.fps_smoothed is None
                        else self.fps_smoothed * 0.9 + current_fps * 0.1
                    )

    def _wait_for_camera_reconnect(self) -> None:
        """カメラが利用できない間、終了せずに接続(または再接続)できるまで待機する。"""
        first_connect = self.camera is None
        if first_connect:
            logger.warning("Camera not available; waiting for it to appear")
            send_notification(
                "カメラ待機中", "カメラが見つかりません。接続をお待ちしています"
            )
        else:
            logger.warning("Camera appears disconnected; waiting for reconnection")
            send_notification(
                "カメラ切断", "カメラが切断されました。再接続を待機します"
            )

        with self._capture_lock:
            self._camera_status_text = (
                CAMERA_STATUS_TEXT_WAITING
                if first_connect
                else CAMERA_STATUS_TEXT_DISCONNECTED
            )

        while self._capture_running:
            if self.camera is None:
                try:
                    self.camera = Camera(device_index=CAMERA_DEVICE_INDEX)
                except RuntimeError:
                    pass
                else:
                    break
            elif self.camera.reconnect():
                break
            time.sleep(CAMERA_RECONNECT_INTERVAL_SEC)

        if self._capture_running:
            logger.info("Camera connected")
            if first_connect:
                send_notification("カメラ接続", "カメラが接続されました")
            else:
                send_notification("カメラ再接続", "カメラが再接続されました")

    def _touch_to_fraction(self, x: float, y: float) -> tuple[float, float] | None:
        widget = self.image_widget
        image_width, image_height = widget.norm_image_size
        return touch_to_image_fraction(
            x,
            y,
            widget.x,
            widget.y,
            widget.width,
            widget.height,
            image_width,
            image_height,
        )

    def _on_preview_touch_down(self, instance: Image, touch: MotionEvent) -> bool:
        if not self.image_widget.collide_point(*touch.pos):
            return False
        self._selection_start = (touch.x, touch.y)
        touch.grab(self.image_widget)
        return True

    def _on_preview_touch_move(self, instance: Image, touch: MotionEvent) -> bool:
        if touch.grab_current is not self.image_widget or self._selection_start is None:
            return False
        self._draw_selection_overlay(self._selection_start, (touch.x, touch.y))
        return True

    def _on_preview_touch_up(self, instance: Image, touch: MotionEvent) -> bool:
        if touch.grab_current is not self.image_widget or self._selection_start is None:
            return False
        touch.ungrab(self.image_widget)

        start = self._selection_start
        end = (touch.x, touch.y)
        self._selection_start = None

        distance = ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
        if distance < SELECTION_CLICK_THRESHOLD:
            self.selection_box = None
            self._draw_selection_overlay(None, None)
            return True

        fraction_start = self._touch_to_fraction(*start)
        fraction_end = self._touch_to_fraction(*end)
        if fraction_start is None or fraction_end is None:
            self.selection_box = None
            self._draw_selection_overlay(None, None)
            return True

        self.selection_box = normalize_box(
            fraction_start[0], fraction_start[1], fraction_end[0], fraction_end[1]
        )
        self._draw_selection_overlay(start, end)
        return True

    def _draw_selection_overlay(
        self, p1: tuple[float, float] | None, p2: tuple[float, float] | None
    ) -> None:
        if p1 is None or p2 is None:
            self.selection_line.points = []
            return
        x1, y1 = p1
        x2, y2 = p2
        self.selection_line.points = [x1, y1, x2, y1, x2, y2, x1, y2, x1, y1]

    def _crop_if_selected(self, frame: np.ndarray) -> np.ndarray:
        if self.selection_box is None:
            return frame.copy()
        return crop_frame(frame, self.selection_box).copy()

    def _selected_engine(self) -> str:
        for engine_id, label in OCR_ENGINE_LABELS.items():
            if label == self.engine_spinner.text:
                return engine_id
        return DEFAULT_OCR_ENGINE

    def _on_engine_spinner_text(self, instance: Spinner, value: str) -> None:
        self._save_settings()
        self._update_ocr_button_state()

    def _update_ocr_button_state(self) -> None:
        yomitoku_not_ready = (
            self._selected_engine() == OCR_ENGINE_YOMITOKU and self.analyzer is None
        )
        self.ocr_button.disabled = yomitoku_not_ready
        self.ocr_button.text = "yomitoku読込中..." if yomitoku_not_ready else "OCR実行"

    def _update(self, dt: float) -> None:
        with self._capture_lock:
            frame = self._latest_captured_frame
            fps = self.fps_smoothed
            status_text = self._camera_status_text

        if status_text is not None:
            # 直前のフレームがあればプレビューに残したまま、状態表示だけを切り替える。
            # 解像度・FPS表示を毎フレーム上書きし続けると、この表示がすぐ隠れてしまうため
            # 早期リターンする(未接続時はそもそもフレームが無いのでこれで問題ない)。
            self.camera_status_label.text = f"カメラ: {status_text}"
            return

        if frame is None:
            logger.warning("Failed to read frame from camera; keeping last frame")
            return

        if self.flip_checkbox.active:
            frame = flip_vertical(frame)
            frame = flip_horizontal(frame)

        self.last_frame = frame

        height, width = frame.shape[:2]
        camera_text = f"カメラ: {width}x{height}"
        if fps is not None:
            camera_text += f" {fps:.1f}fps"
        self.camera_status_label.text = camera_text

        texture = Texture.create(size=(width, height), colorfmt="rgb")
        texture.blit_buffer(
            bgr_frame_to_rgb_bytes(frame),
            colorfmt="rgb",
            bufferfmt="ubyte",
        )
        self.image_widget.texture = texture
        self._update_spread_guide_line()

    def _update_spread_guide_line(self) -> None:
        if not self.spread_checkbox.active:
            self.spread_guide_line.points = []
            return
        widget = self.image_widget
        image_width, image_height = widget.norm_image_size
        x1, y1, x2, y2 = spread_guide_line_points(
            widget.x, widget.y, widget.width, widget.height, image_width, image_height
        )
        self.spread_guide_line.points = [x1, y1, x2, y2]

    def _on_ocr_button_press(self, instance: Button) -> None:
        if self.last_frame is None:
            logger.warning("No frame available yet; skipping OCR")
            return

        frame = self._crop_if_selected(self.last_frame)
        sort_output = not self.raw_order_checkbox.active
        engine = self._selected_engine()
        spread = self.spread_checkbox.active
        self.ocr_button.disabled = True
        threading.Thread(
            target=self._run_ocr,
            args=(frame, sort_output, engine, spread),
            daemon=True,
        ).start()

    def _run_ocr(
        self, frame: np.ndarray, sort_output: bool, engine: str, spread: bool
    ) -> None:
        start_time = time.perf_counter()
        try:
            if spread:
                # 見開き2ページを一度にOCRさせると、yomitokuが段をページを
                # またいで誤って結び付けてしまうため、右半分・左半分を
                # 別々にOCRしてから連結する(縦書きの綴じ方向を前提に右→左)。
                right_text, right_page, right_analyzed = self._run_single_ocr(
                    crop_frame(frame, (0.5, 0.0, 1.0, 1.0)), sort_output, engine
                )
                left_text, left_page, left_analyzed = self._run_single_ocr(
                    crop_frame(frame, (0.0, 0.0, 0.5, 1.0)), sort_output, engine
                )
                # ページ内の段落区切りは"\n"のため、ページの境目は空行("\n\n")
                # にして区別できるようにする。
                text = "\n\n".join(t for t in (right_text, left_text) if t)
                page_number = right_page if right_page is not None else left_page
                analyzed_list = [
                    a for a in (right_analyzed, left_analyzed) if a is not None
                ] or None
            else:
                text, page_number, analyzed = self._run_single_ocr(
                    frame, sort_output, engine
                )
                analyzed_list = [analyzed] if analyzed is not None else None
            duration = time.perf_counter() - start_time
            Clock.schedule_once(
                lambda dt: self._apply_ocr_result(
                    text, page_number, analyzed_list, duration
                )
            )
        except Exception:
            logger.exception("OCR processing failed")
            send_notification(
                "OCR失敗",
                "予期しないエラーが発生しました。詳細はログを確認してください。",
            )
        finally:
            Clock.schedule_once(lambda dt: self._update_ocr_button_state())

    def _run_single_ocr(
        self, frame: np.ndarray, sort_output: bool, engine: str
    ) -> tuple[str, int | None, DocumentAnalyzerSchema | None]:
        if engine == OCR_ENGINE_GOOGLE_VISION:
            text, page_number = self._run_google_vision_ocr(frame)
            return text, page_number, None
        elif engine == OCR_ENGINE_YOMITOKU:
            return self._run_yomitoku_ocr(frame, sort_output)
        elif engine.startswith(OPENAI_ENGINE_PREFIX):
            model = engine.removeprefix(OPENAI_ENGINE_PREFIX)
            text, page_number = self._run_openai_ocr(frame, model)
            return text, page_number, None
        logger.warning("Unknown OCR engine: %s", engine)
        return "", None, None

    def _run_yomitoku_ocr(
        self, frame: np.ndarray, sort_output: bool
    ) -> tuple[str, int | None, DocumentAnalyzerSchema]:
        assert self.analyzer is not None, "yomitoku analyzer is not loaded yet"
        img = bgr_frame_to_rgb_array(frame)
        analyzed, _ocr_vis, _layout_vis = self.analyzer(img)
        print(analyzed.model_dump_json(), flush=True)
        text = extract_recognized_text(analyzed, sort=sort_output)
        image_height, image_width = img.shape[:2]
        page_number = extract_page_number(analyzed, image_width, image_height)
        return text, page_number, analyzed

    def _run_google_vision_ocr(self, frame: np.ndarray) -> tuple[str, int | None]:
        api_key = google_vision.get_api_key()
        if api_key is None:
            logger.warning(
                "%s is not set; skipping Google Vision OCR",
                google_vision.API_KEY_ENV_VAR,
            )
            send_notification(
                "Google Cloud Vision",
                f"環境変数 {google_vision.API_KEY_ENV_VAR} が設定されていません",
            )
            return "", None

        image_bytes = encode_frame_as_png(frame)
        try:
            response = google_vision.analyze(image_bytes, api_key)
        except httpx.HTTPError as error:
            logger.warning("Google Vision API request failed: %s", error)
            send_notification(
                "Google Cloud Vision", f"リクエストに失敗しました: {error}"
            )
            return "", None

        print(json.dumps(response), flush=True)
        text = google_vision.extract_recognized_text(response)
        height, width = frame.shape[:2]
        page_number = google_vision.extract_page_number(response, width, height)
        return text, page_number

    def _run_openai_ocr(self, frame: np.ndarray, model: str) -> tuple[str, int | None]:
        api_key = openai_ocr.get_api_key()
        if api_key is None:
            logger.warning(
                "%s is not set; skipping OpenAI OCR", openai_ocr.API_KEY_ENV_VAR
            )
            send_notification(
                "OpenAI", f"環境変数 {openai_ocr.API_KEY_ENV_VAR} が設定されていません"
            )
            return "", None

        image_bytes = encode_frame_as_png(frame)
        try:
            response = openai_ocr.analyze(image_bytes, api_key, model)
        except httpx.HTTPError as error:
            logger.warning("OpenAI API request failed: %s", error)
            send_notification("OpenAI", f"リクエストに失敗しました: {error}")
            return "", None

        print(json.dumps(response), flush=True)
        text = openai_ocr.extract_recognized_text(response)
        page_number = openai_ocr.extract_page_number(response)
        return text, page_number

    def _apply_ocr_result(
        self,
        text: str,
        page_number: int | None,
        analyzed_list: list[DocumentAnalyzerSchema] | None,
        duration: float,
    ) -> None:
        self.result_text_input.text = text
        self.current_page_number = page_number
        self.last_yomitoku_analyzed = analyzed_list
        self._update_ocr_status_label(duration)
        if self.copy_checkbox.active:
            Clipboard.copy(text)
        send_notification("OCR完了", f"{len(text)}文字を認識しました")

    def _update_ocr_status_label(self, duration: float) -> None:
        parts = []
        if self.current_page_number is not None:
            parts.append(f"P.{self.current_page_number}")
        parts.append(f"{duration:.1f}秒")
        self.ocr_status_label.text = f"OCR: {' '.join(parts)}"

    def _on_raw_order_toggled(self, instance: CheckBox, value: bool) -> None:
        if not self.last_yomitoku_analyzed:
            return
        sort_output = not value
        texts = [
            extract_recognized_text(analyzed, sort=sort_output)
            for analyzed in self.last_yomitoku_analyzed
        ]
        self.result_text_input.text = "\n\n".join(t for t in texts if t)

    def _on_save_button_press(self, instance: Button) -> None:
        if self.last_frame is None:
            logger.warning("No frame available yet; skipping save")
            return

        frame = self._crop_if_selected(self.last_frame)
        output_dir = (
            self.save_directory
            if self.save_to_fixed_directory_checkbox.active
            else Path(tempfile.gettempdir())
        )
        timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
        output_path = save_frame_as_png(frame, output_dir, timestamp)
        Clipboard.copy(str(output_path))
        send_notification("画像を保存しました", str(output_path))

    def _save_settings(self) -> None:
        save_settings(
            SETTINGS_PATH,
            {
                "copy_to_clipboard": self.copy_checkbox.active,
                "flip": self.flip_checkbox.active,
                "raw_order": self.raw_order_checkbox.active,
                "save_to_fixed_directory": self.save_to_fixed_directory_checkbox.active,
                "spread": self.spread_checkbox.active,
                "save_directory": str(self.save_directory),
                "ocr_engine": self._selected_engine(),
                "result_height": self.result_splitter.height,
            },
        )

    def _on_result_text_touch_up(self, instance: TextInput, touch: MotionEvent) -> bool:
        if instance.selection_text:
            Clipboard.copy(
                format_selection_as_quote(
                    instance.selection_text, self.current_page_number
                )
            )
        return False

    def on_stop(self) -> None:
        self._capture_running = False
        # キャプチャスレッドが self.camera に触れなくなるまで待ってから release する。
        # 待たずに release すると、スレッドがまだ読み取り/再接続の途中でカメラの
        # capture オブジェクトへ同時アクセスすることになり、デバイスが正常に
        # 閉じられずカメラが使用不能になることがある。
        self._capture_thread.join()
        if self.camera is not None:
            self.camera.release()
