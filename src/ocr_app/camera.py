from pathlib import Path
from types import TracebackType
from typing import Self

import cv2
import numpy as np


def bgr_frame_to_rgb_array(frame: np.ndarray) -> np.ndarray:
    """OpenCVのBGRフレームをRGBのnumpy配列に変換する（上下反転なし）。

    OCR (yomitoku) はKivyのTextureと異なり画像の上下反転を必要としないため、
    色順の変換のみを行う。
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def bgr_frame_to_rgb_bytes(frame: np.ndarray) -> bytes:
    """OpenCVのBGRフレームをKivy Texture用のRGBバイト列に変換する。

    Kivyの Texture は左下原点、OpenCV のフレームは左上原点のため、
    上下反転してから色順を BGR -> RGB に変換する。
    """
    flipped = cv2.flip(frame, 0)
    rgb = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    return rgb.tobytes()


def flip_vertical(frame: np.ndarray) -> np.ndarray:
    """フレームを上下反転する（カメラが上下逆さに取り付けられている場合の補正用）。"""
    return cv2.flip(frame, 0)


def flip_horizontal(frame: np.ndarray) -> np.ndarray:
    """フレームを左右反転する（カメラの取り付け向きの補正用）。"""
    return cv2.flip(frame, 1)


def crop_frame(frame: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    """boxで指定された相対矩形(left, top, right, bottom、いずれも0〜1)でフレームを切り出す。"""
    left, top, right, bottom = box
    height, width = frame.shape[:2]
    x1, x2 = int(left * width), int(right * width)
    y1, y2 = int(top * height), int(bottom * height)
    return frame[y1:y2, x1:x2]


def encode_frame_as_png(frame: np.ndarray) -> bytes:
    """フレームをPNG形式にエンコードしたバイト列を返す(外部APIへのアップロード用)。"""
    _ok, buffer = cv2.imencode(".png", frame)
    return buffer.tobytes()


def save_frame_as_png(frame: np.ndarray, output_dir: Path, timestamp: str) -> Path:
    """フレームをPNGとして output_dir に保存し、保存先のパスを返す。output_dir が無ければ作成する。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"ocr-app-capture-{timestamp}.png"
    cv2.imwrite(str(output_path), frame)
    return output_path


DEFAULT_CAPTURE_WIDTH = 3840
DEFAULT_CAPTURE_HEIGHT = 2160


class Camera:
    def __init__(
        self,
        device_index: int = 0,
        width: int = DEFAULT_CAPTURE_WIDTH,
        height: int = DEFAULT_CAPTURE_HEIGHT,
    ) -> None:
        self._device_index = device_index
        self._width = width
        self._height = height

        capture = self._open_capture()
        if capture is None:
            raise RuntimeError(f"Failed to open camera device index {device_index}")
        self._capture = capture

    def _open_capture(self) -> cv2.VideoCapture | None:
        capture = cv2.VideoCapture(self._device_index)
        if not capture.isOpened():
            capture.release()
            return None

        # MJPGを指定しないとUVCカメラは低解像度のYUYVにフォールバックしやすいため、
        # 高解像度キャプチャにはFOURCCの指定が必要。
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        return capture

    def read_frame(self) -> np.ndarray | None:
        try:
            ok, frame = self._capture.read()
        except cv2.error:
            return None
        if not ok:
            return None
        return frame

    def reconnect(self) -> bool:
        """カメラを閉じて再オープンを試みる。切断からの復帰用。成功したらTrue。"""
        self._capture.release()
        capture = self._open_capture()
        if capture is None:
            return False
        self._capture = capture
        return True

    def release(self) -> None:
        self._capture.release()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()
