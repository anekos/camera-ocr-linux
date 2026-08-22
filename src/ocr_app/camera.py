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
        self._capture = cv2.VideoCapture(device_index)
        if not self._capture.isOpened():
            raise RuntimeError(f"Failed to open camera device index {device_index}")

        # MJPGを指定しないとUVCカメラは低解像度のYUYVにフォールバックしやすいため、
        # 高解像度キャプチャにはFOURCCの指定が必要。
        self._capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read_frame(self) -> np.ndarray | None:
        ok, frame = self._capture.read()
        if not ok:
            return None
        return frame

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
