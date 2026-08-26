import cv2
import numpy as np

from ocr_app.qr import detect_qr_code


def _make_qr_frame(text: str) -> np.ndarray:
    encoder = cv2.QRCodeEncoder.create()
    qr = encoder.encode(text)
    big = cv2.resize(qr, (400, 400), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((500, 500), 255, dtype=np.uint8)
    canvas[50:450, 50:450] = big
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def test_detect_qr_code_returns_encoded_text() -> None:
    frame = _make_qr_frame("hello world")

    assert detect_qr_code(frame) == "hello world"


def test_detect_qr_code_returns_none_when_no_qr_code_present() -> None:
    frame = np.full((500, 500, 3), 255, dtype=np.uint8)

    assert detect_qr_code(frame) is None
