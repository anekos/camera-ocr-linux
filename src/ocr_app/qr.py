import cv2
import numpy as np


def detect_qr_code(frame: np.ndarray) -> str | None:
    """フレームからQRコードを検出し、内容の文字列を返す。検出できなければNone。"""
    detector = cv2.QRCodeDetector()
    text, _points, _straight_qrcode = detector.detectAndDecode(frame)
    return text if text else None
