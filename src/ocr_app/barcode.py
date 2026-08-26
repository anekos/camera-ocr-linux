import cv2
import numpy as np


def detect_barcode(frame: np.ndarray) -> str | None:
    """フレームからバーコード(EAN-13/EAN-8/UPC-A/UPC-E等)を検出し、内容の文字列を返す。

    検出できなければNone。
    """
    detector = cv2.barcode.BarcodeDetector()
    ok, decoded_info, _decoded_type, _points = detector.detectAndDecodeWithType(frame)
    if not ok or not decoded_info:
        return None
    text = decoded_info[0]
    return text if text else None
