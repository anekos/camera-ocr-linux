from pathlib import Path

import cv2
import numpy as np

from ocr_app.barcode import detect_barcode

FIXTURES_DIR = Path(__file__).parent / "fixtures"
# ean13.pngの生成方法:
#   zint -b 13 -d "4901234567894" -o ean13.png --notext
#   その後、余白(クワイエットゾーン)40pxを白で四辺に追加。
# 数字ラベル付き画像やクワイエットゾーンが無い画像はcv2.barcode.BarcodeDetector
# が検出できなかったため、notext + 余白ありの構成にしている。


def test_detect_barcode_returns_decoded_text() -> None:
    frame = cv2.imread(str(FIXTURES_DIR / "ean13.png"))
    assert frame is not None

    assert detect_barcode(frame) == "4901234567894"


def test_detect_barcode_returns_none_when_no_barcode_present() -> None:
    frame = np.full((500, 500, 3), 255, dtype=np.uint8)

    assert detect_barcode(frame) is None
