import numpy as np

from ocr_app.camera import bgr_frame_to_rgb_bytes


def test_bgr_frame_to_rgb_bytes_converts_color_order_and_flips_vertically() -> None:
    # 2x1画像（height=2, width=1）、BGRチャンネル順。
    # 行0（上）: BGRで(255, 0, 0) = 青 -> RGBでは(0, 0, 255)
    # 行1（下）: BGRで(0, 255, 0) = 緑 -> RGBでは(0, 255, 0)
    frame = np.array(
        [
            [[255, 0, 0]],
            [[0, 255, 0]],
        ],
        dtype=np.uint8,
    )

    result = bgr_frame_to_rgb_bytes(frame)

    # 上下反転により、元の行1（緑）が先頭、元の行0（青）が末尾になる。
    expected = bytes([0, 255, 0]) + bytes([0, 0, 255])
    assert result == expected
