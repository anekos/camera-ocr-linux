import cv2
import numpy as np


def bgr_frame_to_rgb_bytes(frame: np.ndarray) -> bytes:
    """OpenCVのBGRフレームをKivy Texture用のRGBバイト列に変換する。

    Kivyの Texture は左下原点、OpenCV のフレームは左上原点のため、
    上下反転してから色順を BGR -> RGB に変換する。
    """
    flipped = cv2.flip(frame, 0)
    rgb = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    return rgb.tobytes()
