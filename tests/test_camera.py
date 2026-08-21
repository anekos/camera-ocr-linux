import numpy as np
import pytest

from ocr_app.camera import bgr_frame_to_rgb_array, bgr_frame_to_rgb_bytes


def test_bgr_frame_to_rgb_array_converts_color_order_without_flipping() -> None:
    # 2x2画像、BGRチャンネル順。4隅を異なる色にして、行・列の入れ替わりも検出できるようにする。
    # (0,0): 青 (255,0,0)BGR -> (0,0,255)RGB
    # (0,1): 緑 (0,255,0)BGR -> (0,255,0)RGB
    # (1,0): 赤 (0,0,255)BGR -> (255,0,0)RGB
    # (1,1): 白 (255,255,255)BGR -> (255,255,255)RGB
    frame = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )

    result = bgr_frame_to_rgb_array(frame)

    expected = np.array(
        [
            [[0, 0, 255], [0, 255, 0]],
            [[255, 0, 0], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(result, expected)


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


class _FakeVideoCapture:
    def __init__(
        self,
        opened: bool = True,
        read_result: tuple[bool, np.ndarray | None] = (True, None),
    ) -> None:
        self._opened = opened
        self._read_result = read_result
        self.released = False

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        return self._read_result

    def release(self) -> None:
        self.released = True


def test_camera_raises_when_device_cannot_be_opened(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ocr_app.camera import Camera

    monkeypatch.setattr(
        "ocr_app.camera.cv2.VideoCapture",
        lambda index: _FakeVideoCapture(opened=False),
    )

    with pytest.raises(RuntimeError, match="device index 0"):
        Camera(device_index=0)


def test_camera_read_frame_returns_none_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ocr_app.camera import Camera

    monkeypatch.setattr(
        "ocr_app.camera.cv2.VideoCapture",
        lambda index: _FakeVideoCapture(opened=True, read_result=(False, None)),
    )

    camera = Camera(device_index=0)

    assert camera.read_frame() is None


def test_camera_read_frame_returns_frame_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ocr_app.camera import Camera

    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "ocr_app.camera.cv2.VideoCapture",
        lambda index: _FakeVideoCapture(opened=True, read_result=(True, frame)),
    )

    camera = Camera(device_index=0)

    assert camera.read_frame() is frame


def test_camera_context_manager_releases_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ocr_app.camera import Camera

    fake = _FakeVideoCapture(opened=True)
    monkeypatch.setattr("ocr_app.camera.cv2.VideoCapture", lambda index: fake)

    with Camera(device_index=0):
        pass

    assert fake.released is True
