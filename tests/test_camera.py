from pathlib import Path

import cv2
import numpy as np
import pytest

from ocr_app.camera import (
    bgr_frame_to_rgb_array,
    bgr_frame_to_rgb_bytes,
    crop_frame,
    encode_frame_as_png,
    flip_horizontal,
    flip_vertical,
    save_frame_as_png,
)


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


def test_flip_vertical_reverses_row_order() -> None:
    # 2x1画像（height=2, width=1）。上下反転で行0と行1が入れ替わる。
    frame = np.array(
        [
            [[255, 0, 0]],
            [[0, 255, 0]],
        ],
        dtype=np.uint8,
    )

    result = flip_vertical(frame)

    expected = np.array(
        [
            [[0, 255, 0]],
            [[255, 0, 0]],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(result, expected)


def test_flip_horizontal_reverses_column_order() -> None:
    # 1x2画像（height=1, width=2）。左右反転で列0と列1が入れ替わる。
    frame = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
        ],
        dtype=np.uint8,
    )

    result = flip_horizontal(frame)

    expected = np.array(
        [
            [[0, 255, 0], [255, 0, 0]],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(result, expected)


def test_crop_frame_extracts_the_given_fractional_box() -> None:
    # 4x4画像。box=(0.5, 0.5, 1.0, 1.0) は右下2x2を切り出す。
    frame = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)

    result = crop_frame(frame, (0.5, 0.5, 1.0, 1.0))

    np.testing.assert_array_equal(result, frame[2:4, 2:4])


def test_encode_frame_as_png_round_trips_through_cv2_imdecode() -> None:
    frame = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)

    encoded = encode_frame_as_png(frame)
    decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)

    np.testing.assert_array_equal(decoded, frame)


def test_save_frame_as_png_writes_readable_image_at_expected_path(
    tmp_path: Path,
) -> None:
    frame = np.zeros((2, 3, 3), dtype=np.uint8)

    output_path = save_frame_as_png(frame, tmp_path, "20260101-120000")

    assert output_path == tmp_path / "ocr-app-capture-20260101-120000.png"
    saved = cv2.imread(str(output_path))
    assert saved is not None
    assert saved.shape == frame.shape


def test_save_frame_as_png_creates_missing_output_dir(tmp_path: Path) -> None:
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    output_dir = tmp_path / "nested" / "captures"

    output_path = save_frame_as_png(frame, output_dir, "20260101-120000")

    assert output_path.exists()


class _FakeVideoCapture:
    def __init__(
        self,
        opened: bool = True,
        read_result: tuple[bool, np.ndarray | None] = (True, None),
    ) -> None:
        self._opened = opened
        self._read_result = read_result
        self.released = False
        self.set_calls: list[tuple[int, float]] = []

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        return self._read_result

    def release(self) -> None:
        self.released = True

    def set(self, prop_id: int, value: float) -> bool:
        self.set_calls.append((prop_id, value))
        return True


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


def test_camera_requests_mjpg_and_capture_resolution_on_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ocr_app.camera import Camera

    fake = _FakeVideoCapture(opened=True)
    monkeypatch.setattr("ocr_app.camera.cv2.VideoCapture", lambda index: fake)

    Camera(device_index=0, width=3840, height=2160)

    assert fake.set_calls == [
        (cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG")),
        (cv2.CAP_PROP_FRAME_WIDTH, 3840),
        (cv2.CAP_PROP_FRAME_HEIGHT, 2160),
    ]


def test_camera_context_manager_releases_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ocr_app.camera import Camera

    fake = _FakeVideoCapture(opened=True)
    monkeypatch.setattr("ocr_app.camera.cv2.VideoCapture", lambda index: fake)

    with Camera(device_index=0):
        pass

    assert fake.released is True
