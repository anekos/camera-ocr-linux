# カメラ映像プレビューGUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** USBカメラ（`/dev/video0`、device index 0）の映像を、Kivyウィンドウ上にリアルタイムでプレビュー表示するデスクトップアプリを作る。OCR機能はこの計画の対象外。

**Architecture:** `opencv-python`の`cv2.VideoCapture`でフレームを取得し、BGR→RGB変換＋上下反転を行う純粋関数でKivyの`Texture`用バイト列に変換、`Clock.schedule_interval`で定期的に`Image`ウィジェットのテクスチャを更新する。カメラ結線とフレーム変換ロジックを分離し、変換ロジックのみを実機なしで単体テストする。

**Tech Stack:** Python 3.13, Kivy, opencv-python, numpy, pytest, uv

**Spec:** `docs/superpowers/specs/2026-08-21-camera-preview-gui-design.md`

## Global Constraints

- カメラデバイスは `/dev/video0`（device index `0`）に固定する。デバイス選択UIは作らない。
- 画面構成はカメラプレビューのみ。ボタンや他のUI要素は追加しない。
- 依存関係には `kivy`, `opencv-python`, `numpy` を追加する（`numpy`はコード内で`np.ndarray`型として直接importするため、opencv-pythonの推移的依存であっても明示的に追加する）。
- フレーム変換は BGR→RGB変換 + 上下反転（Kivyの`Texture`は左下原点、OpenCVフレームは左上原点のため）を行う。
- 起動時にカメラが開けない場合は、明確な例外（`RuntimeError`）を送出してアプリを終了する。原因不明のクラッシュにしない。
- 実行中にフレーム取得に失敗した場合は、直近のフレームを表示し続けアプリを継続動作させる（例外を送出しない）。
- カメラ結線・GUI描画自体は自動テスト対象外とし、実機での目視確認で検証する。

---

### Task 1: 依存関係の追加

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`（`uv add`により自動更新）

**Interfaces:**
- Consumes: なし
- Produces: `kivy`, `opencv-python`, `numpy` パッケージがプロジェクトの依存関係として利用可能になる

- [ ] **Step 1: 依存関係を追加する**

Run: `uv add kivy opencv-python numpy`

Expected: コマンドが成功し、`pyproject.toml`の`dependencies`に`kivy`, `opencv-python`, `numpy`が追加され、`uv.lock`が更新される。

- [ ] **Step 2: インポートできることを確認する**

Run: `uv run python -c "import kivy, cv2, numpy; print('ok')"`

Expected: `ok`が出力され、エラーが出ない。

- [ ] **Step 3: コミットする**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add kivy, opencv-python, numpy dependencies"
```

---

### Task 2: フレーム変換の純粋関数（BGR→RGB + 上下反転）

**Files:**
- Create: `src/ocr_app/camera.py`
- Create: `tests/test_camera.py`

**Interfaces:**
- Consumes: `numpy.ndarray`（形状 `(height, width, 3)`, dtype `uint8`, BGRチャンネル順、OpenCVの`VideoCapture.read()`が返すフレームと同じ形式）
- Produces: `bgr_frame_to_rgb_bytes(frame: np.ndarray) -> bytes` — Task 4で`Texture.blit_buffer`に渡すバイト列を生成する関数として使われる

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_camera.py`を作成する:

```python
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
```

- [ ] **Step 2: テストが失敗することを確認する**

Run: `uv run pytest tests/test_camera.py -v`

Expected: FAIL（`ModuleNotFoundError: No module named 'ocr_app.camera'` または `ImportError`）

- [ ] **Step 3: 最小限の実装を書く**

`src/ocr_app/camera.py`を作成する:

```python
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
```

- [ ] **Step 4: テストが通ることを確認する**

Run: `uv run pytest tests/test_camera.py -v`

Expected: PASS

- [ ] **Step 5: コミットする**

```bash
git add src/ocr_app/camera.py tests/test_camera.py
git commit -m "feat: add BGR-to-RGB frame conversion for camera preview"
```

---

### Task 3: `Camera`クラス（オープン・フレーム取得・解放）

**Files:**
- Modify: `src/ocr_app/camera.py`
- Modify: `tests/test_camera.py`

**Interfaces:**
- Consumes: `cv2.VideoCapture`（`ocr_app.camera.cv2.VideoCapture`としてテストでモンキーパッチ可能）
- Produces:
  - `Camera(device_index: int = 0)` — コンストラクタ。開けない場合は`RuntimeError`を送出
  - `Camera.read_frame() -> np.ndarray | None` — 取得失敗時は`None`を返す
  - `Camera.release() -> None`
  - `Camera`はコンテキストマネージャ（`__enter__`/`__exit__`で`release()`を呼ぶ）。この`__enter__`/`__exit__`はテスト（`with Camera(...):`）で使うためのものであり、Task 4のKivyアプリでは`build()`でインスタンス化し`on_stop()`で明示的に`release()`を呼ぶ（Appのライフサイクルが`with`ブロックと噛み合わないため）
  - Task 4では `Camera(device_index=CAMERA_DEVICE_INDEX)` を直接インスタンス化し、`self.camera.release()` を`on_stop()`で呼ぶ形で使われる

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_camera.py`に追記する:

```python
import pytest


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
```

- [ ] **Step 2: テストが失敗することを確認する**

Run: `uv run pytest tests/test_camera.py -v`

Expected: FAIL（`ImportError: cannot import name 'Camera' from 'ocr_app.camera'`）

- [ ] **Step 3: 最小限の実装を書く**

`src/ocr_app/camera.py`に追記する:

```python
from types import TracebackType


class Camera:
    def __init__(self, device_index: int = 0) -> None:
        self._capture = cv2.VideoCapture(device_index)
        if not self._capture.isOpened():
            raise RuntimeError(f"Failed to open camera device index {device_index}")

    def read_frame(self) -> np.ndarray | None:
        ok, frame = self._capture.read()
        if not ok:
            return None
        return frame

    def release(self) -> None:
        self._capture.release()

    def __enter__(self) -> "Camera":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()
```

- [ ] **Step 4: テストが通ることを確認する**

Run: `uv run pytest tests/test_camera.py -v`

Expected: PASS（5件すべて成功）

- [ ] **Step 5: 型チェック・リントを確認する**

Run: `uv run mypy . && uv run ruff check`

Expected: エラーなし

- [ ] **Step 6: コミットする**

```bash
git add src/ocr_app/camera.py tests/test_camera.py
git commit -m "feat: add Camera class for opening/reading/releasing device"
```

---

### Task 4: KivyアプリでのカメラプレビューGUI

**Files:**
- Create: `src/ocr_app/app.py`
- Modify: `src/ocr_app/main.py`

**Interfaces:**
- Consumes:
  - `ocr_app.camera.Camera`（Task 3）
  - `ocr_app.camera.bgr_frame_to_rgb_bytes`（Task 2）
- Produces: `ocr_app.app.OcrApp`（Kivy `App`サブクラス）、`ocr_app.main.main() -> None`

- [ ] **Step 1: Kivy Appを実装する**

`src/ocr_app/app.py`を作成する:

```python
import logging

from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image

from ocr_app.camera import Camera, bgr_frame_to_rgb_bytes

logger = logging.getLogger(__name__)

TARGET_FPS = 30
CAMERA_DEVICE_INDEX = 0


class OcrApp(App):
    def build(self) -> Image:
        self.image_widget = Image()
        self.camera = Camera(device_index=CAMERA_DEVICE_INDEX)
        Clock.schedule_interval(self._update, 1.0 / TARGET_FPS)
        return self.image_widget

    def _update(self, dt: float) -> None:
        frame = self.camera.read_frame()
        if frame is None:
            logger.warning("Failed to read frame from camera; keeping last frame")
            return

        height, width = frame.shape[:2]
        texture = Texture.create(size=(width, height), colorfmt="rgb")
        texture.blit_buffer(
            bgr_frame_to_rgb_bytes(frame),
            colorfmt="rgb",
            bufferfmt="ubyte",
        )
        self.image_widget.texture = texture

    def on_stop(self) -> None:
        self.camera.release()
```

- [ ] **Step 2: エントリポイントを書き換える**

`src/ocr_app/main.py`を以下の内容に置き換える:

```python
from ocr_app.app import OcrApp


def main() -> None:
    OcrApp().run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 型チェック・リントを確認する**

Run: `uv run mypy . && uv run ruff check`

Expected: エラーなし

- [ ] **Step 4: 単体テストが引き続き通ることを確認する**

Run: `uv run pytest -v`

Expected: 全テストPASS（`app.py`自体は自動テスト対象外）

- [ ] **Step 5: コミットする**

```bash
git add src/ocr_app/app.py src/ocr_app/main.py
git commit -m "feat: display realtime USB camera preview in Kivy window"
```

- [ ] **Step 6: 実機で目視確認する**

Run: `uv run ocr-app`

Expected:
- ウィンドウが開き、USBカメラ（`/dev/video0`）の映像がリアルタイムに表示される
- カメラの前で手を動かすなどして、映像が追従して更新されることを確認する
- ウィンドウを閉じたとき、例外を出さずにプロセスが正常終了する

この手動確認結果をユーザーに報告する。
