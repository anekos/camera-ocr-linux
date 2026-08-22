import os

os.environ.setdefault("SDL_VIDEO_X11_WMCLASS", "ocr-app.snca.net")

from kivy.config import Config

# 右クリック等をマルチタッチのシミュレーションとして扱い赤い円を表示する
# Kivyの標準機能を無効化する。範囲選択のドラッグ操作と紛らわしいため。
Config.set("input", "mouse", "mouse,disable_multitouch")

from ocr_app.app import OcrApp


def main() -> None:
    OcrApp().run()


if __name__ == "__main__":
    main()
