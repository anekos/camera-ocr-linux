import os

os.environ.setdefault("SDL_VIDEO_X11_WMCLASS", "ocr-app.snca.net")

from ocr_app.app import OcrApp


def main() -> None:
    OcrApp().run()


if __name__ == "__main__":
    main()
