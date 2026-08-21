# ocr-app

USBカメラの映像をリアルタイムに表示し、ボタン一つでOCR（[yomitoku](https://github.com/kotaro-kinoshita/yomitoku)）にかけられるLinux向けデスクトップアプリ。

## 機能

- USBカメラ（`/dev/video0`固定）の映像をリアルタイムプレビュー表示（MJPG・3840x2160@30fpsでキャプチャ）
- 「OCR実行」ボタンで、その時点のプレビュー画像に対してOCRを実行
  - OCRはバックグラウンドスレッドで実行するため、実行中もプレビューは止まらない
  - 解析結果全体（JSON）はstdoutに出力
  - 本文の認識結果は画面下部のテキストエリアに表示（見出し・ページヘッダー/フッターは除外）
  - テキストエリアの文字はドラッグで選択可能。選択すると自動でクリップボードにコピー
  - 「クリップボードにコピー」チェックボックスがONの場合、OCR結果全体も自動でクリップボードにコピー
  - ページ番号らしき数値が見つかれば、画面下部に表示（見開きなど複数候補がある場合は、画像中心から最も遠い＝外側の候補を採用）
- カメラのプレビューと結果テキストエリアの境界はドラッグでリサイズ可能

## 必要環境

- Linux（USBカメラ・ROCm GPUを利用）
- Python 3.13以上
- [uv](https://docs.astral.sh/uv/)
- AMD ROCm対応GPU（yomitokuの推論をROCm版PyTorchで実行するため）

## セットアップ

```sh
uv sync
```

依存関係にはKivy・opencv-python・yomitoku・ROCm版PyTorch（`torch`/`torchvision`/`pytorch-triton-rocm`）が含まれる。ROCm版PyTorchは `https://download.pytorch.org/whl/rocm6.4` から取得する。

## 実行

```sh
uv run ocr-app
```

初回起動時はOCRモデルの読み込み（Hugging Faceからのダウンロードを含む）に数十秒〜1分程度かかる。

## 開発

```sh
make test    # lint (mypy / ruff) + pytest
make setup   # 依存関係の同期 + pre-commitフックの導入
```

## 構成

- `src/ocr_app/main.py` — エントリポイント
- `src/ocr_app/app.py` — KivyのGUIアプリ本体（プレビュー・操作UI）
- `src/ocr_app/camera.py` — OpenCVによるカメラ制御、フレーム変換
- `src/ocr_app/ocr_result.py` — yomitokuの解析結果からの本文抽出・ページ番号抽出
