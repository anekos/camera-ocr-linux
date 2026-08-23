# ocr-app

USBカメラの映像をリアルタイムに表示し、ボタン一つでOCR（[yomitoku](https://github.com/kotaro-kinoshita/yomitoku)、またはGoogle Cloud Vision）にかけられるLinux向けデスクトップアプリ。

## 機能

- USBカメラ（`/dev/video0`固定）の映像をリアルタイムプレビュー表示（MJPG・3840x2160@30fpsでキャプチャ）
- プレビュー上をドラッグすると範囲選択でき、OCR・画像保存は選択範囲のみを対象にする（クリックだけなら選択解除）
- 「反転」チェックボックスで、カメラの取り付け向きに合わせて上下左右反転
- OCRエンジンをSpinnerで選択（yomitoku / Google Cloud Vision）
- 「OCR実行」ボタンで、その時点のプレビュー画像（選択範囲があればそれ）に対してOCRを実行
  - OCRはバックグラウンドスレッドで実行するため、実行中もプレビューは止まらない
  - 解析結果全体（JSON）はstdoutに出力
  - 本文の認識結果は画面下部のテキストエリアに表示
    - yomitokuの場合、見出し・ページヘッダー/フッターは本文から除外。「元の順で出力」チェックボックスがONなら並び替えをせず解析結果の元の順序で出力。このチェックボックスは再OCRせずに直前の解析結果からテキストエリアを書き直す
  - ページ番号らしき数値が見つかれば、画面下部のステータス行に表示
    - yomitoku: page_header/page_footerロールの段落から、画像中心から最も遠い（＝外側の）候補を採用
    - Google Cloud Vision: 数字のみのtextAnnotationを候補にし、同様に画像中心から最も遠いものを採用（role分類が無いための代替ロジック。本文中の数字を誤検出する可能性あり）
  - 「見開き(右→左)」チェックボックスがONの場合、見開き2ページを画像の左右半分に分割してそれぞれ個別にOCRし、右半分→左半分の順にテキストを連結する（yomitokuは見開き全体を一度にOCRすると段の並びがページをまたいで乱れるため）。プレビューには分割位置の目安としてシアン色の縦ガイド線を表示。ページ境界は空行で区切り、ページ内の段落改行と区別できるようにしている
- テキストエリアの文字はドラッグで選択可能。選択を終えると自動でクリップボードにコピーする。コピーされる内容は引用形式 `` `連結したテキスト` P.<ページ番号> ``（改行は連結して1行にし、ページ番号が無ければ` P.`部分は省略）
- 「クリップボードコピー」チェックボックスがONの場合、OCR結果全体も自動でクリップボードにコピー
- 「画像を保存」ボタンで、プレビュー画像（選択範囲があればそれ）をPNGとして保存し、保存先パスをクリップボードにコピー。「指定保存先」チェックボックスで、保存先を一時ディレクトリ／固定ディレクトリのどちらにするか切り替え
- OCR完了時・画像保存時にデスクトップ通知（`notify-send`）を送信
- 各チェックボックスの状態・選択中のOCRエンジン・保存先ディレクトリ・結果表示エリアの高さは `~/.config/ocr-app/settings.json` に自動保存し、起動時に復元
- カメラのプレビューと結果テキストエリアの境界はドラッグでリサイズ可能（高さは設定に保存され、次回起動時に復元）
- 結果テキストエリアはマウスホイールでスクロール可能（Kivyのデフォルトより大きい移動量に調整済み）

## 環境変数

- `OCR_APP_GOOGLE_API_KEY` — Google Cloud VisionをOCRエンジンとして使う場合のAPIキー。未設定の場合はGoogle Cloud Vision実行時にデスクトップ通知でエラーを表示する（yomitokuのみを使う場合は不要）

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
- `src/ocr_app/selection.py` — プレビュー上のタッチ座標と画像内相対座標の変換、見開きガイド線の座標計算
- `src/ocr_app/settings.py` — 設定ファイルの読み書き
- `src/ocr_app/notifications.py` — デスクトップ通知（`notify-send`）
- `src/ocr_app/formatting.py` — テキストエリア選択時のコピー内容（引用形式）の整形
- `src/ocr_app/ocr/yomitoku.py` — yomitokuの解析結果からの本文抽出・ページ番号抽出
- `src/ocr_app/ocr/google_vision.py` — Google Cloud Vision APIクライアント、本文抽出・ページ番号抽出
