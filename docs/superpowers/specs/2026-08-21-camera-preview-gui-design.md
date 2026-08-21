# USBカメラ映像プレビューGUI 設計

## 背景・目的

USBカメラから画像を取得しOCRを行うデスクトップアプリを開発する。本設計はその第一段階として、OCR部分を除いた「カメラ映像をリアルタイム表示するGUI部分」のみを対象とする。OCR機能は後日別途指示する。

## スコープ

- USBカメラ（`/dev/video0`、device index 0固定）の映像をウィンドウ上にリアルタイム表示する
- 画面構成はカメラプレビューのみ（ボタン・カメラ選択UI等は含まない）
- OCR機能、画像保存機能は対象外（将来の拡張として別途設計する）

## アーキテクチャ

- **GUIフレームワーク**: Kivy
  - クロスプラットフォームで、`Image`ウィジェットへの`Texture`書き込みによりリアルタイム映像表示が可能
- **カメラ取得**: `opencv-python`（`cv2.VideoCapture`）
  - KivyのCameraウィジェット標準プロバイダ（OpenCV/GStreamer経由）は使わず、`cv2.VideoCapture`から取得したフレームを直接Kivyの`Texture`に変換して表示する
  - 理由: デバイス選択・解像度設定を細かく制御でき、将来のOCR用フレーム取得とも直結しやすいため

## コンポーネント構成

```
src/ocr_app/
├── main.py       # エントリポイント（App起動）
├── app.py        # Kivy Appクラス、カメラプレビューウィジェット
└── camera.py      # OpenCVカメララッパー、フレーム変換関数
```

### `camera.py`

- カメラのオープン・フレーム取得・解放を担当するクラス（コンテキストマネージャとして実装）
- OpenCV(BGR) → Kivy Texture(RGB、上下反転)への変換を行う純粋関数を分離する
  - Kivyの`Texture`は左下原点、OpenCVのフレームは左上原点のため上下反転が必要
  - この関数はカメラ実機なしでテスト可能にする

### `app.py`

- Kivyの`App`サブクラス
- レイアウトは`Image`ウィジェット1つのみ
- `Clock.schedule_interval`で一定間隔ごとに`camera.py`からフレームを取得し、`Image.texture`を更新する
- 更新間隔は約30fps相当（カメラのfpsに応じて調整可能な形にする）

### `main.py`

- 既存の`main()`をKivy Appの起動処理に置き換える

## データフロー

```
Clockタイマー (約30fps)
  → cv2.VideoCapture.read()
  → BGR→RGB変換 + 上下反転（純粋関数）
  → Texture.blit_buffer
  → Image.texture更新
  → 画面再描画
```

## エラーハンドリング

- **起動時**: カメラが開けない場合（`VideoCapture.isOpened() == False`）は、エラーログを出力した上で明確な例外を送出しアプリを終了する。原因不明のクラッシュにしない。
- **実行中**: フレーム取得に失敗した場合は直近のフレームを表示し続け、警告ログを出す。アプリは継続動作する。

## テスト

- フレーム変換関数（BGR→RGB + 上下反転）はダミーのnumpy配列を用いたpytestの単体テストでカバーする
- カメラ結線・GUI描画自体はユニットテスト対象外とし、`uv run ocr-app`を実機で起動し、USBカメラ映像が表示されることを目視で確認する

## 依存関係の追加

`pyproject.toml`の`dependencies`に以下を追加する:

- `kivy`
- `opencv-python`

## 将来の拡張（対象外・参考情報）

- 複数カメラ対応・カメラ選択UI
- 静止画キャプチャボタン（OCR入力用）
- OCR処理本体
