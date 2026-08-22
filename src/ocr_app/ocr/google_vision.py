import base64
import os

import httpx

API_KEY_ENV_VAR = "OCR_APP_GOOGLE_API_KEY"
VISION_API_URL = "https://vision.googleapis.com/v1/images:annotate"
REQUEST_TIMEOUT = 30.0


def get_api_key() -> str | None:
    """環境変数からGoogle Cloud Vision APIキーを取得する。未設定ならNone。"""
    return os.environ.get(API_KEY_ENV_VAR) or None


def build_request_body(image_bytes: bytes) -> dict:
    """images:annotate に送るリクエストボディを組み立てる(TEXT_DETECTION固定)。"""
    content = base64.b64encode(image_bytes).decode("ascii")
    return {
        "requests": [
            {
                "image": {"content": content},
                "features": [{"type": "TEXT_DETECTION"}],
            }
        ]
    }


def analyze(image_bytes: bytes, api_key: str) -> dict:
    """Google Cloud Vision APIを呼び出し、レスポンスJSONを返す。

    HTTPエラー時はhttpx.HTTPStatusErrorを送出する(呼び出し側で処理する)。
    """
    response = httpx.post(
        VISION_API_URL,
        params={"key": api_key},
        json=build_request_body(image_bytes),
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    result: dict = response.json()
    return result


def extract_recognized_text(response: dict) -> str:
    """レスポンスから認識済みテキスト全体(先頭のtextAnnotation)を取り出す。"""
    responses = response.get("responses", [])
    if not responses:
        return ""
    annotations = responses[0].get("textAnnotations", [])
    if not annotations:
        return ""
    text: str = annotations[0].get("description", "")
    return text


def extract_page_number(
    response: dict, image_width: int, image_height: int
) -> int | None:
    """ページ番号を抽出する。

    Google Cloud VisionのレスポンスにはYomitokuのようなpage_header/footerの
    role分類が無く、現時点ではページ番号抽出に未対応のため常にNoneを返す。
    yomitoku.extract_page_numberと同じシグネチャに揃えており、対応する場合は
    ここに実装を追加する。
    """
    return None
