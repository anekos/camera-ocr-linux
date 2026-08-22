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


def _annotation_center(annotation: dict) -> tuple[float, float] | None:
    vertices = annotation.get("boundingPoly", {}).get("vertices", [])
    if not vertices:
        return None
    xs = [v.get("x", 0) for v in vertices]
    ys = [v.get("y", 0) for v in vertices]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _distance_from_image_center(
    center: tuple[float, float], image_width: int, image_height: int
) -> float:
    x, y = center
    return ((x - image_width / 2) ** 2 + (y - image_height / 2) ** 2) ** 0.5


def extract_page_number(
    response: dict, image_width: int, image_height: int
) -> int | None:
    """ページ番号を抽出する。

    Google Cloud VisionのレスポンスにはYomitokuのようなpage_header/footerの
    role分類が無いため、数字のみのtextAnnotationを候補とし、画像中心から
    最も離れている(=余白側にある可能性が高い)ものを採用する。index 0は
    ページ全体の認識結果を表す集約要素のため候補から除外する。
    """
    responses = response.get("responses", [])
    if not responses:
        return None

    candidates: list[tuple[float, int]] = []
    for annotation in responses[0].get("textAnnotations", [])[1:]:
        text = annotation.get("description", "").strip()
        if not text.isdigit():
            continue
        center = _annotation_center(annotation)
        if center is None:
            continue
        distance = _distance_from_image_center(center, image_width, image_height)
        candidates.append((distance, int(text)))

    if not candidates:
        return None

    _distance, page_number = max(candidates, key=lambda candidate: candidate[0])
    return page_number
