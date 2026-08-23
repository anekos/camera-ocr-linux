import base64
import json
import os

import httpx

API_KEY_ENV_VAR = "OCR_APP_OPEN_AI_API_KEY"
CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
REQUEST_TIMEOUT = 60.0

PROMPT = (
    "この画像に写っている本文を一字一句省略・要約せずそのまま書き起こしてください。"
    "改行や段落構成はできるだけ元の見た目に合わせてください。"
    "ページが2段組・3段組など複数の段に分かれている場合は、"
    "存在するすべての段を読み取ってください。1段目だけを読んで"
    "終わらせず、2段目以降に文字が無いか必ず確認し、あれば続けて"
    "書き起こしてください。"
    "ページヘッダー・フッターの余白にページ番号らしき数字があれば、"
    "本文には含めずpage_numberとして分離してください。"
    "見つからない場合はpage_numberをnullにしてください。"
)

RESPONSE_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "ocr_result",
        "schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "page_number": {"type": ["integer", "null"]},
            },
            "required": ["text", "page_number"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


def get_api_key() -> str | None:
    """環境変数からOpenAI APIキーを取得する。未設定ならNone。"""
    return os.environ.get(API_KEY_ENV_VAR) or None


def build_request_body(image_bytes: bytes, model: str) -> dict:
    """Chat Completions APIに送るリクエストボディを組み立てる。

    画像はdata URL化してmessageに埋め込み、response_formatで本文と
    ページ番号をJSONとして構造化して返すよう指示する。
    """
    data_url = f"data:image/png;base64,{base64.b64encode(image_bytes).decode('ascii')}"
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url, "detail": "high"},
                    },
                ],
            }
        ],
        "response_format": RESPONSE_JSON_SCHEMA,
    }


def analyze(image_bytes: bytes, api_key: str, model: str) -> dict:
    """OpenAIのChat Completions APIを呼び出し、レスポンスJSONを返す。

    HTTPエラー時はhttpx.HTTPStatusErrorを送出する(呼び出し側で処理する)。
    """
    response = httpx.post(
        CHAT_COMPLETIONS_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json=build_request_body(image_bytes, model),
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    result: dict = response.json()
    return result


def _parse_result_content(response: dict) -> dict:
    choices = response.get("choices", [])
    if not choices:
        return {}
    content = choices[0].get("message", {}).get("content", "")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def extract_recognized_text(response: dict) -> str:
    """レスポンスから本文テキストを取り出す。"""
    text = _parse_result_content(response).get("text", "")
    return text if isinstance(text, str) else ""


def extract_page_number(response: dict) -> int | None:
    """レスポンスからページ番号を取り出す。"""
    page_number = _parse_result_content(response).get("page_number")
    return page_number if isinstance(page_number, int) else None
