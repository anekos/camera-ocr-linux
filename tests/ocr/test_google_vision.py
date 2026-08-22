import base64

import httpx
import pytest

from ocr_app.ocr.google_vision import (
    API_KEY_ENV_VAR,
    analyze,
    build_request_body,
    extract_page_number,
    extract_recognized_text,
    get_api_key,
)


def test_get_api_key_returns_none_when_env_var_not_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)

    assert get_api_key() is None


def test_get_api_key_returns_env_var_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_ENV_VAR, "secret-key")

    assert get_api_key() == "secret-key"


def test_build_request_body_base64_encodes_image_and_requests_text_detection() -> None:
    body = build_request_body(b"fake-png-bytes")

    expected_content = base64.b64encode(b"fake-png-bytes").decode("ascii")
    assert body == {
        "requests": [
            {
                "image": {"content": expected_content},
                "features": [{"type": "TEXT_DETECTION"}],
            }
        ]
    }


def test_extract_recognized_text_returns_first_annotation_description() -> None:
    response = {
        "responses": [
            {
                "textAnnotations": [
                    {"description": "全文テキスト"},
                    {"description": "全"},
                ]
            }
        ]
    }

    assert extract_recognized_text(response) == "全文テキスト"


def test_extract_recognized_text_returns_empty_string_when_no_annotations() -> None:
    response: dict = {"responses": [{}]}

    assert extract_recognized_text(response) == ""


def test_extract_page_number_always_returns_none() -> None:
    # GCVのレスポンスにはyomitokuのようなpage_header/footerのroleが無いため、
    # 現時点ではページ番号抽出には未対応(常にNone)。
    assert extract_page_number({"responses": [{}]}, 100, 100) is None


def test_analyze_posts_request_and_returns_response_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_post(
        url: str, params: dict, json: dict, timeout: float
    ) -> httpx.Response:
        captured["url"] = url
        captured["params"] = params
        captured["json"] = json
        return httpx.Response(
            200,
            json={"responses": [{"textAnnotations": []}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr("ocr_app.ocr.google_vision.httpx.post", _fake_post)

    result = analyze(b"fake-png-bytes", "my-api-key")

    assert result == {"responses": [{"textAnnotations": []}]}
    assert captured["url"] == "https://vision.googleapis.com/v1/images:annotate"
    assert captured["params"] == {"key": "my-api-key"}
    assert captured["json"] == build_request_body(b"fake-png-bytes")


def test_analyze_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_post(
        url: str, params: dict, json: dict, timeout: float
    ) -> httpx.Response:
        return httpx.Response(
            403, json={"error": "forbidden"}, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr("ocr_app.ocr.google_vision.httpx.post", _fake_post)

    with pytest.raises(httpx.HTTPStatusError):
        analyze(b"fake-png-bytes", "my-api-key")
