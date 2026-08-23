import base64
import json

import httpx
import pytest

from ocr_app.ocr.openai import (
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


def test_build_request_body_embeds_model_and_base64_image_data_url() -> None:
    body = build_request_body(b"fake-png-bytes", "gpt-4o")

    expected_data_url = "data:image/png;base64," + base64.b64encode(
        b"fake-png-bytes"
    ).decode("ascii")
    assert body["model"] == "gpt-4o"
    content = body["messages"][0]["content"]
    image_parts = [part for part in content if part["type"] == "image_url"]
    assert image_parts == [
        {"type": "image_url", "image_url": {"url": expected_data_url}}
    ]


def test_build_request_body_requests_structured_json_output() -> None:
    body = build_request_body(b"fake-png-bytes", "gpt-4o")

    response_format = body["response_format"]
    assert response_format["type"] == "json_schema"
    schema = response_format["json_schema"]["schema"]
    assert schema["required"] == ["text", "page_number"]


def _response_with_content(content: dict) -> dict:
    return {"choices": [{"message": {"content": json.dumps(content)}}]}


def test_extract_recognized_text_returns_text_field() -> None:
    response = _response_with_content({"text": "本文テキスト", "page_number": None})

    assert extract_recognized_text(response) == "本文テキスト"


def test_extract_recognized_text_returns_empty_string_when_no_choices() -> None:
    assert extract_recognized_text({"choices": []}) == ""


def test_extract_page_number_returns_page_number_field() -> None:
    response = _response_with_content({"text": "本文", "page_number": 42})

    assert extract_page_number(response) == 42


def test_extract_page_number_returns_none_when_field_is_null() -> None:
    response = _response_with_content({"text": "本文", "page_number": None})

    assert extract_page_number(response) is None


def test_extract_page_number_returns_none_when_no_choices() -> None:
    assert extract_page_number({"choices": []}) is None


def test_analyze_posts_request_and_returns_response_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_post(
        url: str, headers: dict, json: dict, timeout: float
    ) -> httpx.Response:
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return httpx.Response(
            200,
            json=_response_with_content({"text": "", "page_number": None}),
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr("ocr_app.ocr.openai.httpx.post", _fake_post)

    result = analyze(b"fake-png-bytes", "my-api-key", "gpt-4o")

    assert result == _response_with_content({"text": "", "page_number": None})
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["headers"] == {"Authorization": "Bearer my-api-key"}
    assert captured["json"] == build_request_body(b"fake-png-bytes", "gpt-4o")


def test_analyze_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_post(
        url: str, headers: dict, json: dict, timeout: float
    ) -> httpx.Response:
        return httpx.Response(
            401, json={"error": "unauthorized"}, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr("ocr_app.ocr.openai.httpx.post", _fake_post)

    with pytest.raises(httpx.HTTPStatusError):
        analyze(b"fake-png-bytes", "my-api-key", "gpt-4o")
