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


def _annotation(text: str, x1: int, y1: int, x2: int, y2: int) -> dict:
    return {
        "description": text,
        "boundingPoly": {
            "vertices": [
                {"x": x1, "y": y1},
                {"x": x2, "y": y1},
                {"x": x2, "y": y2},
                {"x": x1, "y": y2},
            ]
        },
    }


def test_extract_page_number_returns_none_when_no_responses() -> None:
    assert extract_page_number({"responses": []}, 100, 100) is None


def test_extract_page_number_returns_none_when_no_digit_annotations() -> None:
    response = {
        "responses": [
            {
                "textAnnotations": [
                    _annotation("本文", 0, 0, 100, 100),
                    _annotation("本文", 10, 10, 50, 50),
                ]
            }
        ]
    }

    assert extract_page_number(response, 100, 100) is None


def test_extract_page_number_finds_digit_annotation() -> None:
    response = {
        "responses": [
            {
                "textAnnotations": [
                    _annotation("本文 34", 0, 0, 100, 100),
                    _annotation("本文", 10, 10, 50, 50),
                    _annotation("34", 90, 90, 100, 100),
                ]
            }
        ]
    }

    assert extract_page_number(response, 1000, 1000) == 34


def test_extract_page_number_ignores_index_zero_full_text_annotation() -> None:
    # index 0はページ全体の認識結果を表す集約要素であり、たまたま数字のみ
    # (例: 見出しなどが無いページでページ番号だけが写った場合)でも、
    # 個別の候補としては扱わない。
    response = {
        "responses": [
            {
                "textAnnotations": [
                    _annotation("383", 0, 0, 1000, 1000),
                ]
            }
        ]
    }

    assert extract_page_number(response, 1000, 1000) is None


def test_extract_page_number_prefers_candidate_farthest_from_image_center() -> None:
    response = {
        "responses": [
            {
                "textAnnotations": [
                    _annotation("本文 1 2", 0, 0, 1000, 1000),
                    # 中心(500,500)に近い
                    _annotation("1", 490, 490, 510, 510),
                    # 画像の角に近い(外側)
                    _annotation("2", 0, 0, 20, 20),
                ]
            }
        ]
    }

    assert extract_page_number(response, 1000, 1000) == 2


def test_extract_page_number_ignores_annotation_without_bounding_poly() -> None:
    response = {
        "responses": [
            {
                "textAnnotations": [
                    {"description": "本文 34"},
                    {"description": "34"},
                ]
            }
        ]
    }

    assert extract_page_number(response, 1000, 1000) is None


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
