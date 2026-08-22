import subprocess

import pytest

from ocr_app.notifications import send_notification


def test_send_notification_calls_notify_send_with_title_and_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        "ocr_app.notifications.subprocess.run",
        lambda args, **kwargs: calls.append(args),
    )

    send_notification("タイトル", "本文")

    assert calls == [["notify-send", "タイトル", "本文"]]


def test_send_notification_does_not_raise_when_notify_send_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_not_found(args: list[str], **kwargs: object) -> None:
        raise FileNotFoundError("notify-send")

    monkeypatch.setattr("ocr_app.notifications.subprocess.run", _raise_not_found)

    send_notification("タイトル", "本文")  # 例外を送出しないことを確認


def test_send_notification_does_not_raise_when_notify_send_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(args, returncode=1)

    monkeypatch.setattr("ocr_app.notifications.subprocess.run", _fail)

    send_notification("タイトル", "本文")  # 例外を送出しないことを確認
