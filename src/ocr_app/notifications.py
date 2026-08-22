import logging
import subprocess

logger = logging.getLogger(__name__)


def send_notification(title: str, message: str) -> None:
    """notify-send経由でデスクトップ通知を送る。notify-sendが無い/失敗しても例外は送出しない。"""
    try:
        subprocess.run(["notify-send", title, message], check=False)
    except FileNotFoundError:
        logger.warning("notify-send not found; skipping desktop notification")
