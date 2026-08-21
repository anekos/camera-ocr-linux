import json
from pathlib import Path


def load_settings(path: Path) -> dict[str, bool]:
    """設定ファイルを読み込む。存在しないか壊れている場合は空の辞書を返す。"""
    if not path.exists():
        return {}
    try:
        return dict(json.loads(path.read_text()))
    except json.JSONDecodeError:
        return {}


def save_settings(path: Path, settings: dict[str, bool]) -> None:
    """設定を保存する。親ディレクトリが無ければ作成する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings))
