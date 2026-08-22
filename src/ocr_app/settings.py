import json
from pathlib import Path

Settings = dict[str, bool | str | float]


def load_settings(path: Path) -> Settings:
    """設定ファイルを読み込む。存在しないか壊れている場合は空の辞書を返す。"""
    if not path.exists():
        return {}
    try:
        return dict(json.loads(path.read_text()))
    except json.JSONDecodeError:
        return {}


def save_settings(path: Path, settings: Settings) -> None:
    """設定を保存する。親ディレクトリが無ければ作成する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings))


def get_bool(settings: Settings, key: str, default: bool) -> bool:
    """設定からbool値を取り出す。無いか型が違えばdefaultを返す。"""
    value = settings.get(key, default)
    return value if isinstance(value, bool) else default


def get_float(settings: Settings, key: str, default: float) -> float:
    """設定からfloat値を取り出す。無いか型が違えばdefaultを返す。"""
    value = settings.get(key, default)
    return (
        value
        if isinstance(value, float | int) and not isinstance(value, bool)
        else default
    )


def resolve_save_directory(settings: Settings, default: Path) -> Path:
    """設定に保存先(save_directory)が指定されていればそれを、無ければdefaultを返す。"""
    configured = settings.get("save_directory")
    if isinstance(configured, str):
        return Path(configured)
    return default
