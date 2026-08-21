from pathlib import Path

from ocr_app.settings import load_settings, save_settings


def test_load_settings_returns_empty_dict_when_file_does_not_exist(
    tmp_path: Path,
) -> None:
    assert load_settings(tmp_path / "settings.json") == {}


def test_save_settings_then_load_settings_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "settings.json"

    save_settings(path, {"copy_to_clipboard": True, "flip": False})

    assert load_settings(path) == {"copy_to_clipboard": True, "flip": False}


def test_load_settings_returns_empty_dict_for_corrupted_file(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text("not valid json")

    assert load_settings(path) == {}
