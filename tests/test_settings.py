from pathlib import Path

from ocr_app.settings import (
    get_bool,
    get_float,
    load_settings,
    resolve_save_directory,
    save_settings,
)


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


def test_resolve_save_directory_returns_default_when_not_in_settings(
    tmp_path: Path,
) -> None:
    default = tmp_path / "default-dir"

    assert resolve_save_directory({}, default) == default


def test_resolve_save_directory_returns_configured_path_when_present(
    tmp_path: Path,
) -> None:
    default = tmp_path / "default-dir"
    configured = tmp_path / "configured-dir"

    result = resolve_save_directory({"save_directory": str(configured)}, default)

    assert result == configured


def test_get_bool_returns_default_when_key_missing() -> None:
    assert get_bool({}, "flip", default=False) is False


def test_get_bool_returns_stored_value_when_present() -> None:
    assert get_bool({"flip": True}, "flip", default=False) is True


def test_get_float_returns_default_when_key_missing() -> None:
    assert get_float({}, "result_height", default=150.0) == 150.0


def test_get_float_returns_stored_value_when_present() -> None:
    assert get_float({"result_height": 220.0}, "result_height", default=150.0) == 220.0


def test_get_float_returns_default_when_type_is_wrong() -> None:
    assert get_float({"result_height": True}, "result_height", default=150.0) == 150.0
