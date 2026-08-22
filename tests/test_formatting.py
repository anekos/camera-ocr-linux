from ocr_app.formatting import format_selection_as_quote


def test_format_selection_as_quote_wraps_in_backticks() -> None:
    assert format_selection_as_quote("こんにちは", page_number=None) == "`こんにちは`"


def test_format_selection_as_quote_joins_line_breaks() -> None:
    assert (
        format_selection_as_quote("こん\nにち\nは", page_number=None) == "`こんにちは`"
    )


def test_format_selection_as_quote_appends_page_number_when_present() -> None:
    assert (
        format_selection_as_quote("こんにちは", page_number=383) == "`こんにちは` P.383"
    )


def test_format_selection_as_quote_omits_page_number_when_none() -> None:
    assert format_selection_as_quote("こんにちは", page_number=None) == "`こんにちは`"
