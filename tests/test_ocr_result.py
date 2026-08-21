from yomitoku.document_analyzer import (
    DocumentAnalyzerSchema,
    FigureSchema,
    ParagraphSchema,
)

from ocr_app.ocr_result import extract_page_number, extract_recognized_text


def _paragraph(
    contents: str | None,
    order: int | None,
    role: str | None = None,
    box: list[int] | None = None,
) -> ParagraphSchema:
    return ParagraphSchema(
        box=box or [0, 0, 0, 0],
        contents=contents,
        direction=None,
        order=order,
        role=role,
    )


def _document(
    paragraphs: list[ParagraphSchema], figures: list[FigureSchema]
) -> DocumentAnalyzerSchema:
    return DocumentAnalyzerSchema(
        paragraphs=paragraphs, tables=[], words=[], figures=figures
    )


def test_extract_recognized_text_joins_paragraphs_in_order() -> None:
    document = _document(
        paragraphs=[
            _paragraph("second", order=1),
            _paragraph("first", order=0),
        ],
        figures=[],
    )

    assert extract_recognized_text(document) == "first\nsecond"


def test_extract_recognized_text_skips_paragraphs_with_no_contents() -> None:
    document = _document(
        paragraphs=[
            _paragraph("first", order=0),
            _paragraph(None, order=1),
            _paragraph("second", order=2),
        ],
        figures=[],
    )

    assert extract_recognized_text(document) == "first\nsecond"


def test_extract_recognized_text_includes_paragraphs_inside_figures() -> None:
    document = _document(
        paragraphs=[_paragraph("outside", order=0)],
        figures=[
            FigureSchema(
                box=[0, 0, 0, 0],
                order=None,
                direction=None,
                paragraphs=[_paragraph("inside", order=1)],
            )
        ],
    )

    assert extract_recognized_text(document) == "outside\ninside"


def test_extract_recognized_text_keeps_original_order_when_sort_is_false() -> None:
    document = _document(
        paragraphs=[
            _paragraph("second", order=1),
            _paragraph("first", order=0),
        ],
        figures=[],
    )

    assert extract_recognized_text(document, sort=False) == "second\nfirst"


def test_extract_recognized_text_excludes_non_body_roles() -> None:
    document = _document(
        paragraphs=[
            _paragraph("header", order=0, role="page_header"),
            _paragraph("heading", order=1, role="section_headings"),
            _paragraph("body", order=2, role=None),
            _paragraph("footer", order=3, role="page_footer"),
        ],
        figures=[],
    )

    assert extract_recognized_text(document) == "body"


def test_extract_page_number_finds_leading_digits_in_footer() -> None:
    document = _document(
        paragraphs=[
            _paragraph("body", order=0, role=None),
            _paragraph("12 chapter title", order=1, role="page_footer"),
        ],
        figures=[],
    )

    assert extract_page_number(document, image_width=100, image_height=100) == 12


def test_extract_page_number_finds_trailing_digits_in_header() -> None:
    document = _document(
        paragraphs=[
            _paragraph("chapter title 34", order=0, role="page_header"),
        ],
        figures=[],
    )

    assert extract_page_number(document, image_width=100, image_height=100) == 34


def test_extract_page_number_returns_none_when_no_header_or_footer() -> None:
    document = _document(
        paragraphs=[_paragraph("body", order=0, role=None)],
        figures=[],
    )

    assert extract_page_number(document, image_width=100, image_height=100) is None


def test_extract_page_number_returns_none_when_no_digits_found() -> None:
    document = _document(
        paragraphs=[_paragraph("no digits here", order=0, role="page_footer")],
        figures=[],
    )

    assert extract_page_number(document, image_width=100, image_height=100) is None


def test_extract_page_number_prefers_candidate_farthest_from_image_center() -> None:
    document = _document(
        paragraphs=[
            # box中心(495,495)は画像中心(500,500)にほぼ近い
            _paragraph("1", order=0, role="page_footer", box=[490, 490, 500, 500]),
            # box中心(10,10)は画像中心から遠い(外側の角に近い)
            _paragraph("2", order=1, role="page_footer", box=[0, 0, 20, 20]),
        ],
        figures=[],
    )

    assert extract_page_number(document, image_width=1000, image_height=1000) == 2


def test_extract_page_number_compares_header_and_footer_candidates_together() -> None:
    document = _document(
        paragraphs=[
            # box中心(500,500)は画像中心と一致(内側)
            _paragraph("1", order=0, role="page_header", box=[490, 490, 510, 510]),
            # box中心(10,10)は画像の角に近い(外側)
            _paragraph("2", order=1, role="page_footer", box=[0, 0, 20, 20]),
        ],
        figures=[],
    )

    assert extract_page_number(document, image_width=1000, image_height=1000) == 2
