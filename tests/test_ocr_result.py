from yomitoku.document_analyzer import (
    DocumentAnalyzerSchema,
    FigureSchema,
    ParagraphSchema,
)

from ocr_app.ocr_result import extract_recognized_text


def _paragraph(contents: str | None, order: int | None) -> ParagraphSchema:
    return ParagraphSchema(
        box=[0, 0, 0, 0], contents=contents, direction=None, order=order, role=None
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
