import re

from yomitoku.document_analyzer import DocumentAnalyzerSchema, ParagraphSchema

PAGE_NUMBER_ROLES = ("page_header", "page_footer")


def _collect_paragraphs(analyzed: DocumentAnalyzerSchema) -> list[ParagraphSchema]:
    paragraphs: list[ParagraphSchema] = list(analyzed.paragraphs)
    for figure in analyzed.figures:
        paragraphs.extend(figure.paragraphs)
    paragraphs.sort(key=lambda paragraph: paragraph.order or 0)
    return paragraphs


def extract_recognized_text(analyzed: DocumentAnalyzerSchema) -> str:
    """OCR結果から本文の認識済みテキストを、読み取り順に連結して取り出す。

    段落は文書直下のものと図(figure)内のものを合わせて order 順に並べる。
    見出し(section_headings)やページヘッダー・フッターなど、role が
    None でない段落は本文ではないため除外する。
    """
    contents = [
        paragraph.contents
        for paragraph in _collect_paragraphs(analyzed)
        if paragraph.role is None and paragraph.contents is not None
    ]
    return "\n".join(contents)


def _parse_page_number(text: str) -> int | None:
    """`123 subtitle` や `subtitle 123` のような文字列から先頭or末尾の数字列を取り出す。"""
    text = text.strip()
    if (m := re.match(r"^(\d+).*", text)) is not None:
        return int(m.group(1))
    if (m := re.match(r"^.+?(\d+)$", text)) is not None:
        return int(m.group(1))
    return None


def extract_page_number(analyzed: DocumentAnalyzerSchema) -> int | None:
    """ページヘッダー・フッターの段落からページ番号らしき数値を探す。

    order順に見て、最初に数字列が見つかった段落の値を返す。
    """
    for paragraph in _collect_paragraphs(analyzed):
        if paragraph.role not in PAGE_NUMBER_ROLES or paragraph.contents is None:
            continue
        if (page_number := _parse_page_number(paragraph.contents)) is not None:
            return page_number
    return None
