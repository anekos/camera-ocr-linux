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


def _distance_from_image_center(
    box: list[int], image_width: int, image_height: int
) -> float:
    left, top, right, bottom = box
    box_center_x = (left + right) / 2
    box_center_y = (top + bottom) / 2
    return (
        (box_center_x - image_width / 2) ** 2 + (box_center_y - image_height / 2) ** 2
    ) ** 0.5


def extract_page_number(
    analyzed: DocumentAnalyzerSchema, image_width: int, image_height: int
) -> int | None:
    """ページヘッダー・フッターの段落からページ番号らしき数値を探す。

    ページ番号は綴じ側(中央寄り)より余白側(外側)に印字されることが多いため、
    複数の候補があれば画像中心からの距離が最大のもの、つまり最も外側にある
    ものを採用する。
    """
    candidates: list[tuple[float, int]] = []
    for paragraph in _collect_paragraphs(analyzed):
        if paragraph.role not in PAGE_NUMBER_ROLES or paragraph.contents is None:
            continue
        page_number = _parse_page_number(paragraph.contents)
        if page_number is None:
            continue
        distance = _distance_from_image_center(paragraph.box, image_width, image_height)
        candidates.append((distance, page_number))

    if not candidates:
        return None

    _distance, page_number = max(candidates, key=lambda candidate: candidate[0])
    return page_number
