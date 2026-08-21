from yomitoku.document_analyzer import DocumentAnalyzerSchema, ParagraphSchema


def extract_recognized_text(analyzed: DocumentAnalyzerSchema) -> str:
    """OCR結果から認識済みテキストを、読み取り順に連結して取り出す。

    段落は文書直下のものと図(figure)内のものを合わせて order 順に並べる。
    """
    paragraphs: list[ParagraphSchema] = list(analyzed.paragraphs)
    for figure in analyzed.figures:
        paragraphs.extend(figure.paragraphs)

    paragraphs.sort(key=lambda paragraph: paragraph.order or 0)

    contents = [
        paragraph.contents for paragraph in paragraphs if paragraph.contents is not None
    ]
    return "\n".join(contents)
