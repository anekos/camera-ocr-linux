def format_selection_as_quote(text: str, page_number: int | None) -> str:
    """テキストエリアで選択した範囲を引用形式に整形する。

    改行はOCRによる見かけ上の行区切りにすぎないため連結して1行にし、
    バッククォートで囲む。ページ番号があれば末尾に "P.<ページ番号>" を付ける。
    """
    quoted = f"`{text.replace('\n', '')}`"
    if page_number is None:
        return quoted
    return f"{quoted} P.{page_number}"
