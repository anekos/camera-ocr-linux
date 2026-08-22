def touch_to_image_fraction(
    touch_x: float,
    touch_y: float,
    widget_x: float,
    widget_y: float,
    widget_width: float,
    widget_height: float,
    image_width: float,
    image_height: float,
) -> tuple[float, float] | None:
    """ウィジェット上のタッチ座標を、表示されている画像内の相対位置(0〜1)に変換する。

    画像はウィジェット内に中央揃えで表示されている前提(keep_ratioによる
    レターボックス)。戻り値は(fx, fy)で、fx=0が画像の左端、fy=0が画像の
    上端(フレーム配列のrow 0)に対応する。Kivyの座標系はY軸が上向きだが、
    フレーム配列は上端がrow 0のためfyは反転させる。タッチが画像の外なら
    Noneを返す。
    """
    image_x = widget_x + (widget_width - image_width) / 2
    image_y = widget_y + (widget_height - image_height) / 2

    if not (image_x <= touch_x <= image_x + image_width):
        return None
    if not (image_y <= touch_y <= image_y + image_height):
        return None

    fx = (touch_x - image_x) / image_width
    fy = 1.0 - (touch_y - image_y) / image_height
    return fx, fy


def normalize_box(
    x1: float, y1: float, x2: float, y2: float
) -> tuple[float, float, float, float]:
    """2点から (left, top, right, bottom) の矩形を作り、0〜1の範囲にクランプする。"""
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    return (
        min(1.0, max(0.0, left)),
        min(1.0, max(0.0, top)),
        min(1.0, max(0.0, right)),
        min(1.0, max(0.0, bottom)),
    )
