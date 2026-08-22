from ocr_app.selection import (
    normalize_box,
    spread_guide_line_points,
    touch_to_image_fraction,
)


def test_touch_to_image_fraction_top_left_of_image_is_origin() -> None:
    # widget: 100x50、画像は幅いっぱい(100)だが高さは40(上下に5pxずつレターボックス)。
    # 画像の左上(image_x=0, image_y=5+40=45)をタッチ。
    result = touch_to_image_fraction(
        touch_x=0,
        touch_y=45,
        widget_x=0,
        widget_y=0,
        widget_width=100,
        widget_height=50,
        image_width=100,
        image_height=40,
    )

    assert result == (0.0, 0.0)


def test_touch_to_image_fraction_bottom_right_of_image_is_one_one() -> None:
    result = touch_to_image_fraction(
        touch_x=100,
        touch_y=5,
        widget_x=0,
        widget_y=0,
        widget_width=100,
        widget_height=50,
        image_width=100,
        image_height=40,
    )

    assert result == (1.0, 1.0)


def test_touch_to_image_fraction_center_is_half_half() -> None:
    result = touch_to_image_fraction(
        touch_x=50,
        touch_y=25,
        widget_x=0,
        widget_y=0,
        widget_width=100,
        widget_height=50,
        image_width=100,
        image_height=40,
    )

    assert result == (0.5, 0.5)


def test_touch_to_image_fraction_returns_none_outside_letterbox_margin() -> None:
    # y=2 はレターボックス余白(0〜5)の中で、画像の外。
    result = touch_to_image_fraction(
        touch_x=50,
        touch_y=2,
        widget_x=0,
        widget_y=0,
        widget_width=100,
        widget_height=50,
        image_width=100,
        image_height=40,
    )

    assert result is None


def test_normalize_box_orders_reversed_points() -> None:
    result = normalize_box(x1=0.8, y1=0.6, x2=0.2, y2=0.1)

    assert result == (0.2, 0.1, 0.8, 0.6)


def test_normalize_box_clamps_to_unit_range() -> None:
    result = normalize_box(x1=-0.5, y1=-0.5, x2=1.5, y2=1.5)

    assert result == (0.0, 0.0, 1.0, 1.0)


def test_spread_guide_line_points_is_vertical_line_at_widget_horizontal_center() -> (
    None
):
    # widget: 100x50、画像は幅いっぱい(100)だが高さは40(上下に5pxずつレターボックス)。
    # 画像は中央揃えのため、水平中央はレターボックスの有無に関わらずwidgetの中央と一致する。
    result = spread_guide_line_points(
        widget_x=0,
        widget_y=0,
        widget_width=100,
        widget_height=50,
        image_width=100,
        image_height=40,
    )

    assert result == (50, 5, 50, 45)


def test_spread_guide_line_points_offsets_with_widget_position() -> None:
    result = spread_guide_line_points(
        widget_x=10,
        widget_y=20,
        widget_width=100,
        widget_height=50,
        image_width=100,
        image_height=40,
    )

    assert result == (60, 25, 60, 65)
