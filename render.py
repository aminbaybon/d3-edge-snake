from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from snake import Direction, SnakeGame

BG = (8, 14, 22)
FIELD = (12, 28, 22)
GRID_A = (18, 48, 36)
GRID_B = (14, 38, 30)
HEAD = (255, 221, 90)
BODY = (62, 214, 132)
BODY_TAIL = (28, 120, 78)
FOOD = (255, 72, 78)
FOOD_GLOW = (255, 140, 90)
LEAF = (90, 210, 120)
INK = (236, 242, 248)
MUTED = (150, 168, 184)
HUD = (16, 22, 34)


FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\tahoma.ttf"),
    Path(r"C:\Windows\Fonts\Tahoma.ttf"),
    Path(r"C:\Windows\Fonts\segoeui.ttf"),
    Path(r"C:\Windows\Fonts\arial.ttf"),
]


@lru_cache(maxsize=1)
def _font_path() -> str | None:
    for path in FONT_CANDIDATES:
        if path.is_file():
            return str(path)
    return None


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = _font_path()
    if path:
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def shape_fa(text: str) -> str:
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
    except ImportError:
        try:
            import arabic_reshaper
            from bidi import get_display
        except ImportError:
            return text
    return get_display(arabic_reshaper.reshape(text))


def _text(draw: ImageDraw.ImageDraw, xy, text: str, fill=INK, size: int = 18, anchor: str = "lt") -> None:
    draw.text(xy, shape_fa(text), font=_font(size), fill=fill, anchor=anchor)


def _cell_center(ox: int, oy: int, cell: int, pos: tuple[int, int]) -> tuple[float, float]:
    x, y = pos
    return ox + x * cell + cell / 2, oy + y * cell + cell / 2


def _lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def render_board(game: SnakeGame, last_move: str = "", last_p: float | None = None) -> Image.Image:
    cell = 56
    hud = 64
    margin = 28
    board = game.size * cell
    width = margin * 2 + board
    height = hud + margin + board + 36
    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle([12, 12, width - 12, hud - 4], radius=16, fill=HUD)
    _text(draw, (28, hud / 2), "Snake", fill=HEAD, size=22, anchor="lm")
    chips = [
        f"score {game.score}",
        f"step {game.steps}",
        game.heading.value,
    ]
    if last_move:
        conf = int(round((last_p or 0) * 100))
        chips.append(f"{last_move} {conf}%")
    x = width - 28
    for chip in reversed(chips):
        _text(draw, (x, hud / 2), chip, fill=INK, size=16, anchor="rm")
        x -= 108

    ox, oy = margin, hud + 8
    draw.rounded_rectangle([ox - 8, oy - 8, ox + board + 8, oy + board + 8], radius=22, fill=FIELD)
    for y in range(game.size):
        for x in range(game.size):
            x0, y0 = ox + x * cell, oy + y * cell
            color = GRID_A if (x + y) % 2 == 0 else GRID_B
            draw.rounded_rectangle([x0 + 3, y0 + 3, x0 + cell - 4, y0 + cell - 4], radius=10, fill=color)

    visible = [p for p in game.snake if 0 <= p[0] < game.size and 0 <= p[1] < game.size]
    if len(visible) >= 2:
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        for i in range(len(visible) - 1):
            t = i / max(len(visible) - 1, 1)
            color = _lerp(BODY_TAIL, BODY, t) + (255,)
            a = _cell_center(ox, oy, cell, visible[i])
            b = _cell_center(ox, oy, cell, visible[i + 1])
            width_px = int(cell * 0.62)
            ld.line([a, b], fill=color, width=width_px)
            r = width_px / 2
            for pt in (a, b):
                ld.ellipse([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r], fill=color)
        img = Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")
        draw = ImageDraw.Draw(img)
    elif visible:
        cx, cy = _cell_center(ox, oy, cell, visible[0])
        r = cell * 0.31
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=BODY)

    if visible:
        hx, hy = visible[-1]
        cx, cy = _cell_center(ox, oy, cell, (hx, hy))
        r = cell * 0.34
        draw.ellipse([cx - r - 2, cy - r - 2, cx + r + 2, cy + r + 2], fill=(255, 244, 160))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=HEAD)
        eye = {
            Direction.UP: ((-8, -9), (8, -9)),
            Direction.DOWN: ((-8, 9), (8, 9)),
            Direction.LEFT: ((-10, -7), (-10, 7)),
            Direction.RIGHT: ((10, -7), (10, 7)),
        }[game.heading]
        for ex, ey in eye:
            draw.ellipse([cx + ex - 3.4, cy + ey - 3.4, cx + ex + 3.4, cy + ey + 3.4], fill=(28, 32, 36))
            draw.ellipse([cx + ex - 1.2, cy + ey - 2.4, cx + ex + 1.4, cy + ey - 0.2], fill=(255, 255, 255))

    fx, fy = game.food
    ax, ay = _cell_center(ox, oy, cell, (fx, fy))
    glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse([ax - 22, ay - 22, ax + 22, ay + 22], fill=FOOD_GLOW + (70,))
    glow = glow.filter(ImageFilter.GaussianBlur(6))
    img = Image.alpha_composite(img.convert("RGBA"), glow).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.ellipse([ax - 14, ay - 14, ax + 14, ay + 14], fill=FOOD)
    draw.ellipse([ax - 8, ay - 11, ax - 1, ay - 4], fill=(255, 180, 170))
    draw.polygon([(ax + 2, ay - 14), (ax + 12, ay - 22), (ax + 1, ay - 8)], fill=LEAF)

    if not game.alive:
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rounded_rectangle([ox - 8, oy - 8, ox + board + 8, oy + board + 8], radius=22, fill=(6, 10, 16, 150))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(img)
        msg = "Won" if game.death == "win" else "Game over"
        _text(draw, (ox + board / 2, oy + board / 2 - 8), msg, fill=HEAD if game.death == "win" else (255, 130, 130), size=36, anchor="mm")
        detail = {"wall": "hit a wall", "self": "hit itself", "win": "board filled"}.get(game.death, "")
        if detail:
            _text(draw, (ox + board / 2, oy + board / 2 + 28), detail, fill=MUTED, size=16, anchor="mm")

    return img
