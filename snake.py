from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum


class Direction(str, Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


DELTA: dict[Direction, tuple[int, int]] = {
    Direction.UP: (0, -1),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
    Direction.RIGHT: (1, 0),
}

OPPOSITE: dict[Direction, Direction] = {
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
}

FA_DIR: dict[Direction, str] = {
    Direction.UP: "بالا",
    Direction.DOWN: "پایین",
    Direction.LEFT: "چپ",
    Direction.RIGHT: "راست",
}

DIR_FROM_TEXT: dict[str, Direction] = {}
for _d in Direction:
    DIR_FROM_TEXT[_d.value] = _d
    DIR_FROM_TEXT[_d.value.upper()] = _d
    DIR_FROM_TEXT[FA_DIR[_d]] = _d
    DIR_FROM_TEXT[FA_DIR[_d].replace("ی", "ي")] = _d


def parse_direction(value: object) -> Direction | None:
    if isinstance(value, Direction):
        return value
    text = str(value).strip().lower()
    folded = (
        text.replace("ی", "ي")
        .replace("ك", "ک")
        .replace("north", "up")
        .replace("south", "down")
        .replace("west", "left")
        .replace("east", "right")
        .replace("بالا", "up")
        .replace("پايين", "down")
        .replace("پایین", "down")
        .replace("چپ", "left")
        .replace("راست", "right")
    )
    try:
        return Direction(folded)
    except ValueError:
        return DIR_FROM_TEXT.get(str(value).strip())


@dataclass
class SnakeGame:
    size: int = 10
    seed: int = 7
    snake: list[tuple[int, int]] = field(default_factory=list)
    food: tuple[int, int] = (0, 0)
    heading: Direction = Direction.RIGHT
    score: int = 0
    steps: int = 0
    alive: bool = True
    death: str = ""
    rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.reset(self.seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = int(seed)
        self.rng = random.Random(self.seed)
        mid = self.size // 2
        self.snake = [(mid - 2, mid), (mid - 1, mid), (mid, mid)]
        self.heading = Direction.RIGHT
        self.score = 0
        self.steps = 0
        self.alive = True
        self.death = ""
        self.food = self._place_food()

    @property
    def head(self) -> tuple[int, int]:
        return self.snake[-1]

    @property
    def length(self) -> int:
        return len(self.snake)

    def legal_moves(self) -> list[Direction]:
        banned = OPPOSITE[self.heading]
        return [d for d in Direction if d is not banned]

    def occupied(self) -> set[tuple[int, int]]:
        return set(self.snake)

    def _place_food(self) -> tuple[int, int]:
        empty = [
            (x, y)
            for y in range(self.size)
            for x in range(self.size)
            if (x, y) not in self.occupied()
        ]
        if not empty:
            return self.head
        return self.rng.choice(empty)

    def next_cell(self, direction: Direction) -> tuple[int, int]:
        dx, dy = DELTA[direction]
        x, y = self.head
        return x + dx, y + dy

    def would_die(self, direction: Direction) -> str:
        x, y = self.next_cell(direction)
        if not (0 <= x < self.size and 0 <= y < self.size):
            return "wall"
        body = self.snake if (x, y) == self.food else self.snake[1:]
        if (x, y) in body:
            return "self"
        return ""

    def relative_food(self) -> tuple[int, int, str]:
        hx, hy = self.head
        fx, fy = self.food
        dx, dy = fx - hx, fy - hy
        parts: list[str] = []
        if dx > 0:
            parts.append(f"{dx} right")
        elif dx < 0:
            parts.append(f"{-dx} left")
        if dy > 0:
            parts.append(f"{dy} down")
        elif dy < 0:
            parts.append(f"{-dy} up")
        if not parts:
            text = "on the head"
        else:
            text = " and ".join(parts)
        return dx, dy, text

    def ascii_board(self) -> str:
        cells = {pos: "·" for pos in [(x, y) for y in range(self.size) for x in range(self.size)]}
        for pos in self.snake[:-1]:
            cells[pos] = "▓"
        cells[self.head] = "●"
        cells[self.food] = "★"
        lines = ["  " + " ".join(str(x) for x in range(self.size))]
        for y in range(self.size):
            row = " ".join(cells[(x, y)] for x in range(self.size))
            lines.append(f"{y} {row}")
        return "\n".join(lines)

    def step(self, direction: Direction | str) -> Direction:
        parsed = parse_direction(direction) or self.heading
        if parsed not in self.legal_moves():
            parsed = self.heading
        self.heading = parsed
        reason = self.would_die(parsed)
        nxt = self.next_cell(parsed)
        self.steps += 1
        if reason:
            self.alive = False
            self.death = reason
            self.snake = [*self.snake[1:], nxt]
            return parsed
        ate = nxt == self.food
        self.snake.append(nxt)
        if ate:
            self.score += 1
            self.food = self._place_food()
            if len(self.occupied()) >= self.size * self.size:
                self.alive = False
                self.death = "win"
        else:
            self.snake.pop(0)
        return parsed

    def greedy_hint(self) -> Direction:
        hx, hy = self.head
        fx, fy = self.food
        ranked: list[tuple[int, int, Direction]] = []
        for move in self.legal_moves():
            nx, ny = self.next_cell(move)
            dist = abs(fx - nx) + abs(fy - ny)
            death = 1 if self.would_die(move) else 0
            ranked.append((death, dist, move))
        ranked.sort(key=lambda item: (item[0], item[1]))
        return ranked[0][2]


def to_fa_digits(value: object) -> str:
    return str(value).translate(str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹"))

