"""Action-space policies for Snake.

The rendered picture is a deterministic function of the grid. The action space
is the three headings that are not a 180-degree reversal. d3-edge samples that
space from pixels. The policies here read the grid itself.
"""

from __future__ import annotations

from collections import deque

from snake import DELTA, Direction, SnakeGame

CLOCKWISE: tuple[Direction, ...] = (
    Direction.UP,
    Direction.RIGHT,
    Direction.DOWN,
    Direction.LEFT,
)


def relative_to_absolute(heading: Direction, action: int) -> Direction:
    """Map a relative action to a heading. 0 left, 1 straight, 2 right."""
    index = CLOCKWISE.index(heading)
    if action == 0:
        return CLOCKWISE[(index - 1) % 4]
    if action == 2:
        return CLOCKWISE[(index + 1) % 4]
    return heading


def _in_bounds(cell: tuple[int, int], size: int) -> bool:
    x, y = cell
    return 0 <= x < size and 0 <= y < size


def _step_cell(cell: tuple[int, int], direction: Direction) -> tuple[int, int]:
    dx, dy = DELTA[direction]
    return cell[0] + dx, cell[1] + dy


def _bfs(
    start: tuple[int, int],
    goal: tuple[int, int],
    blocked: set[tuple[int, int]],
    size: int,
) -> list[tuple[int, int]] | None:
    """Shortest path of cells after `start`, or None. `goal` may sit in `blocked`."""
    if start == goal:
        return []
    blocked = set(blocked)
    blocked.discard(goal)
    blocked.add(start)
    queue: deque[tuple[int, int]] = deque([start])
    previous: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while queue:
        current = queue.popleft()
        for direction in Direction:
            nxt = _step_cell(current, direction)
            if nxt in previous or nxt in blocked or not _in_bounds(nxt, size):
                continue
            previous[nxt] = current
            if nxt == goal:
                path = [nxt]
                cursor = current
                while cursor != start:
                    path.append(cursor)
                    parent = previous[cursor]
                    if parent is None:
                        break
                    cursor = parent
                path.reverse()
                return path
            queue.append(nxt)
    return None


def _simulate(snake: list[tuple[int, int]], food: tuple[int, int], path: list[tuple[int, int]], size: int) -> list[tuple[int, int]] | None:
    """Follow `path` as a snake. Eating happens only on the last cell, if it is the food."""
    body = list(snake)
    last = len(path) - 1
    for index, cell in enumerate(path):
        if not _in_bounds(cell, size):
            return None
        eating = index == last and cell == food
        tail = body[0]
        if cell in body and not (cell == tail and not eating):
            return None
        body.append(cell)
        if not eating:
            body.pop(0)
    return body


def _reaches_tail(snake: list[tuple[int, int]], size: int) -> bool:
    if len(snake) <= 2:
        return True
    head = snake[-1]
    tail = snake[0]
    blocked = set(snake) - {head, tail}
    return _bfs(head, tail, blocked, size) is not None


def _flood(snake: list[tuple[int, int]], size: int) -> int:
    head = snake[-1]
    blocked = set(snake)
    seen = {head}
    queue: deque[tuple[int, int]] = deque([head])
    count = 0
    while queue:
        current = queue.popleft()
        for direction in Direction:
            nxt = _step_cell(current, direction)
            if nxt in seen or nxt in blocked or not _in_bounds(nxt, size):
                continue
            seen.add(nxt)
            queue.append(nxt)
            count += 1
    return count


def _direction_of(origin: tuple[int, int], nxt: tuple[int, int]) -> Direction:
    dx, dy = nxt[0] - origin[0], nxt[1] - origin[1]
    for direction, (sx, sy) in DELTA.items():
        if (sx, sy) == (dx, dy):
            return direction
    return Direction.RIGHT


def _cycle(size: int) -> list[tuple[int, int]] | None:
    """Hamiltonian cycle. A grid has one only when a side is even."""
    if size < 2 or size % 2 == 1:
        return None
    path: list[tuple[int, int]] = []
    for y in range(size):
        columns = range(1, size) if y % 2 == 0 else range(size - 1, 0, -1)
        for x in columns:
            path.append((x, y))
    for y in range(size - 1, -1, -1):
        path.append((0, y))
    return path


_CYCLE_NEXT: dict[int, dict[tuple[int, int], tuple[int, int]]] = {}


def _cycle_successor(cell: tuple[int, int], size: int) -> tuple[int, int] | None:
    cached = _CYCLE_NEXT.get(size)
    if cached is None:
        path = _cycle(size)
        if path is None:
            return None
        cached = {path[i]: path[(i + 1) % len(path)] for i in range(len(path))}
        _CYCLE_NEXT[size] = cached
    return cached.get(cell)


def plan_move(game: SnakeGame) -> tuple[Direction, dict[str, float]]:
    """Shortest safe path to the food; otherwise walk a Hamiltonian cycle.

    A food path is accepted only when the snake can follow every cell and, after
    eating, its head can still reach its tail. On an even board the fallback is
    the next cell of a fixed cycle, and only if that step still reaches the tail.
    """
    snake = list(game.snake)
    food = game.food
    size = game.size
    head = snake[-1]
    tail = snake[0]
    legal = set(game.legal_moves())

    body_walls = set(snake)
    body_walls.discard(tail)
    food_path = _bfs(head, food, body_walls, size)
    if food_path:
        followed = _simulate(snake, food, food_path, size)
        first = _direction_of(head, food_path[0])
        if followed is not None and first in legal and _reaches_tail(followed, size):
            return first, _scores(game, first)

    # On an even board the leftover cells form a cycle. Walking it cannot trap
    # the snake the way a shortest-path coil can, so it is the fallback once the
    # apple is no longer safely reachable.
    successor = _cycle_successor(head, size)
    if successor is not None:
        cycle_move = _direction_of(head, successor)
        if cycle_move in legal and not game.would_die(cycle_move):
            nxt = game.next_cell(cycle_move)
            eating = nxt == food
            followed = [*snake, nxt] if eating else [*snake[1:], nxt]
            if _reaches_tail(followed, size):
                return cycle_move, _scores(game, cycle_move)

    tail_walls = set(snake) - {head, tail}
    tail_path = _bfs(head, tail, tail_walls, size)
    if tail_path:
        first = _direction_of(head, tail_path[0])
        if first in legal and not game.would_die(first):
            return first, _scores(game, first)

    ranked: list[tuple[int, int, Direction]] = []
    for move in game.legal_moves():
        if game.would_die(move):
            ranked.append((-1, 0, move))
            continue
        nxt = game.next_cell(move)
        eating = nxt == food
        body = [*snake, nxt] if eating else [*snake[1:], nxt]
        ranked.append((_flood(body, size), 0 if eating else 1, move))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    chosen = ranked[0][2]
    return chosen, _scores(game, chosen)


def _scores(game: SnakeGame, chosen: Direction) -> dict[str, float]:
    """UI masses. The chosen action is certain; the others stay visible at zero."""
    scores = {direction.value: 0.0 for direction in Direction}
    scores[chosen.value] = 1.0
    return scores


def greedy_move(game: SnakeGame) -> Direction:
    return game.greedy_hint()
