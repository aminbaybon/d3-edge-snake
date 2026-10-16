"""Grid observation and Double DQN policy for Snake.

Action 0 turns left, 1 goes straight, 2 turns right. That is the whole legal
action space: a reversal is not representable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from policy import _reaches_tail, plan_move, relative_to_absolute
from snake import Direction, SnakeGame

WEIGHTS = Path(__file__).resolve().parent / "rl_snake.pt"

# Mean probability d3-edge assigned on the recorded 8×8 episode (seed 7).
# Left never appeared in the distribution, so its mass is a numerical floor.
D3_PRIOR: dict[str, float] = {
    "up": 0.2871621809690305,
    "down": 0.20916539548819614,
    "left": 1e-6,
    "right": 0.5036724235427733,
}


def normalized_prior() -> dict[str, float]:
    total = sum(D3_PRIOR.values())
    return {key: value / total for key, value in D3_PRIOR.items()}


def _heading_frame(heading: Direction, dx: int, dy: int) -> tuple[int, int]:
    """Return (forward, to-the-right) in the snake's own frame."""
    if heading is Direction.RIGHT:
        return dx, dy
    if heading is Direction.DOWN:
        return dy, -dx
    if heading is Direction.LEFT:
        return -dx, -dy
    return -dy, dx


def observe(game: SnakeGame) -> np.ndarray:
    """Features aligned with the three relative actions, not a picture of the board.

    A flattened grid left the same MLP near a score of 2 after 75k updates. These
    features say, for the action about to be taken, whether it dies and whether
    the apple is in front or to that side.
    """
    heading = game.heading
    dangers = [
        1.0 if game.would_die(relative_to_absolute(heading, action)) else 0.0
        for action in (0, 1, 2)
    ]
    hx, hy = game.head
    fx, fy = game.food
    dx, dy = fx - hx, fy - hy
    forward, side = _heading_frame(heading, dx, dy)
    direction = [1.0 if heading is item else 0.0 for item in Direction]
    food_sign = [
        1.0 if dx < 0 else 0.0,
        1.0 if dx > 0 else 0.0,
        1.0 if dy < 0 else 0.0,
        1.0 if dy > 0 else 0.0,
    ]
    scale = float(game.size)
    return np.array(
        [*dangers, *direction, *food_sign, forward / scale, side / scale, game.length / (scale * scale)],
        dtype=np.float32,
    )


def observation_size(board: int) -> int:
    del board
    return 3 + 4 + 4 + 2 + 1


class QNet(nn.Module):
    def __init__(self, board: int, hidden: int = 128) -> None:
        super().__init__()
        width = observation_size(board)
        self.body = nn.Sequential(
            nn.Linear(width, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.body(obs)


def load_qnet(path: Path | None = None, device: str | None = None) -> tuple[QNet, int, torch.device]:
    ckpt_path = path or WEIGHTS
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    checkpoint = torch.load(ckpt_path, map_location=torch_device, weights_only=False)
    board = int(checkpoint["board"])
    net = QNet(board, int(checkpoint.get("hidden", 128)))
    net.load_state_dict(checkpoint["state_dict"])
    net.to(torch_device)
    net.eval()
    return net, board, torch_device


def q_values(net: QNet, game: SnakeGame, device: torch.device) -> np.ndarray:
    obs = torch.from_numpy(observe(game)).to(device).unsqueeze(0)
    with torch.inference_mode():
        return net(obs).squeeze(0).detach().cpu().numpy()


# The learned value must beat the model's safe proposal by this much before
# the network replaces it. A smaller gap keeps the downloaded model's move.
OVERRIDE_MARGIN = 0.5


def edit_model_move(
    net: QNet,
    game: SnakeGame,
    device: torch.device,
    model_probs: dict[str, float],
) -> tuple[Direction, str]:
    """Keep the model's proposal, and replace it when play has shown it is worse.

    The 0.8B weights stay frozen. This network was trained by playing, and it
    only edits the live decision: a move that dies is refused, and a safe move
    is replaced when another safe move is clearly more valuable.
    """
    ranked = sorted(model_probs.items(), key=lambda item: item[1], reverse=True)
    model_name = ranked[0][0] if ranked else ""
    model_move = Direction(model_name) if model_name in {item.value for item in Direction} else None
    safe = [direction for direction in game.legal_moves() if not game.would_die(direction)]
    pool = safe or list(game.legal_moves())
    values = q_values(net, game, device)
    q_of = {
        relative_to_absolute(game.heading, action): float(values[action])
        for action in range(3)
    }
    learned = max(pool, key=lambda direction: q_of[direction])
    mentioned = [direction for direction in pool if model_probs.get(direction.value, 0.0) > 0.0]
    proposal = max(mentioned, key=lambda direction: model_probs[direction.value]) if mentioned else None
    fatal = model_move is not None and bool(game.would_die(model_move))

    if proposal is None:
        return learned, "The model named no safe move, so the learned network chose."
    if fatal:
        chosen = learned if q_of[learned] > q_of[proposal] + OVERRIDE_MARGIN else proposal
        return chosen, f"The model said {model_move.value}, which would die. Played {chosen.value} instead."
    if learned is not proposal and q_of[learned] > q_of[proposal] + OVERRIDE_MARGIN:
        return learned, f"The model said {proposal.value}. The learned network changed it to {learned.value}."
    return proposal, "Played the model's move."


class LoopWatch:
    """Hand the learned network the controls when the snake is circling.

    The model keeps playing while the path is new. A repeated route, a head
    returning to a cell it already left since the last apple, or a second
    fatal proposal means the model is stuck. The network then steers until
    the snake eats, and the model resumes.
    """

    def __init__(self) -> None:
        self.heads: list[tuple[int, int]] = []
        self.moves: list[str] = []
        self.driving = False
        self.planner = False
        self.score = 0
        self.fatal_proposals = 0

    def looping(self, game: SnakeGame) -> bool:
        if self.driving:
            return True
        if len(self.moves) >= 8:
            here = game.head
            prior = self.heads[:-1]
            if here in prior[-16:]:
                self.driving = True
                return True
        for period in range(2, 9):
            if len(self.moves) >= period * 2 and self.moves[-period:] == self.moves[-2 * period : -period]:
                self.driving = True
                return True
        return False

    def note_fatal(self) -> None:
        self.fatal_proposals += 1
        if self.fatal_proposals >= 2:
            self.driving = True

    def record(self, game: SnakeGame, move: Direction) -> None:
        self.heads.append(game.head)
        self.moves.append(move.value)
        if game.score != self.score:
            self.score = game.score
            self.heads.clear()
            self.moves.clear()
            self.driving = False
            self.planner = False
            self.fatal_proposals = 0


def _keeps_tail(game: SnakeGame, move: Direction) -> bool:
    if game.would_die(move):
        return False
    nxt = game.next_cell(move)
    snake = list(game.snake)
    followed = [*snake, nxt] if nxt == game.food else [*snake[1:], nxt]
    return _reaches_tail(followed, game.size)


def _food_distance(cell: tuple[int, int], food: tuple[int, int]) -> int:
    return abs(cell[0] - food[0]) + abs(cell[1] - food[1])


def rescue_move(net: QNet, game: SnakeGame, device: torch.device) -> tuple[Direction, dict[str, float], str]:
    """Steer toward the apple without closing the path back to the tail.

    The network may pick a step only when that step is closer to the apple and
    the tail stays reachable. If no such step exists, the grid search takes
    over. A one-step tail check is not enough to finish a long snake; the
    search is what keeps that path open for the whole chase.
    """
    values = q_values(net, game, device)
    here = _food_distance(game.head, game.food)
    closer: list[tuple[float, Direction]] = []
    for action in range(3):
        direction = relative_to_absolute(game.heading, action)
        if not _keeps_tail(game, direction):
            continue
        nxt = game.next_cell(direction)
        distance = 0 if nxt == game.food else _food_distance(nxt, game.food)
        if distance < here:
            closer.append((float(values[action]), direction))
    if closer:
        closer.sort(key=lambda item: item[0], reverse=True)
        chosen = closer[0][1]
        weights = np.exp(np.array([item[0] for item in closer]) - closer[0][0])
        weights /= weights.sum()
        probs = {direction.value: 0.0 for direction in Direction}
        for weight, (_, direction) in zip(weights, closer):
            probs[direction.value] = float(weight)
        return chosen, probs, "network"
    move, probs = plan_move(game)
    return move, probs, "search"


def steer_move(net: QNet, game: SnakeGame, device: torch.device) -> tuple[Direction, dict[str, float]]:
    """The learned move among steps that do not die immediately."""
    values = q_values(net, game, device)
    ranked: list[tuple[float, Direction]] = []
    for action in range(3):
        direction = relative_to_absolute(game.heading, action)
        if game.would_die(direction):
            continue
        ranked.append((float(values[action]), direction))
    if not ranked:
        return dqn_move(net, game, device)
    ranked.sort(key=lambda item: item[0], reverse=True)
    chosen = ranked[0][1]
    weights = np.exp(np.array([item[0] for item in ranked]) - ranked[0][0])
    weights /= weights.sum()
    probs = {direction.value: 0.0 for direction in Direction}
    for weight, (_, direction) in zip(weights, ranked):
        probs[direction.value] = float(weight)
    return chosen, probs


def dqn_move(
    net: QNet,
    game: SnakeGame,
    device: torch.device,
    prior_beta: float = 0.0,
) -> tuple[Direction, dict[str, float]]:
    values = q_values(net, game, device)
    prior = normalized_prior()
    scored: list[tuple[float, int, Direction]] = []
    for action in range(3):
        direction = relative_to_absolute(game.heading, action)
        bonus = prior_beta * float(np.log(prior[direction.value]))
        scored.append((float(values[action]) + bonus, action, direction))
    scored.sort(key=lambda item: item[0], reverse=True)
    chosen = scored[0][2]
    weights = np.exp(np.array([item[0] for item in scored]) - scored[0][0])
    weights /= weights.sum()
    probs = {direction.value: 0.0 for direction in Direction}
    for weight, (_, _, direction) in zip(weights, scored):
        probs[direction.value] = float(weight)
    return chosen, probs
