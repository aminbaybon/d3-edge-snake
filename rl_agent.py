"""Grid observation and Double DQN policy for Snake.

Action 0 turns left, 1 goes straight, 2 turns right. That is the whole legal
action space: a reversal is not representable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from policy import relative_to_absolute
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
