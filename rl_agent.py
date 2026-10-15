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
