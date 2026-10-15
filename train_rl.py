"""Train a Double DQN on the Snake grid and compare it with the d3-edge prior.

Double DQN is the suitable algorithm here: three discrete actions, a deterministic
fully observed grid, and a sparse apple. PPO would throw away each transition
after one update. Fine-tuning d3-edge itself does not fit in 4 GB, and one of
its decisions already costs about three seconds.
"""

from __future__ import annotations

import json
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from policy import greedy_move, plan_move, relative_to_absolute
from rl_agent import QNet, WEIGHTS, dqn_move, normalized_prior, observation_size, observe
from snake import SnakeGame

BOARD = 8
HIDDEN = 128
GAMMA = 0.99
LR = 1e-3
BATCH = 128
BUFFER = 20_000
TRAIN_STEPS = 200_000
WARMUP = 2_000
TARGET_EVERY = 1_000
EPS_END = 0.05
EPS_DECAY_STEPS = 150_000
EPISODE_CAP = 160
EVAL_EVERY = 25_000
EVAL_SEEDS = 20
FINAL_SEEDS = 30
FINAL_CAP = 250
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "run" / "rl_eval.json"


def manhattan(game: SnakeGame) -> int:
    hx, hy = game.head
    fx, fy = game.food
    return abs(fx - hx) + abs(fy - hy)


def shaped_reward(game: SnakeGame, before_len: int, before_dist: int, alive_before: bool) -> tuple[float, bool]:
    """Apple +1, death -1, otherwise a small step toward the current apple.

    A potential -distance must not be applied on the eat transition: the new
    apple is usually farther away, so γΦ(s')-Φ(s) is about -6 and cancels the
    apple. That run climbed to a mean score of about 3 and then died more often.
    """
    won = game.death == "win"
    died = alive_before and not game.alive and not won
    ate = game.length > before_len or won
    if ate:
        reward = 1.0
    elif died:
        reward = -1.0
    else:
        reward = 0.05 * float(before_dist - manhattan(game))
    return reward, died or won


def epsilon(step: int) -> float:
    if step >= EPS_DECAY_STEPS:
        return EPS_END
    frac = step / EPS_DECAY_STEPS
    return 1.0 + frac * (EPS_END - 1.0)


def pick_action(net: QNet, game: SnakeGame, device: torch.device, eps: float, prior_beta: float) -> int:
    if random.random() < eps:
        return random.randrange(3)
    direction, _ = dqn_move(net, game, device, prior_beta=prior_beta)
    for action in range(3):
        if relative_to_absolute(game.heading, action) is direction:
            return action
    return 1


def evaluate(policy: str, net: QNet | None, device: torch.device, seeds: int, cap: int, prior_beta: float = 0.0) -> dict[str, float]:
    scores: list[int] = []
    steps: list[int] = []
    deaths = 0
    wins = 0
    for seed in range(seeds):
        game = SnakeGame(size=BOARD, seed=seed)
        while game.alive and game.steps < cap:
            if policy == "plan":
                move, _ = plan_move(game)
            elif policy == "greedy":
                move = greedy_move(game)
            elif policy == "dqn":
                assert net is not None
                move, _ = dqn_move(net, game, device, prior_beta=prior_beta)
            else:
                raise ValueError(policy)
            game.step(move)
        scores.append(game.score)
        steps.append(game.steps)
        if game.death == "win":
            wins += 1
        elif not game.alive:
            deaths += 1
    return {
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "max_score": float(np.max(scores)),
        "mean_steps": float(np.mean(steps)),
        "death_rate": deaths / seeds,
        "win_rate": wins / seeds,
        "seeds": seeds,
        "step_cap": cap,
    }


def train(prior_beta: float, label: str) -> tuple[QNet, list[dict[str, float | int | str]]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    online = QNet(BOARD, HIDDEN).to(device)
    target = QNet(BOARD, HIDDEN).to(device)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=LR)
    loss_fn = nn.SmoothL1Loss()
    buffer: deque[tuple[np.ndarray, int, float, np.ndarray, float]] = deque(maxlen=BUFFER)
    curve: list[dict[str, float | int | str]] = []
    game = SnakeGame(size=BOARD, seed=0)
    obs = observe(game)
    started = time.perf_counter()
    episode_return = 0.0
    print(f"training {label} on {device}", flush=True)

    for step in range(1, TRAIN_STEPS + 1):
        # The prior is a reward bonus only. Action selection stays epsilon-greedy,
        # so a bad teacher cannot also freeze the behavior policy on "right".
        action = pick_action(online, game, device, epsilon(step), 0.0)
        before_len = game.length
        before_dist = manhattan(game)
        direction = relative_to_absolute(game.heading, action)
        game.step(direction)
        reward, terminal = shaped_reward(game, before_len, before_dist, True)
        if prior_beta > 0:
            reward += prior_beta * float(np.log(normalized_prior()[direction.value]))
        episode_return += reward
        # A step cap is a timeout, not a death, so the backup still bootstraps.
        nxt = observe(game)
        buffer.append((obs, action, reward, nxt, float(terminal)))
        obs = nxt
        if terminal or game.steps >= EPISODE_CAP or not game.alive:
            game.reset(random.randrange(1_000_000))
            obs = observe(game)
            episode_return = 0.0

        if step >= WARMUP:
            batch = random.sample(buffer, BATCH)
            states = torch.from_numpy(np.stack([item[0] for item in batch])).to(device)
            actions = torch.tensor([item[1] for item in batch], device=device, dtype=torch.int64)
            rewards = torch.tensor([item[2] for item in batch], device=device)
            next_states = torch.from_numpy(np.stack([item[3] for item in batch])).to(device)
            dones = torch.tensor([item[4] for item in batch], device=device)
            q = online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_actions = online(next_states).argmax(dim=1)
                next_q = target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                backup = rewards + GAMMA * next_q * (1.0 - dones)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(q, backup)
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), 5.0)
            optimizer.step()

        if step % TARGET_EVERY == 0:
            target.load_state_dict(online.state_dict())

        if step % EVAL_EVERY == 0:
            online.eval()
            stats = evaluate("dqn", online, device, EVAL_SEEDS, cap=120, prior_beta=0.0)
            online.train()
            stats["step"] = step
            stats["label"] = label
            stats["seconds"] = round(time.perf_counter() - started, 1)
            curve.append(stats)
            print(
                f"  {label} step {step}: mean score {stats['mean_score']:.2f} "
                f"deaths {stats['death_rate']:.2f} ({stats['seconds']}s)",
                flush=True,
            )

    online.eval()
    return online, curve


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plain, plain_curve = train(0.0, "dqn")
    mixed, mixed_curve = train(0.5, "dqn+d3-prior")
    torch.save(
        {
            "state_dict": plain.state_dict(),
            "board": BOARD,
            "hidden": HIDDEN,
            "observation_size": observation_size(BOARD),
            "algorithm": "double-dqn",
            "train_steps": TRAIN_STEPS,
        },
        WEIGHTS,
    )
    final = {
        "board": BOARD,
        "train_steps": TRAIN_STEPS,
        "algorithm": "double dqn, relative 3-action space, apple +1, death -1, distance delta 0.05 except on the eat step",
        "failed_shaping": {
            "note": "gamma*Phi(next)-Phi with Phi=-manhattan, including the food respawn, on a flattened grid",
            "curve_mean_score": [2.2, 3.1, 3.0, 3.3],
            "curve_death_rate": [0.70, 0.70, 0.80, 0.90],
            "steps": [25000, 50000, 75000, 100000],
        },
        "failed_grid_with_fixed_reward": {
            "note": "same MLP on a flattened 8x8 grid, apple +1, but the eat step no longer resets the potential",
            "curve_mean_score": [2.10, 2.15, 2.40],
            "curve_death_rate": [0.70, 0.60, 0.75],
            "steps": [25000, 50000, 75000],
        },
        "device": str(device),
        "d3_prior": normalized_prior(),
        "d3_prior_source": "mean of the four decisions in the seed-7 wall crash",
        "policies": {
            "d3-edge": {
                "mean_score": 0.0,
                "median_score": 0.0,
                "max_score": 0.0,
                "mean_steps": 4.0,
                "death_rate": 1.0,
                "win_rate": 0.0,
                "seeds": 1,
                "step_cap": 20,
                "note": "seed 7, size 8: right, right, right, right, wall. 3.0–3.2 s/step",
            },
            "greedy": evaluate("greedy", None, device, FINAL_SEEDS, FINAL_CAP),
            "double-dqn": evaluate("dqn", plain, device, FINAL_SEEDS, FINAL_CAP, prior_beta=0.0),
            "double-dqn+d3-prior-at-decision": evaluate("dqn", plain, device, FINAL_SEEDS, FINAL_CAP, prior_beta=1.0),
            "double-dqn-trained-with-d3-reward": evaluate("dqn", mixed, device, FINAL_SEEDS, FINAL_CAP, prior_beta=0.0),
            "search": evaluate("plan", None, device, FINAL_SEEDS, FINAL_CAP),
        },
        "learning_curve_dqn": plain_curve,
        "learning_curve_dqn_with_d3_reward": mixed_curve,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(final["policies"], ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
